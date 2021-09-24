import numpy as np
import xarray as xr
import yaml
import json

from parflowio.pyParflowio import PFData
from xarray.backends  import BackendEntrypoint

class ParflowBackendEntrypoint(BackendEntrypoint):

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        name='parflow_variable',
        meta_yaml=None,
        read_inputs=False, # TODO: This is currently broken!
        read_outputs=True,
    ):

        filetype = self.is_meta_or_pfb(filename_or_obj)
        if filetype == 'pfb':
            da = self.load_single_pfb(filename_or_obj)
            ds = da.to_dataset()
        elif filetype == 'pfmetadata':
            ds = xr.Dataset()
            ds.attrs['parflow_version'] = self.pf_meta['parflow']['build']['version']
            if read_outputs:
                for var, var_meta in self.pf_meta['outputs'].items():
                    ds[var] = self.load_pfb_from_meta(var_meta)
            if read_inputs:
                for var, var_meta in self.pf_meta['inputs'].items():
                    if var == 'configuration':
                        continue # TODO: Determine what to do with this
                    ds[var] = self.load_pfb_from_meta(var_meta)

        #TODO : Set name and other stuff as necessary (maybe coordinate transforms)?
        if meta_yaml:
            meta = self.process_meta(ds, yaml)
        return ds

    def is_meta_or_pfb(self, filename_or_obj):

        def _check_dict_is_valid_meta(meta):
            assert 'parflow' in meta.keys(), \
                ('Metadata file missing "parflow" key - ',
                 'are you sure this is a valid Parflow metadata file?')

        if isinstance(filename_or_obj, str):
            try:
                pfd = PFData(filename_or_obj)
                stat = pfd.loadHeader()
                assert stat == 0
                stat = pfd.loadData()
                assert stat == 0
                return 'pfb'
            except AssertionError:
                with open(filename_or_obj, 'r') as f:
                    pf_meta = json.load(f)
                    _check_dict_is_valid_meta(pf_meta)
                    self.pf_meta = pf_meta
                    return 'pfmetadata'
        elif isinstance(filename_or_obj, dict):
            _check_dict_is_valid_meta(filename_or_obj)
            self.pf_meta = filename_or_obj
            return 'pfmetadata'
        else:
            raise NotImplementedError("Was unable to determine input type!")

    def load_yaml_meta(self, path):
        """
        Load a Parflow `yaml` file
        """
        with open(path, 'r') as f:
            self.meta_yaml = yaml.load(f)
        raise NotImplementedError('')

    def load_single_pfb(
            self,
            filename_or_obj,
            name='parflow_variable',
            dims=None,
            coords=None
    ) -> xr.DataArray:
        """
        Load a `pfb` file directly as an xr.DataArray
        """
        pfd = PFData(filename_or_obj)
        stat = pfd.loadHeader()
        assert stat == 0
        stat = pfd.loadData()
        assert stat == 0
        arr = pfd.viewDataArray()
        if not dims:
            dims = list(pfd.getIndexOrder())
        if not coords:
            coords = self.decode_coords(pfd)
        assert sorted(dims) == sorted(list(coords.keys())), \
                (f"Mismatch in dims and coord names on file {filename_or_obj}!",
                 f"dims: {dims}, coords: {coords}")

        da = xr.DataArray(arr, coords=coords, dims=dims, name=name)
        return da

    def load_pfb_from_meta(self, var_meta):
        """
        Load a pfb file or set of pfb files from the metadata

        Parameters
        ----------
        var_meta: dict
            A dictionary which tells us how to read the data
        """
        assert var_meta['type'] == 'pfb', "Can't load non-pfb data!"
        if var_meta.get('time-varying', None):
            # Note: The way that var_meta['data'] is aranged is idiosyncratic:
            #       It is a list with a single dictionary inside - check if this
            #       is always the case
            time_idx = np.arange(*var_meta['data'][0]['time-range'])
            file_template = var_meta['data'][0]['file-series']
            pad, fmt = file_template.split('.')[-2:]
            basepath = '.'.join(file_template.split('.')[:-2])
            all_files = [f'{basepath}.{pad%n}.{fmt}' for n in time_idx]
            base_da = xr.concat([self.load_single_pfb(f) for f in all_files], dim='time')
            base_da.attrs['units'] = var_meta.get('units', 'not_specified')
            return base_da
        else:
            raise NotImplementedError('Currently only support for reading time-varying data!')

    def decode_coords(self, pfd: PFData, dims=['x', 'y', 'z']):
        """
        Decodes coordinates.

        Parameters
        ----------
        pfd: PFData
            The Parflow Data object
        dims: list (unused currently)
            A list of dimensions to decode

        Returns
        -------
        coords: dict
            Coorinates to pass to the xarray
            datastructure constructor
        """
        x_start, y_start, z_start = pfd.getX(), pfd.getY(), pfd.getZ()
        nx, ny, nz = pfd.getNX(), pfd.getNY(), pfd.getNZ()
        dx, dy, dz = pfd.getDX(), pfd.getDY(), pfd.getDZ()
        coords = {'x': dx * np.arange(0, nx) + x_start,
                  'y': dy * np.arange(0, ny) + y_start,
                  'z': dz * np.arange(0, nz) + z_start, }
        return coords


@xr.register_dataset_accessor("parflow")
class ParflowAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_pfb(self):
        raise NotImplementedError('coming soon!')
