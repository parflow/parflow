import contextlib
import json
import numpy as np
import os
import warnings
import xarray as xr
import yaml

from parflow.tools import hydrology
from parflowio.pyParflowio import PFData
from xarray.backends  import BackendEntrypoint, BackendArray
from xarray.backends.locks import SerializableLock
from xarray.core import indexing

PARFLOW_LOCK = SerializableLock()
NO_LOCK = contextlib.nullcontext()

class ParflowBackendEntrypoint(BackendEntrypoint):

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        name='parflow_variable',
        meta_yaml=None,
        read_inputs=True,
        read_outputs=True,
    ):

        filetype = self.is_meta_or_pfb(filename_or_obj)
        if filetype == 'pfb':
            # Reads a single pfb
            da = self.load_single_pfb(filename_or_obj, name=name)
            ds = da.to_dataset()
        elif filetype == 'pfmetadata':
            # Reads full simulation input/output from pfmetadata
            self.base_dir = os.path.dirname(filename_or_obj)
            ds = xr.Dataset()
            ds.attrs['pf_metadata_file'] = filename_or_obj
            ds.attrs['parflow_version'] = self.pf_meta['parflow']['build']['version']
            if read_outputs:
                for var, var_meta in self.pf_meta['outputs'].items():
                    ds[var] = self.load_pfb_from_meta(var_meta)
            if read_inputs:
                for var, var_meta in self.pf_meta['inputs'].items():
                    if var == 'configuration':
                        continue # TODO: Determine what to do with this
                    if len(var_meta['data']) == 1:
                        ds[var] = self.load_pfb_from_meta(var_meta)
                    else:
                        for sub_dict in var_meta['data']:
                            component = sub_dict['component']
                            ds[f'{var}_{component}'] = self.load_pfb_from_meta(var_meta, component)

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
    ) -> xr.DataArray:
        """
        Load a `pfb` file directly as an xr.DataArray
        """
        backend_array = ParflowBackendArray(filename_or_obj)
        data = indexing.LazilyIndexedArray(backend_array)
        var = xr.Variable(backend_array.dims, data)
        da = xr.DataArray(var, name=name)
        return da

    def load_pfb_from_meta(self, var_meta, component=None):
        """
        Load a pfb file or set of pfb files from the metadata

        Parameters
        ----------
        var_meta: dict
            A dictionary which tells us how to read the data
        component: Optional[str]
            An optional component for anisotropic fields
        """
        base_da = xr.DataArray()
        ALLOWED_TYPES = ['pfb', 'pfb 2d timeseries']
        pfb_type = var_meta['type']
        assert pfb_type in ALLOWED_TYPES, "Can't load non-pfb data!"
        if var_meta.get('time-varying', None):
            # Note: The way that var_meta['data'] is aranged is idiosyncratic:
            #       It is a list with a single dictionary inside - check if this
            #       is always the case
            file_template = var_meta['data'][0]['file-series']
            if pfb_type == 'pfb':
                time_idx = np.arange(*var_meta['data'][0]['time-range'])
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = sorted([f'{basename}.{pad%n}.{fmt}' for n in time_idx])
            elif pfb_type == 'pfb 2d timeseries':
                time_start = np.arange(*var_meta['data'][0]['times-between'])
                time_end = time_start + var_meta['data'][0]['times-between'][-1] - 1
                file_template = var_meta['data'][0]['file-series']
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = [f'{basename}.{pad%(s,e)}.{fmt}'
                             for s, e in zip(time_start, time_end)]

            # Check if basname contains any of the files if not,
            # fall back to `self.base_dir` from the pfmetadata file
            if not os.path.exists(all_files[0]):
                all_files = sorted([f'{self.base_dir}/{af}' for af in all_files])

            # Put it all together
            # NOTE: This will have to be changed to support lazy loading/indexing
            # See here for discussion: https://github.com/pydata/xarray/issues/4628
            base_da = xr.open_mfdataset(
                    all_files, concat_dim='time', combine='nested')['parflow_variable']

            if pfb_type == 'pfb 2d timeseries':
                base_da = (base_da.rename({'time':'junk'})
                                  .stack(time=['junk', 'z']))
                base_da = base_da.assign_coords({'time': np.arange(len(base_da['time']))})

        elif component:
            for sub_dict in var_meta['data']:
                if sub_dict['component'] == component:
                    file = sub_dict['file']
                    if not os.path.exists(file):
                        file = f'{self.base_dir}/{file}'
                    base_da = self.load_single_pfb(file).squeeze()
                    break
        elif 'data' in var_meta:
            file = var_meta['data'][0]['file']
            if not os.path.exists(file):
                file = f'{self.base_dir}/{file}'
            base_da = self.load_single_pfb(file).squeeze()
        else:
            msg = f"Currently can't support for reading for {var_meta}"
            warnings.warn(msg)

        base_da.attrs['units'] = var_meta.get('units', 'not_specified')
        return base_da


    def guess_can_open(self, filename_or_obj):
        openable_extensions = ['pfb', 'pfmetadata', 'pbidb']
        for ext in openable_extensions:
            if filename_or_obj.endswith(ext):
                return True
        return False


class ParflowBackendArray(BackendArray):
    """
    This is a note to myself: ParflowBackendArray's are
    inherently spatial and of a single time slice. If we
    are interested in lazily loading and allowing for out
    of core computation we'll need to map time slices to
    file names in the higher level components.
    (ParflowBackendEntrypoint, most likely)

    That means that in the constructor the filename will be
    required. I'm not sure if that means that we can interpret
    the shape internally here though, but that might clean up
    the higher level code.
    """

    def __init__(self, filename_or_obj, lock=None):
        if lock in (True, None):
            lock = PARFLOW_LOCK
        elif lock is False:
            lock = NO_LOCK
        self.lock = lock
        self.filename_or_obj = filename_or_obj
        self._shape = None
        self._dims = None

    def __getitem__(
            self, key: xr.core.indexing.ExplicitIndexer
    ) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        with self.lock:
            pfd = PFData(self.filename_or_obj)
            stat = pfd.loadHeader()
            assert stat == 0, 'Failed to load header in ParflowBackendArray!'
            stat = pfd.loadData()
            assert stat == 0, 'Failed to load data in ParflowBackendArray!'
            sub = pfd.copyDataArray()[key]
            pfd.close()
            return sub

    @property
    def dims(self):
        if self._dims is None:
            pfd = PFData(self.filename_or_obj)
            stat = pfd.loadHeader()
            assert stat == 0, 'Failed to load header in ParflowBackendArray!'
            self._dims = list(pfd.getIndexOrder())
            pfd.close()
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            pfd = PFData(self.filename_or_obj)
            stat = pfd.loadHeader()
            assert stat == 0, 'Failed to load header in ParflowBackendArray!'
            accessor_mapping = {'x': pfd.getNX(),
                                'y': pfd.getNY(),
                                'z': pfd.getNZ() }
            self._shape = [accessor_mapping[d] for d in self.dims]
            pfd.close()
        return self._shape

    @property
    def dtype(self):
        return np.float64


@xr.register_dataset_accessor("parflow")
class ParflowAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.hydrology = hydrology

    def to_pfb(self):
        raise NotImplementedError('coming soon!')
