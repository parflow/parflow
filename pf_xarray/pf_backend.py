import contextlib
import json
import numpy as np
import os
import warnings
import xarray as xr
import yaml

from parflow.tools import hydrology
from parflowio.pyParflowio import PFData
from xarray.backends  import BackendEntrypoint, BackendArray, CachingFileManager
from xarray.backends.locks import SerializableLock
from xarray.core import indexing

PARFLOW_LOCK = SerializableLock()
NO_LOCK = contextlib.nullcontext()

class ParflowBackendEntrypoint(BackendEntrypoint):

    open_dataset_parameters = [
            "filename_or_obj",
            "drop_variables",
            "name",
            "meta_yaml",
            "read_inputs",
            "read_outputs",
            "parallel",
            "inferred_dims",
            "inferred_shape"
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        base_dir=None,
        drop_variables=None,
        name='parflow_variable',
        meta_yaml=None,
        read_inputs=True,
        read_outputs=True,
        parallel=False,
        inferred_dims=None,
        inferred_shape=None,
    ):
        filetype = self.is_meta_or_pfb(filename_or_obj)
        if filetype == 'pfb':
            # Reads a single pfb
            data = self.load_single_pfb(
                      filename_or_obj,
                      dims=inferred_dims,
                      shape=inferred_shape)
            ds = xr.DataArray(data, name=name).to_dataset()
        elif filetype == 'pfmetadata':
            # Reads full simulation input/output from pfmetadata
            if base_dir:
                self.base_dir = base_dir
            else:
                self.base_dir = os.path.dirname(filename_or_obj)
            ds = xr.Dataset()
            ds.attrs['pf_metadata_file'] = filename_or_obj
            ds.attrs['parflow_version'] = self.pf_meta['parflow']['build']['version']
            if read_outputs:
                for var, var_meta in self.pf_meta['outputs'].items():
                    if read_outputs is True or var in read_outputs:
                        ds[var] = self.load_pfb_from_meta(var_meta, parallel=parallel)
            if read_inputs:
                for var, var_meta in self.pf_meta['inputs'].items():
                    if var == 'configuration':
                        continue # TODO: Determine what to do with this
                    if read_inputs is True or var in read_inputs:
                        if len(var_meta['data']) == 1:
                            ds[var] = self.load_pfb_from_meta(var_meta)
                        else:
                            for sub_dict in var_meta['data']:
                                component = sub_dict['component']
                                ds[f'{var}_{component}'] = self.load_pfb_from_meta(
                                        var_meta, component, parallel=parallel)

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
                f = ParflowData(filename_or_obj)
                f.close()
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

    def _infer_dims_and_shape(self, file):
        # TODO: Figure out how to use this
        pfd = xr.DataArray(self.load_single_pfb(file), name='_').to_dataset()
        dims = list(pfd.dims.keys())
        shape = list(pfd.dims.values())
        del pfd
        return dims, shape

    def load_single_pfb(
            self,
            filename_or_obj,
            lock=None,
            dims=None,
            shape=None
    ) -> xr.DataArray:
        """
        Load a `pfb` file directly as an xr.DataArray
        """
        if lock in (True, None):
            lock = PARFLOW_LOCK
        elif lock is False:
            lock = NO_LOCK
        manager = CachingFileManager(ParflowData, filename_or_obj)
        if not dims:
            dims = ('z', 'y', 'x')
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(manager, lock=lock, dims=dims, shape=shape))
        var = xr.Variable(dims, data)
        return var

    def load_pfb_from_meta(self, var_meta, component=None, parallel=False):
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
            n_time = 0
            if pfb_type == 'pfb':
                time_idx = np.arange(*var_meta['data'][0]['time-range'])
                n_time = time_idx[-1]
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = [f'{basename}.{pad%n}.{fmt}' for n in time_idx]
            elif pfb_type == 'pfb 2d timeseries':
                time_start = np.arange(*var_meta['data'][0]['times-between'])
                time_end = time_start + var_meta['data'][0]['times-between'][-1] - 1
                ntime = time_end[-1]
                file_template = var_meta['data'][0]['file-series']
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = [f'{basename}.{pad%(s,e)}.{fmt}'
                             for s, e in zip(time_start, time_end)]

            # Check if basname contains any of the files if not,
            # fall back to `self.base_dir` from the pfmetadata file
            if not os.path.exists(all_files[0]):
                all_files = [f'{self.base_dir}/{af}' for af in all_files]

            # Put it all together
            # NOTE: This will have to be changed to support lazy loading/indexing
            # See here for discussion: https://github.com/pydata/xarray/issues/4628
            inf_dims, inf_shape = self._infer_dims_and_shape(all_files[0])
            base_da = xr.open_mfdataset(
                    all_files,
                    engine='parflow',
                    concat_dim='time',
                    combine='nested',
                    decode_cf=False,
                    inferred_dims=inf_dims,
                    inferred_shape=inf_shape,
                    )['parflow_variable']

            if pfb_type == 'pfb 2d timeseries':
                base_da = (base_da.rename({'time':'junk'})
                                  .stack(time=['junk', 'z'])
                                  .drop('time'))

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


class ParflowData:

    def __init__(self, filename):
        self.filename = filename
        self.pfd = PFData(self.filename)
        stat = self.pfd.loadHeader()
        assert stat == 0, 'Failed to load header in ParflowData!'
        self.dims = self.get_dims()
        self.shape = self.get_shape()

    def __enter__(self):
        return self.pfd

    def __exit__(self):
        self.pfd.unloadData()
        self.pfd.close()
        del self.pfd

    def close(self):
        self.pfd.unloadData()
        self.pfd.close()
        del self.pfd

    def get_dims(self):
        return list(self.pfd.getIndexOrder())

    def get_shape(self):
        accessor_mapping = {'x': self.pfd.getNX,
                            'y': self.pfd.getNY,
                            'z': self.pfd.getNZ}
        return  [accessor_mapping[d]() for d in self.get_dims()]


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

    def __init__(
         self,
         manager,
         lock,
         dims=None,
         shape=None
    ):
        self.manager = manager
        self.lock = lock
        self._shape = shape
        self._dims = dims

    def __getitem__(
            self, key: xr.core.indexing.ExplicitIndexer
    ) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        with self.lock:
            f = self.manager.acquire(needs_lock=False)
            stat = f.pfd.loadData()
            assert stat == 0, 'Failed to load data in ParflowBackendArray!'
            sub = f.pfd.copyDataArray()[key]
            f.pfd.unloadData()
            return sub

    @property
    def dims(self):
        if self._dims is None:
            self._dims = self.manager.acquire().dims
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.manager.acquire().shape
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
