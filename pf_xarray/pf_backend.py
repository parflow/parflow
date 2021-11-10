import contextlib
import dask
import json
import numpy as np
import os
import warnings
import xarray as xr
import yaml

from . import util
from .io import ParflowBinaryReader, read_stack_of_pfbs, read_pfb
from collections.abc import Iterable
from dask import delayed
from parflow.tools import hydrology
from parflowio.pyParflowio import PFData
from typing import Mapping, List, Union
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
        read_inputs=True,
        read_outputs=True,
        parallel=False,
        inferred_dims=None,
        inferred_shape=None,
        chunks={},
        strict_ext_check=False,
    ):
        filetype = self.is_meta_or_pfb(filename_or_obj, strict=strict_ext_check)
        if filetype == 'pfb':
            # Reads a single pfb
            data = self.load_single_pfb(
                      filename_or_obj,
                      dims=inferred_dims,
                      shape=inferred_shape)
            ds = xr.Dataset({name: data}).chunk(chunks)
        elif filetype == 'pfmetadata':
            # Reads full simulation input/output from pfmetadata
            if base_dir:
                self.base_dir = base_dir
            else:
                self.base_dir = os.path.dirname(filename_or_obj)
            ds = self.load_pfmetadata(
                    filename_or_obj,
                    self.pf_meta,
                    read_inputs=read_inputs,
                    read_outputs=read_outputs,
                    parallel=parallel
            )

    def load_pfmetadata(
        self,
        filename_or_obj,
        pf_meta,
        read_inputs,
        read_outputs,
        parallel
    ):
        ds = xr.Dataset()
        ds.attrs['pf_metadata_file'] = filename_or_obj
        ds.attrs['parflow_version'] = pf_meta['parflow']['build']['version']
        if read_outputs:
            for var, var_meta in self.pf_meta['outputs'].items():
                if read_outputs is True or var in read_outputs:
                    var_type = ''
                    ds[var] = self.load_pfb_from_meta(var_meta, parallel=parallel)
        if read_inputs:
            for var, var_meta in self.pf_meta['inputs'].items():
                if var == 'configuration':
                    continue # TODO: Determine what to do with this
                if read_inputs is True or var in read_inputs:
                    var_type = ''
                    if len(var_meta['data']) == 1:
                        v = self.load_pfb_from_meta(var_meta)
                        ds[var] = v
                    else:
                        for sub_dict in var_meta['data']:
                            component = sub_dict['component']
                            ds[f'{var}_{component}'] = self.load_pfb_from_meta(
                                    var_meta, component, parallel=parallel)
        return ds

    def is_meta_or_pfb(self, filename_or_obj, strict=True):

        def _check_dict_is_valid_meta(meta):
            assert 'parflow' in meta.keys(), \
                ('Metadata file missing "parflow" key - ',
                 'are you sure this is a valid Parflow metadata file?')
        if not strict:
            ext = filename_or_obj.split('.')[-1]
            if ext == 'pfmetadata':
                with open(filename_or_obj, 'r') as f:
                    pf_meta = json.load(f)
                    _check_dict_is_valid_meta(pf_meta)
                    self.pf_meta = pf_meta
            return ext

        if isinstance(filename_or_obj, str):
            try:
                with ParflowBinaryReader(filename_or_obj) as pfd:
                    assert 'nx' in pfd.header
                    assert 'ny' in pfd.header
                    assert 'nz' in pfd.header
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

    def _infer_dims_and_shape(self, file):
        # TODO: Figure out how to use this
        pfd = xr.Dataset({'_': self.load_single_pfb(file)})
        dims = list(pfd.dims.keys())
        shape = list(pfd.dims.values())
        del pfd
        return dims, shape

    def load_single_pfb(
            self,
            filename_or_obj,
            dims=None,
            shape=None
    ) -> xr.DataArray:
        """
        Load a `pfb` file directly as an xr.DataArray
        """
        if not dims:
            dims = ('x', 'y', 'z')
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(filename_or_obj, dims=dims, shape=shape))
        var = xr.Variable(dims, data, )
        return var

    def load_stack_of_pfb(
        self,
        filenames,
        dims=None,
        shape=None
    ):
        if not dims:
            dims = ('time', 'x', 'y', 'z')
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(filenames, dims=dims, shape=shape))
        var = xr.Variable(dims, data)
        return var

    def load_component_pfb(self):
        """
        These filetypes have dimensions (component, x, y, z)
        where component represents an anisotropy
        """
        pass

    def load_time_varying_pfb(self):
        """
        THese filetypes have dimensions (time, x, y, z)
        where a each file represents an individual time
        """
        pass

    def load_time_varying_2d_ts_pfb(self):
        """
        These filetypes have dimensions (time_stride, x, y, time_slice)
        where the time dimension will be strided along separate files
        and each individual file contains time_slice number of timesteps
        """
        pass

    def load_clm_output_pfb(self):
        """
        These filetypes have dimensions (time, x, y, variable)
        where the variable ordering is fixed and each file represents an
        individual timestep
        """
        pass


    def load_pfb_from_meta(self, var_meta, parallel=False):
        type_to_loader_fun = {
            'pfb': self.load_time_varying_pfb,
            'pfb 2d timeseries': self.load_time_varying_2d_ts_pfb,
            'pfb_components': self.load_component_pfb,
            'clm output': self.load_clm_output_pfb,
        }
        ALLOWED_TYPES = list(type_to_loader_fun.keys())
        base_type = var_meta['type']
        if base_type == 'pfb':
            # Is it component?
            if len(var_meta['data']) > 1 and 'component' in var_meta['data'][0]:
                ret_das = self.load_component_pfb(var_meta)
            # Is it clm output?
            elif var_meta['data']:
                ret_das = self.load_clm_output_pfb(var_meta)
            # Is it normal?
            else:
                ret_das = self.load_time_varying_pfb(var_meta)
        elif base_type == 'pfb 2d timeseries':
            ret_das = self.load_time_varying_2d_ts_pfb(var_meta)
        return ret_das



    def load_pfb_from_meta_old(self, var_meta, component=None, parallel=False):
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
                concat_dim = 'time'
                time_idx = np.arange(*var_meta['data'][0]['time-range'])
                n_time = time_idx[-1]
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = [f'{basename}.{pad%n}.{fmt}' for n in time_idx]
            elif pfb_type == 'pfb 2d timeseries':
                concat_dim = 'z' # z is time here
                time_start = np.arange(*var_meta['data'][0]['times-between'])
                time_end = time_start + var_meta['data'][0]['times-between'][-1] - 1
                ntime = time_end[-1]
                file_template = var_meta['data'][0]['file-series']
                pad, fmt = file_template.split('.')[-2:]
                basename = '.'.join(file_template.split('.')[:-2])
                all_files = [f'{basename}.{pad%(s,e)}.{fmt}'
                             for s, e in zip(time_start, time_end)]

            # Check if basename contains any of the files if not,
            # fall back to `self.base_dir` from the pfmetadata file
            if not os.path.exists(all_files[0]):
                all_files = [f'{self.base_dir}/{af}' for af in all_files]

            # Put it all together
            inf_dims, inf_shape = self._infer_dims_and_shape(all_files[0])
            inf_dims = ('time', *inf_dims)
            inf_shape = (len(all_files), *inf_shape)
            base_da = self.load_stack_of_pfb(
                    all_files, dims=inf_dims, shape=inf_shape)
            base_da = xr.Dataset({'_': base_da})['_']
            if pfb_type == 'pfb 2d timeseries':
                base_da = base_da.stack({'time':['time', 'z']})

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


@delayed
def _getitem_no_state(file_or_seq, key, mode, z_first=False, z_is='z'):
    if mode == 'single':
        accessor = {d: util._key_to_explicit_accessor(k)
                    for d, k in zip(['x','y','z'], key)}
        if z_first:
            d = ['z', 'y', 'x']
        else:
            d = ['x', 'y', 'z']

        with ParflowBinaryReader(file_or_seq) as pfd:
            sub = pfd.read_subarray(
                start_x=int(accessor['x']['start']),
                start_y=int(accessor['y']['start']),
                start_z=int(accessor['z']['start']),
                nx=int(accessor['x']['stop']),
                ny=int(accessor['y']['stop']),
                nz=int(accessor['z']['stop']),
                z_first=z_first
            )
        sub = sub[accessor[d[0]]['indices'],
                  accessor[d[1]]['indices'],
                  accessor[d[2]]['indices']]#.squeeze()
    elif mode == 'sequence':
        accessor = {d: util._key_to_explicit_accessor(k)
                    for d, k in zip(['time', 'x','y','z'], key)}
        t_start = accessor['time']['start']
        t_end = accessor['time']['stop'] - 1
        if t_start == t_end:
            t_end += 1
        sub = read_stack_of_pfbs(
            file_or_seq[t_start:t_end],
            key=accessor,
            z_first=z_first,
        )
        sub = sub[accessor['time']['indices'],
                  accessor['x']['indices'],
                  accessor['y']['indices'],
                  accessor['z']['indices']]
        if z_is == 'time':
            if z_first:
                sub = np.concatenate(sub, axis=0)
            else:
                sub = np.concatenate(sub, axis=-1)
        sub = sub.squeeze()
    return sub

@delayed
def _get_item_no_state_2d_timeseries(file_or_seq, key, mode):
    pass


class ParflowBackendArray(BackendArray):

    def __init__(
         self,
         file_or_seq,
         dims=None,
         shape=None,
    ):
        self.file_or_seq = file_or_seq
        if isinstance(self.file_or_seq, str):
            self.mode = 'single'
            self.header_file = self.file_or_seq
        elif isinstance(self.file_or_seq, Iterable):
            self.mode = 'sequence'
            self.header_file = self.file_or_seq[0]
        self._shape = shape
        self._dims = dims
        # Weird hack here, have to pull the dtype like this
        # to have valid `nbytes` attribute
        self.dtype = np.dtype(np.float64)

    def __getitem__(
            self, key: xr.core.indexing.ExplicitIndexer
    ) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )

    @property
    def dims(self):
        if self._dims is None:
            if self.mode == 'single':
                self._dims = ('x', 'y', 'z')
            elif self.mode == 'sequence':
                self._dims = ('time', 'x', 'y', 'z')
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            with ParflowBinaryReader(
                self.header_file,
                precompute_subgrid_info=False
            ) as pfd:
                base_shape = (pfd.header['nx'], pfd.header['ny'], pfd.header['nz'])
            if self.mode == 'sequence':
                base_shape = (len(self.file_or_seq), *base_shape)
            # Add some logic for automatically squeezing here?
            self._shape = base_shape
        return self._shape

    def _getitem(self, key: tuple) -> np.typing.ArrayLike:
        size = self._size_from_key(key)
        key = self._explicit_indices_from_keys(size, key)
        sub = delayed(_getitem_no_state)(self.file_or_seq, key, self.mode)
        sub = dask.array.from_delayed(sub, size, dtype=np.float64)
        return sub

    def _explicit_indices_from_keys(self, size , key):
        ret_key = []
        for dim_size, dim_key in zip(size, key):
            if isinstance(dim_key, slice):
                start = dim_key.start if dim_key.start is not None else 0
                stop = dim_key.stop+1 if dim_key.stop is not None else dim_size+1
                step = dim_key.step if dim_key.step is not None else 1
                ret_key.append(slice(start, stop, step))
            else:
                # Do nothing here
                ret_key.append(dim_key)
        return tuple(ret_key)

    def _size_from_key(self, key):
        ret_size = []
        for i, k in enumerate(key):
            if isinstance(k, slice):
                if util._check_key_is_empty([k]):
                    ret_size.append(self.shape[i])
                else:
                    ret_size.append(len(np.arange(k.start, k.stop, k.step)))
            elif isinstance(k, Iterable):
                ret_size.append(len(k))
            else:
                ret_size.append(1)
        return ret_size

    def get_dims(self):
        dims = ('x', 'y', 'z')
        return dims

    def get_shape(self):
        with ParflowBinaryReader(
            self.header_file,
            precompute_subgrid_info=False
        ) as pfd:
            base_shape = (pfd.header['nx'], pfd.header['ny'], pfd.header['nz'])
        # Add some logic for automatically squeezing here?
        return base_shape


#@xr.register_dataset_accessor("parflow")
#class ParflowAccessor:
#    def __init__(self, xarray_obj):
#        self._obj = xarray_obj
#        self.hydrology = hydrology
#
#    def to_pfb(self):
#        raise NotImplementedError('coming soon!')
