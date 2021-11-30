import contextlib
import dask
import json
import numpy as np
import os
import warnings
import xarray as xr
import yaml

from pprint import pprint
from . import util
from .io import ParflowBinaryReader, read_stack_of_pfbs, read_pfb
from collections.abc import Iterable
from dask import delayed
#from parflow.tools import hydrology
#from parflowio.pyParflowio import PFData
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
        return ds

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
                    das = self.load_pfb_from_meta(var_meta, name=var, parallel=parallel)
                    for k, v in das.items():
                        ds[k] = v
        if read_inputs:
            for var, var_meta in self.pf_meta['inputs'].items():
                if var == 'configuration':
                    continue # TODO: Determine what to do with this
                if read_inputs is True or var in read_inputs:
                    das = self.load_pfb_from_meta(var_meta, name=var, parallel=parallel)
                    for k, v in das.items():
                        ds[k] = v
        return ds

    def is_meta_or_pfb(self, filename_or_obj, strict=True):

        def _check_dict_is_valid_meta(meta):
            assert 'parflow' in meta.keys(), \
                ('Metadata file missing "parflow" key - ',
                 'are you sure this is a valid Parflow metadata file?')
        if not strict:
            # Just check the extension
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

    def _infer_dims_and_shape(self, file, z_first=True, z_is='z'):
        # TODO: Figure out how to use this
        pfd = xr.Dataset({'_': self.load_single_pfb(file, z_first=z_first, z_is=z_is)})
        dims = list(pfd.dims.keys())
        shape = list(pfd.dims.values())
        del pfd
        return dims, shape

    def load_single_pfb(
            self,
            filename_or_obj,
            dims=None,
            shape=None,
            z_first=True,
            z_is='z',
    ) -> xr.DataArray:
        """
        Load a `pfb` file directly as an xr.DataArray
        """
        if not dims:
            if z_first:
                dims = ('z', 'y', 'x')
            else:
                dims = ('x', 'y', 'z')
        print(filename_or_obj)
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(
                filename_or_obj,
                dims=dims,
                shape=shape,
                z_first=z_first,
                z_is=z_is
        ))
        var = xr.Variable(dims, data, ).squeeze()
        return var

    def load_stack_of_pfb(
        self,
        filenames,
        dims=None,
        shape=None,
        z_first=True,
        z_is='z',
    ):
        if not dims:
            if z_first:
                dims = ('time', 'z', 'y', 'x')
            else:
                dims = ('time', 'x', 'y', 'z')
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(
                filenames,
                dims=dims,
                shape=shape,
                z_first=z_first,
                z_is=z_is,
        ))
        var = xr.Variable(dims, data)
        return var

    def load_component_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (component, x, y, z)
        where component represents an anisotropy
        """
        all_da = {}
        if var_meta['domain'] == 'surface':
            dims = ('x', 'y')
        elif var_meta['domain'] == 'subsurface':
            dims = ('z', 'y', 'x')
        for sub_dict in var_meta['data']:
            component = sub_dict['component']
            comp_name = f'{name}_{component}'
            file = sub_dict['file']
            if not os.path.exists(file):
                file = f'{self.base_dir}/{file}'
            v = self.load_single_pfb(file).squeeze()
            all_da[comp_name] = xr.Dataset({comp_name: v})[comp_name]
        return all_da

    def load_time_varying_pfb(self, var_meta, name):
        """
        THese filetypes have dimensions (time, z, y, x)
        where a each file represents an individual time
        """
        file_template = var_meta['data'][0]['file-series']
        n_time = 0
        concat_dim = 'time'
        time_idx = np.arange(*var_meta['data'][0]['time-range'])
        n_time = time_idx[-1]
        pad, fmt = file_template.split('.')[-2:]
        basename = '.'.join(file_template.split('.')[:-2])
        all_files = [f'{basename}.{pad%n}.{fmt}' for n in time_idx]
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
        base_da = xr.Dataset({name: base_da})[name]
        return {name: base_da}

    def load_time_varying_2d_ts_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (time_stride, x, y, time_slice)
        where the time dimension will be strided along separate files
        and each individual file contains time_slice number of timesteps
        """
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
        _, inf_shape = self._infer_dims_and_shape(all_files[0])
        inf_dims = ('time', 'y', 'x')
        inf_shape = (len(all_files)*inf_shape[0], *inf_shape[1:])
        base_da = self.load_stack_of_pfb(
                all_files, dims=inf_dims, shape=inf_shape,
                z_first=True, z_is='time'
        )
        base_da = xr.Dataset({name: base_da})[name]
        return {name: base_da}

    def load_clm_output_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (time, x, y, variable)
        where the variable ordering is fixed and each file represents an
        individual timestep
        """
        warnings.warn("""
            Reading CLM output is not officially supported,
            at this time. We'll try our best to load the data,
            but this may break in the future!
            """
        )
        varnames = [
            'latent_heat_flux',
            'outgoing_longwave',
            'sensible_heat_flux',
            'ground_heat_flux',
            'total_evapotranspiration',
            'ground_evaporation',
            'soil_evaporation',
            'transpiration',
            'swe',
            't_ground',
            'irrigation', # TODO: This may not exist?
            't_soil_layer_i' # TODO: Need a way to determine the number of layers
        ]
        file_template = var_meta['data'][0]['file-series']
        n_time = 0
        concat_dim = 'time'
        time_idx = np.arange(*var_meta['data'][0]['time-range'])
        n_time = time_idx[-1]
        pad, filler, fmt = file_template.split('.')[-3:]
        basename = '.'.join(file_template.split('.')[:-3])
        all_files = [f'{basename}.{pad%n}.{filler}.{fmt}' for n in time_idx]
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
        base_da = xr.Dataset({name: base_da})[name].rename({'z': 'clm_out_var'})
        return {name: base_da}


        raise NotImplementedError('CLM output loading not supported. Coming soon!')

    def load_pfb_from_meta(self, var_meta, name='_', parallel=False):
        base_type = var_meta['type']
        if base_type == 'pfb':
            # Is it component?
            if len(var_meta['data']) > 1 and 'component' in var_meta['data'][0]:
                ret_das = self.load_component_pfb(var_meta, name)
            # Is it normal?
            elif var_meta.get('time-varying', None):
                ret_das = self.load_time_varying_pfb(var_meta, name)
            else:
                filename = var_meta['data'][0]['file']
                if not os.path.exists(filename):
                    filename = f'{self.base_dir}/{filename}'
                v = self.load_single_pfb(filename).squeeze()
                ret_das = {name: xr.Dataset({name: v})[name]}
        elif base_type == 'clm_output':
            ret_das = self.load_clm_output_pfb(var_meta, name)
        elif base_type == 'pfb 2d timeseries':
            ret_das = self.load_time_varying_2d_ts_pfb(var_meta, name)
        else:
            raise ValueError(f'Could not find meta type for {base_type}')
        return ret_das

    def guess_can_open(self, filename_or_obj):
        openable_extensions = ['pfb', 'pfmetadata']#, 'pbidb']
        for ext in openable_extensions:
            if filename_or_obj.endswith(ext):
                return True
        return False


@delayed
def _getitem_no_state(file_or_seq, key, dims, mode, z_first=True, z_is='z'):
    if mode == 'single':
        accessor = {d: util._key_to_explicit_accessor(k)
                    for d, k in zip(dims, key)}
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
                  accessor[d[2]]['indices']].squeeze()
    elif mode == 'sequence':
        accessor = {d: util._key_to_explicit_accessor(k)
                    for d, k in zip(dims, key)}
        t_start = accessor['time']['start']
        t_end = accessor['time']['stop'] - 1
        if z_is == 'time':
            # WARNING:  This is pretty hacky, accounting for first timestep offset
            file_start_time = int(file_or_seq[t_start].split('.')[-2].split('_')[0]) - 1
            accessor['time']['start'] -= file_start_time
            accessor['time']['stop'] -= file_start_time
        if t_start == t_end:
            t_end += 1
        sub = read_stack_of_pfbs(
            file_or_seq[t_start:t_end],
            keys=accessor,
            z_first=z_first,
            z_is=z_is,
        )
        # TODO: This can probably be cleaned up now with just
        # sub = sub[accessor[d]['indices'] for d in dims]
        if z_is == 'time':
            sub = sub[accessor['time']['indices'],
                      accessor['x']['indices'],
                      accessor['y']['indices']]
        elif z_first:
            sub = sub[accessor['time']['indices'],
                      accessor['z']['indices'],
                      accessor['y']['indices'],
                      accessor['x']['indices']]
        else:
            sub = sub[accessor['time']['indices'],
                      accessor['x']['indices'],
                      accessor['y']['indices'],
                      accessor['z']['indices']]
        sub = sub.squeeze()
    return sub


class ParflowBackendArray(BackendArray):

    def __init__(
         self,
         file_or_seq,
         dims=None,
         shape=None,
         z_first=True,
         z_is='z'
    ):
        self.file_or_seq = file_or_seq
        if isinstance(self.file_or_seq, str):
            self.mode = 'single'
            self.header_file = self.file_or_seq
        elif isinstance(self.file_or_seq, Iterable):
            self.mode = 'sequence'
            self.header_file = self.file_or_seq[0]
            # TODO: Should this be done in `load_time_varying_2d_ts_pfb`?
            if z_is == 'time' and shape is not None:
                time_idx = np.nonzero(np.array(dims) == 'time')[0][0]
                ntime = np.array(shape)[time_idx]
                ts_per_file = int(ntime / len(self.file_or_seq))
                self.file_or_seq = np.repeat(self.file_or_seq, ts_per_file)
        self._shape = shape
        self._dims = dims
        self.z_first=z_first
        self.z_is=z_is
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
                if self.z_first:
                    self._dims = ('z', 'y', 'x')
                else:
                    self._dims = ('x', 'y', 'z')
            elif self.mode == 'sequence':
                if self.z_first:
                    self._dims = ('time', 'z', 'y', 'x')
                else:
                    self._dims = ('time', 'x', 'y', 'x')
        return self._dims

    @property
    def shape(self):
        if self._shape is None:
            with ParflowBinaryReader(
                self.header_file,
                precompute_subgrid_info=False
            ) as pfd:
                if self.z_first:
                    base_shape = (pfd.header['nz'], pfd.header['ny'], pfd.header['nx'])
                else:
                    base_shape = (pfd.header['nx'], pfd.header['ny'], pfd.header['nz'])
            if self.mode == 'sequence':
                base_shape = (len(self.file_or_seq), *base_shape)
            # Add some logic for automatically squeezing here?
            self._shape = base_shape
        return self._shape

    def _getitem(self, key: tuple) -> np.typing.ArrayLike:
        size = self._size_from_key(key)
        key = self._explicit_indices_from_keys(size, key)
        sub = delayed(_getitem_no_state)(
                self.file_or_seq, key, self.dims, self.mode,
                self.z_first, self.z_is)
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
        if self.z_first:
            dims = ('z', 'y', 'x')
        else:
            dims = ('x', 'y', 'z')
        return dims

    def get_shape(self):
        with ParflowBinaryReader(
            self.header_file,
            precompute_subgrid_info=False
        ) as pfd:
            if self.z_first:
                base_shape = (pfd.header['nz'], pfd.header['ny'], pfd.header['nx'])
            else:
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
