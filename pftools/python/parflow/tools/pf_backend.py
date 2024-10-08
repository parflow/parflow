import contextlib
import dask
import json
import numpy as np
import pandas as pd
import os
import warnings
import xarray as xr
import yaml

from pprint import pprint
from . import util
from .io import ParflowBinaryReader, read_pfb_sequence, read_pfb
from collections.abc import Iterable
from typing import Mapping, List, Union
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
from dask import delayed


class ParflowBackendEntrypoint(BackendEntrypoint):
    """
    Provides an entrypoint for xarray to read Parflow input/output
    Note: Users should never instantiate, import, or call any of
    this code directly. This hooks into the xarray library directly
    via the entrypoints provided at the time of installation. Users
    should interact with this code by using the xarray library directly.
    """

    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "name",
        "meta_yaml",
        "read_inputs",
        "read_outputs",
        "inferred_dims",
        "inferred_shape",
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        base_dir=None,
        drop_variables=None,
        name="parflow_variable",
        read_inputs=True,
        read_outputs=True,
        inferred_dims=None,
        inferred_shape=None,
        chunks={},
        strict_ext_check=False,
    ) -> xr.Dataset:
        """
        Open Parflow input/output as an xarray.Dataset.

        :param filename_or_obj:
            The pfb file or pfmetadata file to read.
        :param base_dir:
            A base directory to read from. This is optional, but allows
            you to specify a common directory to read multiple files from
            if it is not recorded in the pfmetadata file.
        :param read_inputs:
            Whether or not to read files in the "inputs" section of the
            pfmetadata file. This can be either a boolean value or a list
            of variable names to read from the "inputs" section. If true
            xarray will read all available input variables.
        :param read_outputs:
            Whether or not to read files in the "outputs" section of the
            pfmetadata file. This can be either a boolean value or a list
            of variable names to read from the "outputs" section. If true
            xarray will read all available output variables.
        :param inferred_dims:
            Expected dimensions of the arrays being read in. This can be used
            to slightly optimize performance, but is not necessary.
        :param inferred_shape:
            Expected shape of the arrays being read in. This can be used
            to slightly optimize performance, but is not necessary.
        :param chunks:
            The chunking scheme to apply along dimensions. See:
            https://xarray.pydata.org/en/stable/generated/xarray.Dataset.chunk.html
            for useage options. This is primarily set for automatically paralellizing
            computations via dask.
        :param strict_ext_check:
            Whether or not to strictly check the filename extension for
            determining if we are opening a pfb file or a pfmetadata file.
            Strict checks will actually try to open the file, while non-strict
            simply check the filename extension. This can slightly improve performance
            when reading many pfb files.
        :return:
            An xr.Dataset with a collection of xr.DataArray objects as variables.
        """
        filetype = self.is_meta_or_pfb(filename_or_obj, strict=strict_ext_check)
        if filetype == "pfb":
            # Reads a single pfb
            data = self.load_single_pfb(
                filename_or_obj, dims=inferred_dims, shape=inferred_shape
            )
            ds = xr.Dataset({name: data}).chunk(chunks)
        elif filetype == "pfmetadata":
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
            )
        return ds

    def load_pfmetadata(
        self,
        filename_or_obj,
        pf_meta,
        read_inputs,
        read_outputs,
    ) -> xr.Dataset:
        """
        Helper method to load data specified via a pfmetadata file.

        :param filename_or_obj:
            The pfmetadata file to read
        :param pf_meta:
            The parsed pfmetadata file - automatically registered via
            the `_is_meta_or_pfb` method.
        :param read_inputs:
            Whether or not to read files in the "inputs" section of the
            pfmetadata file. This can be either a boolean value or a list
            of variable names to read from the "inputs" section. If true
            xarray will read all available input variables.
        :param read_outputs:
            Whether or not to read files in the "outputs" section of the
            pfmetadata file. This can be either a boolean value or a list
            of variable names to read from the "outputs" section. If true
            xarray will read all available output variables.
        :return:
            The assembled xarray dataset
        """
        ds = xr.Dataset()
        ds.attrs["pf_metadata_file"] = filename_or_obj
        ds.attrs["parflow_version"] = pf_meta["parflow"]["build"]["version"]
        if "coordinates" in self.pf_meta:
            coords = self.load_coords_from_meta(self.pf_meta["coordinates"])
            ds = ds.assign_coords(coords)
        if read_outputs:
            for var, var_meta in self.pf_meta["outputs"].items():
                if read_outputs is True or var in read_outputs:
                    das = self.load_pfb_from_meta(var_meta, name=var)
                    for k, v in das.items():
                        ds[k] = v
        if read_inputs:
            for var, var_meta in self.pf_meta["inputs"].items():
                if var == "configuration":
                    continue  # TODO: Determine what to do with this
                if read_inputs is True or var in read_inputs:
                    das = self.load_pfb_from_meta(var_meta, name=var)
                    for k, v in das.items():
                        ds[k] = v
        return ds

    def load_coords_from_meta(self, coord_meta) -> Mapping[str, xr.DataArray]:
        """
        Builds coordinate variables from the 'coordinates' section of
        a pfmetadata file.
        """
        coords = {}
        for var, var_meta in coord_meta.items():
            meta_type = var_meta["type"]
            if meta_type == "time":
                coords[var] = pd.DatetimeIndex(
                    pd.date_range(
                        start=var_meta["start"],
                        end=var_meta["stop"],
                        freq=var_meta["freq"],
                    )
                )
            elif meta_type == "pfb":
                coords[var] = self.load_pfb_from_meta(var_meta, name=var)[var]
            else:
                # TODO: add a warning here
                pass
        return coords

    def load_pfb_from_meta(self, var_meta, name="_") -> Mapping[str, xr.Dataset]:
        """
        Determines which sub-reader call to make based on the information
        in the metadata section  for a single variable of the pfmetadata file.

        :param var_meta:
            The metadata for the variable being read.
        :param name:
            A name to give the xarray dataset.
        :returns:
            A dictionary of xarray datasets with keys being the variable names
        """
        base_type = var_meta["type"]
        if base_type == "pfb":
            # Is it component?
            if len(var_meta["data"]) > 1 and "component" in var_meta["data"][0]:
                ret_das = self.load_component_pfb(var_meta, name)
            # Is it time varying?
            elif var_meta.get("time-varying", None):
                ret_das = self.load_time_varying_pfb(var_meta, name)
            # Is it normal
            else:
                filename = var_meta["data"][0]["file"]
                if not os.path.exists(filename):
                    filename = f"{self.base_dir}/{filename}"
                v = self.load_single_pfb(filename)
                ret_das = {name: xr.Dataset({name: v})[name]}
        elif base_type == "clm_output":
            ret_das = self.load_clm_output_pfb(var_meta, name)
        elif base_type == "pfb 2d timeseries":
            ret_das = self.load_time_varying_2d_ts_pfb(var_meta, name)
        else:
            raise ValueError(f"Could not find meta type for {base_type}")
        return ret_das

    def load_component_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (component, x, y, z)
        where component represents an anisotropy
        """
        all_da = {}
        if var_meta["domain"] == "surface":
            dims = ("x", "y")
        elif var_meta["domain"] == "subsurface":
            dims = ("z", "y", "x")
        for sub_dict in var_meta["data"]:
            component = sub_dict["component"]
            comp_name = f"{name}_{component}"
            file = sub_dict["file"]
            if not os.path.exists(file):
                file = f"{self.base_dir}/{file}"
            v = self.load_single_pfb(file)
            all_da[comp_name] = xr.Dataset({comp_name: v})[comp_name]
        return all_da

    def load_time_varying_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (time, z, y, x)
        where a each file represents an individual time
        """
        file_template = var_meta["data"][0]["file-series"]
        n_time = 0
        concat_dim = "time"
        time_idx = np.arange(*var_meta["data"][0]["time-range"])
        n_time = time_idx[-1]
        pad, fmt = file_template.split(".")[-2:]
        basename = ".".join(file_template.split(".")[:-2])
        all_files = [f"{basename}.{pad%n}.{fmt}" for n in time_idx]
        # Check if basename contains any of the files if not,
        # fall back to `self.base_dir` from the pfmetadata file
        if not os.path.exists(all_files[0]):
            all_files = [f"{self.base_dir}/{af}" for af in all_files]

        # Put it all together
        base_da = self.load_sequence_of_pfb(all_files)
        base_da = xr.Dataset({name: base_da})[name]
        return {name: base_da}

    def load_time_varying_2d_ts_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (time_stride, x, y, time_slice)
        where the time dimension will be strided along separate files
        and each individual file contains time_slice number of timesteps
        """
        concat_dim = "z"  # z is time here
        time_start = np.arange(*var_meta["data"][0]["times-between"])
        time_end = time_start + var_meta["data"][0]["times-between"][-1] - 1
        ntime = time_end[-1]
        file_template = var_meta["data"][0]["file-series"]
        pad, fmt = file_template.split(".")[-2:]
        basename = ".".join(file_template.split(".")[:-2])
        all_files = [
            f"{basename}.{pad%(s,e)}.{fmt}" for s, e in zip(time_start, time_end)
        ]
        # Check if basename contains any of the files if not,
        # fall back to `self.base_dir` from the pfmetadata file
        if not os.path.exists(all_files[0]):
            all_files = [f"{self.base_dir}/{af}" for af in all_files]

        # Put it all together
        base_da = self.load_sequence_of_pfb(all_files, z_is="time")
        base_da = xr.Dataset({name: base_da})[name]
        return {name: base_da}

    def load_clm_output_pfb(self, var_meta, name):
        """
        These filetypes have dimensions (time, x, y, variable)
        where the variable ordering is fixed and each file represents an
        individual timestep
        """
        warnings.warn(
            """
            Reading CLM output is not officially supported,
            at this time. We'll try our best to load the data,
            but this may break in the future!
            """
        )
        varnames = [
            "latent_heat_flux",
            "outgoing_longwave",
            "sensible_heat_flux",
            "ground_heat_flux",
            "total_evapotranspiration",
            "ground_evaporation",
            "soil_evaporation",
            "veg_evaporation",
            "transpiration",
            "infiltration",
            "swe",
            "t_ground",
        ]
        file_template = var_meta["data"][0]["file-series"]
        n_time = 0
        concat_dim = "time"
        time_idx = np.arange(*var_meta["data"][0]["time-range"])
        n_time = time_idx[-1]
        pad, filler, fmt = file_template.split(".")[-3:]
        basename = ".".join(file_template.split(".")[:-3])
        all_files = [f"{basename}.{pad%n}.{filler}.{fmt}" for n in time_idx]
        # Check if basename contains any of the files if not,
        # fall back to `self.base_dir` from the pfmetadata file
        if not os.path.exists(all_files[0]):
            all_files = [f"{self.base_dir}/{af}" for af in all_files]

        clm_das = []
        for i, v in enumerate(varnames):
            var_da = self.load_sequence_of_pfb(all_files, init_key={"z": i})
            var_da = xr.Dataset({v: var_da})[v]
            clm_das.append(var_da)
        clm_das = xr.merge(clm_das)
        return clm_das

    def load_single_pfb(
        self,
        filename_or_obj,
        dims=None,
        shape=None,
        z_first=True,
        z_is="z",
    ) -> xr.Variable:
        """
        Load a `pfb` file directly as an xr.Variable
        """
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(
                filename_or_obj, dims=dims, shape=shape, z_first=z_first, z_is=z_is
            )
        )
        if not dims:
            dims = data.array.dims
        if not shape:
            shape = data.array.shape
        var = xr.Variable(
            dims,
            data,
        )
        return var

    def load_sequence_of_pfb(
        self, filenames, dims=None, shape=None, z_first=True, z_is="z", init_key={}
    ) -> xr.Variable:
        data = indexing.LazilyIndexedArray(
            ParflowBackendArray(
                filenames,
                dims=dims,
                shape=shape,
                z_first=z_first,
                z_is=z_is,
                init_key=init_key,
            )
        )
        if not dims:
            dims = data.array.dims
        if not shape:
            shape = data.array.shape
        var = xr.Variable(dims, data)
        return var

    def is_meta_or_pfb(self, filename_or_obj, strict=True):
        """Determine if a file is a pfb file or pfmetadata file"""

        def _check_dict_is_valid_meta(meta):
            assert "parflow" in meta.keys(), (
                'Metadata file missing "parflow" key - ',
                "are you sure this is a valid Parflow metadata file?",
            )

        if not strict:
            # Just check the extension
            ext = filename_or_obj.split(".")[-1]
            if ext == "pfmetadata":
                with open(filename_or_obj, "r") as f:
                    pf_meta = json.load(f)
                    _check_dict_is_valid_meta(pf_meta)
                    self.pf_meta = pf_meta
            return ext

        if isinstance(filename_or_obj, str):
            try:
                with ParflowBinaryReader(filename_or_obj) as pfd:
                    assert "nx" in pfd.header
                    assert "ny" in pfd.header
                    assert "nz" in pfd.header
                return "pfb"
            except AssertionError:
                with open(filename_or_obj, "r") as f:
                    pf_meta = json.load(f)
                    _check_dict_is_valid_meta(pf_meta)
                    self.pf_meta = pf_meta
                    return "pfmetadata"
        elif isinstance(filename_or_obj, dict):
            _check_dict_is_valid_meta(filename_or_obj)
            self.pf_meta = filename_or_obj
            return "pfmetadata"
        else:
            raise NotImplementedError("Was unable to determine input type!")

    def _infer_dims_and_shape(self, file, z_first=True, z_is="z"):
        """Determine the dimensions and shape of a pfb file"""
        pfd = xr.Dataset({"_": self.load_single_pfb(file, z_first=z_first, z_is=z_is)})
        dims = list(pfd.dims.keys())
        shape = list(pfd.dims.values())
        del pfd
        return dims, shape

    def guess_can_open(self, filename_or_obj):
        """Registers the backend to recognize *.pfb and *.pfmetadata files"""
        openable_extensions = ["pfb", "pfmetadata"]
        for ext in openable_extensions:
            if filename_or_obj.endswith(ext):
                return True
        return False


def _getitem_no_state(file_or_seq, key, dims, mode, z_first=True, z_is="z"):
    """
    Base functionality for actually getting data out of PFB files.

    :param file_or_seq:
        File or files that should be read from.
    :param key:
        A key indicating which indices should be read into memory.
    :param dims:
        The dimensions of the dataset.
    :param mode:
        Specification of whether a single file or a sequence of files
        should be read in.
    :param z_first:
        Whether the z axis should be first. If not, it it will be last.
    :param z_is:
        What the z-axis represents. Can be 'z', 'time', or 'variable'
    :return:
        A numpy array of the data
    """
    if mode == "single":
        accessor = {d: util._key_to_explicit_accessor(k) for d, k in zip(dims, key)}
        sub = read_pfb(
            file_or_seq,
            keys=accessor,
            z_first=z_first,
        )
    elif mode == "sequence":
        accessor = {d: util._key_to_explicit_accessor(k) for d, k in zip(dims, key)}
        t_start = accessor["time"]["start"]
        t_end = accessor["time"]["stop"]
        if z_is == "time":
            # WARNING:  This is pretty hacky, accounting for first timestep offset
            try:
                # Parflow files end with TIMESTEP.pfb
                file_start_time = (
                    int(file_or_seq[t_start].split(".")[-2].split("_")[0]) - 1
                )
            except:
                # CLM output files end with TIMESTEP.C.pfb
                file_start_time = (
                    int(file_or_seq[t_start].split(".")[-3].split("_")[0]) - 1
                )
            accessor["time"]["start"] -= file_start_time
            accessor["time"]["stop"] -= file_start_time
        if t_start is not None and t_start == t_end:
            t_end += 1

        # Here we explicitly select which files we need to read
        # to reduce the overall IO overhead. After selecting them
        # out of the list we must remove the accessor indices
        # because they have been used.
        read_files = file_or_seq[t_start:t_end]
        read_files = np.array(read_files)[accessor["time"]["indices"]]
        accessor["time"]["indices"] = slice(None, None, None)
        # Read the array
        sub = read_pfb_sequence(
            read_files,
            keys=accessor,
            z_first=z_first,
            z_is=z_is,
        )
    sub = sub[tuple([accessor[d]["indices"] for d in dims])]
    # Check which axes need to be squeezed out. This is
    # to distinguish between doing `ds.isel(x=[0])` which
    # should keep the x axis (and dimension) and `ds.isel(x=0)`
    # which should remove the x axis (and dimension).
    axes_to_squeeze = tuple(i for i, d in enumerate(dims) if accessor[d]["squeeze"])
    sub = np.squeeze(sub, axis=axes_to_squeeze)
    return sub


class ParflowBackendArray(BackendArray):
    """Backend array that allows for lazy indexing on pfb-based data."""

    def __init__(
        self,
        file_or_seq,
        dims=None,
        shape=None,
        z_first=True,
        z_is="z",
        init_key={},
    ):
        """
        Instantiate a new ParflowBackendArray.

        :param file_or_seq:
            File or files that the array will read from.
        :param dims:
            Names of the dimension along each array axis.
        :param shape:
            The expected shape of the array.
        :param z_first:
            Whether the z axis is first. If not, it will be last.
        :param z_is:
            What the z axis represents. Can be 'z', 'time', 'variable'
        :param init_key:
            An initial key that can be used to prematurely subset.
        """
        self.file_or_seq = file_or_seq
        if isinstance(self.file_or_seq, str):
            self.mode = "single"
            self.header_file = self.file_or_seq
        elif isinstance(self.file_or_seq, Iterable):
            self.mode = "sequence"
            self.header_file = self.file_or_seq[0]
            # TODO: Should this be done in `load_time_varying_2d_ts_pfb`?
            if z_is == "time" and shape is not None:
                time_idx = np.nonzero(np.array(dims) == "time")[0][0]
                ntime = np.array(shape)[time_idx]
                ts_per_file = int(ntime / len(self.file_or_seq))
                self.file_or_seq = np.repeat(self.file_or_seq, ts_per_file)
        self._shape = shape
        self._dims = dims
        self._pfb_dims = None
        self._pfb_shape = None
        self._squeeze_dims = None
        self.z_first = z_first
        self.z_is = z_is
        self.init_key = init_key
        # Weird hack here, have to pull the dtype like this
        # to have valid `nbytes` attribute
        self.dtype = np.dtype(np.float64)

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.ndarray:
        """Dunder method to call implement the underlying indexing scheme"""
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._getitem,
        )

    def _set_dims_and_shape(self):
        with ParflowBinaryReader(
            self.header_file, precompute_subgrid_info=False
        ) as pfd:
            if self.z_first:
                _shape = [pfd.header["nz"], pfd.header["ny"], pfd.header["nx"]]
            else:
                _shape = [pfd.header["nx"], pfd.header["ny"], pfd.header["nz"]]
        if self.mode == "sequence":
            _shape = [len(self.file_or_seq), *_shape]
        # Construct dimension template
        if self.mode == "single":
            if self.z_first:
                _dims = ["z", "y", "x"]
            else:
                _dims = ["x", "y", "z"]
        elif self.mode == "sequence":
            if self.z_first:
                _dims = ["time", "z", "y", "x"]
            else:
                _dims = ["time", "x", "y", "x"]
        # Add some logic for dealing with clm output's inconsistent format
        if self.init_key:
            for i, (dim, size) in enumerate(zip(_dims, _shape)):
                if dim in self.init_key:
                    _shape[i] = self._size_from_key([self.init_key[dim]])[0]
        self._squeeze_dims = tuple(i for i, s in enumerate(_shape) if s == 1)
        if not self._shape:
            self._shape = tuple(s for s in _shape if s > 1)
        if not self._dims:
            self._dims = tuple(d for s, d in zip(_shape, _dims) if s > 1)
        self._pfb_dims = tuple(_dims)
        self._pfb_shape = tuple(_shape)

    @property
    def dims(self):
        """Names of the dimensions of each axis of the array"""
        if self._dims is None:
            self._set_dims_and_shape()
        return self._dims

    @property
    def shape(self):
        """Shape of the data once loaded into memory"""
        if self._shape is None:
            self._set_dims_and_shape()
        return self._shape

    @property
    def pfb_dims(self):
        """names of dimensions in the underlying pfb file"""
        if self._pfb_dims is None:
            self._set_dims_and_shape()
        return self._pfb_dims

    @property
    def pfb_shape(self):
        """names of dimensions in the underlying pfb file"""
        if self._pfb_shape is None:
            self._set_dims_and_shape()
        return self._pfb_shape

    @property
    def squeeze_dims(self):
        """names of dimensions in the underlying pfb file"""
        if self._squeeze_dims is None:
            self._set_dims_and_shape()
        return self._squeeze_dims

    def _getitem(self, key: tuple) -> np.ndarray:
        """Mapping between keys to the actual data"""
        real_size = self._size_from_key(key)
        sub = delayed(_getitem_no_state)(
            self.file_or_seq, key, self.dims, self.mode, self.z_first, self.z_is
        )
        sub = dask.array.from_delayed(sub, self.pfb_shape, dtype=np.float64)
        if self.shape != sub.shape:
            sub = dask.array.squeeze(sub, axis=self.squeeze_dims)
        return sub

    def _size_from_key(self, key):
        """Determine the size of a returned array given an indexing key"""
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
