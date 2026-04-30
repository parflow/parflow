from dataclasses import dataclass
from typing import Iterable, Optional, Mapping
import itertools

import xarray as xr
import numpy as np
from xarray.backends import BackendArray
from xarray.core import indexing
from parflow.tools.io import ParflowBinaryReader, Number

from parflow.tools import util
from parflow.tools.backend.generic import PfbInfo
from parflow.tools.util import read_start_stop_n


def magic_get_fill_value(data: np.ndarray):
    """Try to determine the fill_value used for the data"""
    v_min = np.min(data)
    if v_min == -9999:
        return v_min
    d_min = np.finfo(np.float32).min
    if d_min <= v_min <= d_min / 10:
        return v_min
    return None


class UpdatedParflowBinaryReader(ParflowBinaryReader):
    """
    Modified version of the ParflowBinaryReader with:
    - usage of precompute_subgrid_info and read_sg_info
    - performance improvement in read_subarray
    """

    def __init__(
        self,
        file: str,
        precompute_subgrid_info: bool = True,
        p: int = None,
        q: int = None,
        r: int = None,
        header: dict[str, Number] = None,
        read_sg_info: bool = False,
    ):
        # No call to super().__init__ to avoid systematic call to read_subgrid_info
        self.filename = file
        self.f = open(self.filename, "rb")
        if not header:
            self.header = self.read_header()
        else:
            self.header = header

        if np.all([p, q, r]):
            self.header["p"] = p
            self.header["q"] = q
            self.header["r"] = r

        if precompute_subgrid_info or read_sg_info:
            self.read_subgrid_info()

    def read_subarray(
        self,
        start_x: int,
        start_y: int,
        start_z: int = 0,
        nx: int = 1,
        ny: int = 1,
        nz: int = None,
        z_first: bool = True,
    ) -> np.ndarray:
        """
        Read a subsection of the full pfb file. For an example of what happens
        here consider the following image:

            ::
            +-------+-------+
            |       |       |
            |      x|xx     |
            +-------+-------+
            |      x|xx     |
            |      x|xx     |
            +-------+-------+

        Where each of the borders of the big grid are the
        four subgrids (2,2) that we are trying to index data from.
        The data to be selected falls in each of these subgrids, as
        denoted by the 'x' marks.

        :param start_x:
            The index to start at in the x dimension.
        :param start_y:
            The index to start at in the y dimension.
        :param start_z:
            The index to start at in the z dimension.
            This is optional, and if not provided is 0.
        :param nx:
            The number of values to read in the x dimension.
            This is optional, and if not provided is 1.
        :param ny:
            The number of values to read in the y dimension.
            This is optional, and if not provided is 1.
        :param nz:
            The number of values to read in the z dimension.
            This is optional, and if not provided is None,
            which indicates to read all of the values.
        :param z_first:
            Whether the z dimension should be first. If true returned arrays have
            dimensions ('z', 'y', 'x') else ('x', 'y', 'z')

        :returns:
            A nd array with shape (nx, ny, nz).
        """

        def _get_final_clip(start, end, coords):
            """Helper to clean up code at the end of this"""
            x0 = np.flatnonzero(start == coords)
            x0 = 0 if not x0 else x0[0]
            x1 = np.flatnonzero(end == coords)
            x1 = None if x1 is None or len(x1) == 0 else x1[0]
            return slice(x0, x1)

        def _get_needed_subgrids(start, end, coords):
            """Helper function to clean up subgrid selection"""
            for s, c in enumerate(coords):
                if start in c:
                    break
            for e, c in enumerate(coords):
                if end in c:
                    break
            return np.arange(s, e + 1)

        if not start_x:
            start_x = 0
        if not start_y:
            start_y = 0
        if not start_z:
            start_z = 0
        if not nx:
            nx = self.header["nx"]
        if not ny:
            ny = self.header["ny"]
        if not nz:
            nz = self.header["nz"]

        end_x = start_x + nx
        end_y = start_y + ny
        end_z = start_z + nz
        p, q, r = self.header["p"], self.header["q"], self.header["r"]

        # Convert to numpy array for simpler indexing
        x_coords = np.array(self.coords["x"], dtype=object)
        y_coords = np.array(self.coords["y"], dtype=object)
        z_coords = np.array(self.coords["z"], dtype=object)

        # Determine which subgrids we need to read
        p_subgrids = _get_needed_subgrids(start_x, end_x, x_coords)
        q_subgrids = _get_needed_subgrids(start_y, end_y, y_coords)
        r_subgrids = _get_needed_subgrids(start_z, end_z, z_coords)

        # Determine the coordinates of these subgrids
        x_sg_coords = np.unique(np.hstack(x_coords[p_subgrids]))
        y_sg_coords = np.unique(np.hstack(y_coords[q_subgrids]))
        z_sg_coords = np.unique(np.hstack(z_coords[r_subgrids]))
        # Min values will be used to align in the bounding data
        x_min = np.min(x_sg_coords)
        y_min = np.min(y_sg_coords)
        z_min = np.min(z_sg_coords)
        # Make an array which can fit all of the subgrids
        full_size = (len(z_sg_coords), len(y_sg_coords), len(x_sg_coords))
        bounding_data = np.empty(full_size, dtype=np.float64)
        subgrid_iter = itertools.product(p_subgrids, q_subgrids, r_subgrids)

        for xsg, ysg, zsg in subgrid_iter:
            subgrid_idx = xsg + (p * ysg) + (p * q * zsg)
            # Set up the indices to insert subgrid data into the bounding data
            x0, y0, z0 = self.subgrid_start_indices[subgrid_idx]
            x0, y0, z0 = x0 - x_min, y0 - y_min, z0 - z_min
            dx, dy, dz = self.subgrid_shapes[subgrid_idx]
            x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
            bounding_data[z0:z1, y0:y1, x0:x1] = self.iloc_subgrid(subgrid_idx).T

        # Now clip out the exact part from the bounding box
        clip_x = _get_final_clip(start_x, end_x, x_sg_coords)
        clip_y = _get_final_clip(start_y, end_y, y_sg_coords)
        clip_z = _get_final_clip(start_z, end_z, z_sg_coords)
        if z_first:
            ret_data = bounding_data[clip_z, clip_y, clip_x]
        else:
            ret_data = bounding_data[clip_z, clip_y, clip_x].T

        return ret_data


@dataclass
class SubGrid:
    """Dataclass for subgrid data. Mainly used to simplify the code."""

    subgrid_offsets: np.array
    subgrid_locations: np.array
    subgrid_start_indices: np.array
    subgrid_shapes: np.array
    chunks: Mapping[str, tuple]
    coords: Mapping[str, Iterable[Iterable[int]]]

    @classmethod
    def save(cls, pfb: ParflowBinaryReader):
        return cls(
            pfb.subgrid_offsets,
            pfb.subgrid_locations,
            pfb.subgrid_start_indices,
            pfb.subgrid_shapes,
            pfb.chunks,
            pfb.coords,
        )

    def apply(self, pfb: ParflowBinaryReader):
        pfb.subgrid_offsets = self.subgrid_offsets
        pfb.subgrid_locations = self.subgrid_locations
        pfb.subgrid_start_indices = self.subgrid_start_indices
        pfb.subgrid_shapes = self.subgrid_shapes
        pfb.coords = self.coords
        pfb.chunks = self.chunks


class PfbReader:
    """
    PFB file reader, with a single header read for a sequence and internal methods for the different types of
    sequences encountered (with keys, with z_is=="time" ...).
    """

    def __init__(
        self,
        file_seq: PfbInfo | Iterable[PfbInfo],
        z_first: bool = True,
        z_is: str = "z",
    ):
        if isinstance(file_seq, PfbInfo):
            file_seq = [file_seq]

        self.list_pfb_info = file_seq
        self.z_first = z_first
        self.z_is = z_is

        self._base_header = None
        self._subgrid = None

    def _set_header_subgrid(self):
        with UpdatedParflowBinaryReader(self.list_pfb_info[0].filename) as pfb_init:
            base_header = pfb_init.header
            if self.list_pfb_info[0].run is not None:
                base_header["x"] = self.list_pfb_info[0].run.ComputationalGrid.Lower.X
                base_header["y"] = self.list_pfb_info[0].run.ComputationalGrid.Lower.Y
            self._base_header = base_header
            self._subgrid = SubGrid.save(pfb_init)

    @property
    def base_header(self):
        if self._base_header is None:
            self._set_header_subgrid()
        return self._base_header

    @property
    def subgrid(self) -> SubGrid:
        if self._subgrid is None:
            self._set_header_subgrid()
        return self._subgrid

    def _read_all(self):
        nx, ny, nz = (
            self.base_header["nx"],
            self.base_header["ny"],
            self.base_header["nz"],
        )

        if self.z_first:
            seq_size = (len(self.list_pfb_info), nz, ny, nx)
        else:
            seq_size = (len(self.list_pfb_info), nx, ny, nz)
        pfb_seq = np.empty(seq_size, dtype=np.float64)
        for i, pfb_info in enumerate(self.list_pfb_info):
            with UpdatedParflowBinaryReader(
                pfb_info.filename,
                precompute_subgrid_info=False,
                header=self.base_header,
            ) as pfb:
                self.subgrid.apply(pfb)
                subseq_data = pfb.read_all_subgrids(mode="full", z_first=self.z_first)
                pfb_seq[i, :, :, :] = subseq_data

        if self.z_is == "time":
            if self.z_first:
                pfb_seq = np.concatenate(pfb_seq, axis=0)
            else:
                pfb_seq = np.concatenate(pfb_seq, axis=-1)

        return pfb_seq

    def _read_keys(self, keys):
        # Here we explicitly select which files we need to read
        # to reduce the overall IO overhead. After selecting them
        # out of the list we must remove the accessor indices
        # because they have been used.
        t_start = keys["time"]["start"]
        t_end = keys["time"]["stop"]

        if t_start is not None and t_start == t_end:
            t_end += 1

        file_seq: list[PfbInfo] = self.list_pfb_info[t_start:t_end]
        file_seq = np.array(file_seq)[keys["time"]["indices"]]
        keys["time"]["indices"] = slice(None, None, None)

        start_x, stop_x, nx = read_start_stop_n(keys, "x", 0, self.base_header["nx"])
        start_y, stop_y, ny = read_start_stop_n(keys, "y", 0, self.base_header["ny"])
        start_z, stop_z, nz = read_start_stop_n(
            keys, self.z_is, 0, self.base_header["nz"]
        )

        if self.z_first:
            seq_size = (len(file_seq), nz, ny, nx)
        else:
            seq_size = (len(file_seq), nx, ny, nz)
        pfb_seq = np.empty(seq_size, dtype=np.float64)
        for i, pfb_info in enumerate(file_seq):
            with UpdatedParflowBinaryReader(
                pfb_info.filename,
                precompute_subgrid_info=False,
                header=self.base_header,
            ) as pfb:
                self.subgrid.apply(pfb)
                subseq_data = pfb.read_subarray(
                    start_x, start_y, start_z, nx, ny, nz, z_first=self.z_first
                )
                pfb_seq[i, :, :, :] = subseq_data

        return pfb_seq

    def _read_keys_z_is_time(self, keys):
        z_size = len(self.list_pfb_info) * self.base_header["nz"]
        start_x, stop_x, nx = read_start_stop_n(keys, "x", 0, self.base_header["nx"])
        start_y, stop_y, ny = read_start_stop_n(keys, "y", 0, self.base_header["ny"])
        start_z, stop_z, nz = read_start_stop_n(keys, self.z_is, 0, z_size)

        if self.z_first:
            seq_size = (nz, ny, nx)
        else:
            seq_size = (nz, nx, ny)

        pfb_seq = np.empty(seq_size, dtype=np.float64)

        for i, pfb_info in enumerate(self.list_pfb_info):
            start = i * self.base_header["nz"]
            stop = (i + 1) * self.base_header["nz"] - 1
            if stop_z - 1 < start or start_z > stop:
                continue

            file_start_z = max(start_z - start, 0)
            file_stop_z = min(stop_z - start, self.base_header["nz"])
            file_nz = file_stop_z - file_start_z
            output_start_z = start + file_start_z - start_z
            output_stop_z = start + file_stop_z - start_z

            with UpdatedParflowBinaryReader(
                pfb_info.filename,
                precompute_subgrid_info=False,
                header=self.base_header,
            ) as pfb:
                self.subgrid.apply(pfb)
                subseq_data = pfb.read_subarray(
                    start_x,
                    start_y,
                    file_start_z,
                    nx,
                    ny,
                    file_nz,
                    z_first=self.z_first,
                )

                if self.z_first:
                    pfb_seq[output_start_z:output_stop_z, :, :] = subseq_data
                else:
                    pfb_seq[output_start_z:output_stop_z, :, :] = np.moveaxis(
                        subseq_data, -1, 0
                    )

        return pfb_seq

    def _read_single(self, keys):
        # read_pfb from parflow.tools.io
        with UpdatedParflowBinaryReader(
            self.list_pfb_info[0].filename, header=self.base_header
        ) as pfb:
            if not keys:
                data = pfb.read_all_subgrids(z_first=self.z_first)
            else:
                start_x, stop_x, nx = read_start_stop_n(
                    keys, "x", 0, self.base_header["nx"]
                )
                start_y, stop_y, ny = read_start_stop_n(
                    keys, "y", 0, self.base_header["ny"]
                )
                start_z, stop_z, nz = read_start_stop_n(
                    keys, self.z_is, 0, self.base_header["nz"]
                )

                data = pfb.read_subarray(
                    start_x, start_y, start_z, nx, ny, nz, z_first=self.z_first
                )
        return data

    def read(self, keys: Optional = None):
        if len(self.list_pfb_info) == 1:
            return self._read_single(keys)
        if keys is None:  # Not used ?
            return self._read_all()
        if self.z_is == "time":
            return self._read_keys_z_is_time(keys)
        return self._read_keys(keys)


class ParflowBackendArray(BackendArray):
    """Backend array that allows for lazy indexing on pfb-based data.
    Modified version of the one implemented in pftools with bug correction and using PfbReader
    """

    def __init__(
        self,
        file_or_seq: PfbInfo | list[PfbInfo],
        dims=None,
        shape=None,
        z_first=True,
        z_is="z",
        init_key={},
        replace_fill_value=True,
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
        if isinstance(self.file_or_seq, PfbInfo):
            self.mode = "single"
        elif isinstance(self.file_or_seq, Iterable):
            self.mode = "sequence"

        self._shape = shape
        self._dims = dims
        self._default_chunk = None
        self._squeeze_dims = ()
        self._fill_value = None
        self._replace_fill_value = replace_fill_value
        self.z_first = z_first
        self.z_is = z_is
        self.init_key = init_key
        # Weird hack here, have to pull the dtype like this
        # to have valid `nbytes` attribute
        self.dtype = np.dtype(np.float64)
        self.reader = PfbReader(self.file_or_seq, self.z_first, z_is)

    def __getitem__(self, key: xr.core.indexing.ExplicitIndexer) -> np.ndarray:
        """Dunder method to call implement the underlying indexing scheme"""
        # print(key)
        # key = xr.core.indexing.BasicIndexer(key)
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._getitem,
        )

    def _set_dims_and_shape(self):
        header = self.reader.base_header

        _lens = dict(z=header["nz"], y=header["ny"], x=header["nx"])

        if self.z_first:
            _default_chunk = ["z", "y", "x"]
        else:
            _default_chunk = ["x", "y", "z"]

        _dims = _default_chunk.copy()
        if self.mode == "sequence":
            _dims = ["time"] + _dims
            if self.z_is == "time":
                _lens["time"] = len(self.file_or_seq) * _lens["z"]
                _dims.remove("z")
            else:
                _lens["time"] = len(self.file_or_seq)
                _lens["time_chunk"] = 1
                _default_chunk = ["time_chunk"] + _default_chunk

        # Add some logic for dealing with clm output's inconsistent format
        if self.init_key:
            for dim in _dims:
                if dim in self.init_key:
                    _lens[dim] = self._size_from_key([self.init_key[dim]])[0]

        if _lens["z"] == 1 and self.z_is != "time":
            self._squeeze_dims = _dims.index("z")
            _default_chunk.remove("z")
            _dims.remove("z")

        self._default_chunk = tuple(_lens[d] for d in _default_chunk)

        if not self._dims:
            self._dims = tuple(_dims)

        if not self._shape:
            self._shape = tuple(_lens[d] for d in self._dims)

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
    def squeeze_dims(self):
        """names of dimensions in the underlying pfb file"""
        if self._squeeze_dims is None:
            self._set_dims_and_shape()
        return self._squeeze_dims

    @property
    def default_chunk(self) -> tuple[int]:
        """default chunk to use to read this array with dask"""
        if self._default_chunk is None:
            self._set_dims_and_shape()
        return self._default_chunk

    def _getitem_no_state(self, key):
        """
        Base functionality for actually getting data out of PFB files.
        :param key:
            A key indicating which indices should be read into memory.
        :return:
            A numpy array of the data
        """
        accessor = {
            d: util._key_to_explicit_accessor(k) for d, k in zip(self.dims, key)
        }

        sub = self.reader.read(accessor)

        if self._replace_fill_value and self._fill_value is None:
            self._fill_value = magic_get_fill_value(sub)
            if self._fill_value is None:
                self._replace_fill_value = False

        if self._replace_fill_value:
            sub[sub == self._fill_value] = np.nan

        sub = sub[tuple([accessor[d]["indices"] for d in self.dims])]
        return sub

    def _getitem(self, key: tuple) -> np.ndarray:
        """Mapping between keys to the actual data"""
        sub = self._getitem_no_state(key)
        if self.shape != sub.shape:
            sub = np.squeeze(sub, axis=self.squeeze_dims)
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
