from collections.abc import Iterable
import itertools
from numba import jit
import numpy as np
import struct
from typing import Mapping, List, Union
from numbers import Number
from pprint import pprint


def read_pfb(file: str, mode: str='full', z_first: bool=True):
    """
    Read a single pfb file, and return the data therein

    :param file:
        The file to read.
    :param mode:
        The mode for the reader. See ``ParflowBinaryReader::read_all_subgrids``
        for more information about what modes are available.
    :return:
        An nd array containing the data from the pfb file.
    """
    with ParflowBinaryReader(file) as pfb:
        data = pfb.read_all_subgrids(mode=mode, z_first=z_first)
    return data


def read_stack_of_pfbs(
    file_seq: Iterable[str],
    keys=None,
    z_first: bool=True,
    z_is: str='z'
):
    """
    An efficient wrapper to read a stack of pfb files. This
    approach is faster than looping over the ``read_pfb`` function
    because it caches the subgrid information from the first
    pfb file and then uses that to initialize all other readers.

    :param file_seq:
        An iterable sequence of file names to be read.
    :param keys:
        A set of keys for indexing subarrays of the full pfb. Optional.
        This is mainly a trick for interfacing with xarray, but the format
        of the keys is:

            ::
            {'x': {'start': start_x, 'stop': end_x},
             'y': {'start': start_y, 'stop': end_y},
             'z': {'start': start_z, 'stop': end_z}}

    :param z_first:
        Whether the z dimension should be first. If true returned arrays have
        dimensions ('z', 'y', 'x') else ('x', 'y', 'z')
    :param z_is:
        A descriptor of what the z axis represents. Can be one of
        'z', 'time', 'variable'. Default is 'z'.

    :return:
        An nd array containing the data from the files.
    """
    # Filter out unique files only
    file_seq = list(set(file_seq))
    with ParflowBinaryReader(file_seq[0]) as pfb_init:
        base_header = pfb_init.header
        base_sg_offsets = pfb_init.subgrid_offsets
        base_sg_locations = pfb_init.subgrid_locations
        base_sg_indices = pfb_init.subgrid_start_indices
        base_sg_shapes = pfb_init.subgrid_shapes
        base_sg_chunks = pfb_init.chunks
        base_sg_coords = pfb_init.coords
    if not keys:
        nx, ny, nz = base_header['nx'], base_header['ny'], base_header['nz']
    else:
        start_x = keys['x']['start']
        start_y = keys['y']['start']
        start_z = keys[z_is]['start']
        nx = np.max([keys['x']['stop'] - start_x - 1, 1])
        ny = np.max([keys['y']['stop'] - keys['y']['start'] - 1, 1])
        nz = np.max([keys[z_is]['stop'] - keys[z_is]['start'] - 1, 1])
    #pprint(keys)

    if z_first:
        stack_size = (len(file_seq), nz, ny, nx)
    else:
        stack_size = (len(file_seq), nx, ny, nz)
    pfb_stack = np.empty(stack_size, dtype=np.float64)
    for i, f in enumerate(file_seq):
        with ParflowBinaryReader(
            f, precompute_subgrid_info=False, header=base_header
        ) as pfb:
            pfb.subgrid_offsets = base_sg_offsets
            pfb.subgrid_locations = base_sg_locations
            pfb.subgrid_start_indices = base_sg_indices
            pfb.subgrid_shapes = base_sg_shapes
            pfb.coords = base_sg_coords
            pfb.chunks = base_sg_chunks
            if not keys:
                substack_data = pfb.read_all_subgrids(mode='full', z_first=z_first)
            else:
                substack_data = pfb.read_subarray(
                        start_x, start_y, start_z, nx, ny, nz, z_first=z_first)
            pfb_stack[i, :, : ,:] = substack_data
    if z_is == 'time':
        if z_first:
            pfb_stack = np.concatenate(pfb_stack, axis=0)
        else:
            pfb_stack = np.concatenate(pfb_stack, axis=-1)
    return pfb_stack


class ParflowBinaryReader:
    """
    The ParflowBinaryReader, unsurprisingly, provides functionality
    for reading parflow binary files. It is designed to separate the
    header reading and metadata from subgrids from the reading of the
    underlying subgrid data in an efficient and flexible way. The
    ParflowBinaryReader only ever stores state about the header and/or
    subgrid headers. When reading data it is immediately returned to the
    user in the form of a numpy array. The ParflowBinaryReader implements
    a simple `Context Manager <https://book.pythontips.com/en/latest/context_managers.html>`_
    so it is recommended to use with the standard idiom:

        ::
        with ParflowBinaryReader(file) as pfb:
            data = pfb.read_all_subgrids()

    :param file:
        The pfb file to read
    :param precompute_subgrid_info:
        Whether or not to precompute subgrid information. This defaults to
        ``True`` but can be turned off for reading multiple pfb files to
        reduce the amount of IO overhead when reading a sequence of pfb files.
        This computes the subgrid offset bytes, subgrid locations, subgrid
        indices, subgrid shapes, as well as subgrid coordinates and chunk sizes
    :param p:
        The number of subgrids along the x dimension. This is an optional input,
        if it is not given we will try to precompute it.
    :param q:
        The number of subgrids along the y dimension. This is an optional input,
        if it is not given we will try to precompute it.
    :param r:
        The number of subgrids along the z dimension. This is an optional input,
        if it is not given we will try to precompute it.
    :param header:
        A dictionary representing the header of the pfb file. This is an optional
        input, if it is not given we will read it from the pfb file directly.
    """

    def __init__(
            self,
            file: str,
            precompute_subgrid_info: bool=True,
            p: int=None,
            q: int=None,
            r: int=None,
            header: Mapping[str, Number]=None
    ):
        self.filename = file
        self.f = open(self.filename, 'rb')
        if not header:
            self.header = self.read_header()
        else:
            self.header = header
        self.header['p'] = self.header.get('p', p)
        self.header['q'] = self.header.get('q', q)
        self.header['r'] = self.header.get('r', r)

        # If p, q, and r aren't given we can precompute them
        if not np.all([p, q, r]):
            # NOTE: This is a bit of a fallback and may not always work
            eps = 1 - 1e-6
            first_sg_head = self.read_subgrid_header()
            self.header['p'] = int((self.header['nx'] / first_sg_head['nx']) + eps)
            self.header['q'] = int((self.header['ny'] / first_sg_head['ny']) + eps)
            self.header['r'] = int((self.header['nz'] / first_sg_head['nz']) + eps)

        if precompute_subgrid_info:
            self.compute_subgrid_info()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def compute_subgrid_info(self):
        """ Computes the subgrid information """
        sg_offs, sg_locs, sg_starts, sg_shapes = precalculate_subgrid_info(
                self.header['nx'],
                self.header['ny'],
                self.header['nz'],
                self.header['p'],
                self.header['q'],
                self.header['r'],
                self.header['n_subgrids']
        )
        self.subgrid_offsets = np.array(sg_offs)
        self.subgrid_locations = np.array(sg_locs)
        self.subgrid_start_indices = np.array(sg_starts)
        self.subgrid_shapes = np.array(sg_shapes)
        self.chunks = self._compute_chunks()
        self.coords = self._compute_coords()

    def _compute_chunks(self) -> Mapping[str, tuple]:
        """
        This computes the chunk sizes of the subgrids. Note that it does
        not return a list of length ``n_subgrids`` but rather breaks each
        of the chunks along their primary coordinate axis. Thus you get a
        dictionary full of tuples which looks like:

            ::
            {'x': tuple_with_len_p,
             'y': tuple_with_len_q,
             'z': tuple_with_len_r}
        """
        p, q, r = self.header['p'], self.header['q'], self.header['r'],
        x_chunks = tuple(self.subgrid_shapes[:,0][0:p].flatten())
        y_chunks = tuple(self.subgrid_shapes[:,1][0:p*q:p].flatten())
        z_chunks = tuple(self.subgrid_shapes[:,2][0:p*q*r:p*q].flatten())
        return {'x': x_chunks, 'y': y_chunks, 'z': z_chunks}

    def _compute_coords(self) -> Mapping[str, Iterable[Iterable[int]]]:
        """
        This computes the coordinates of each chunk of the subgrids. Note
        that just like the ``_compute_chunks`` method this returns information
        along the primary coordinate axes. You get a dictionary full of lists
        of coordinate values, which looks like:

            ::
            {'x': [(1,2,...n1),
                   (n1+1, n1+2, ... n1+n2),
                    ... for ni in self.chunks['x']],
             'y': [(1,2,...n1),
                   (n1+1, n1+2, ... n1+n2),
                    ... for ni in self.chunks['y']],
             'z': [(1,2,...n1),
                   (n1+1, n1+2, ... n1+n2),
                    ... for ni in self.chunks['z']],
        """
        coords = {'x': [], 'y': [], 'z': []}
        for c in ['x', 'y', 'z']:
            chunk_start = 0
            for chunk in self.chunks[c]:
                coords[c].append(np.arange(chunk_start, chunk_start + chunk))
                chunk_start += chunk
        return coords

    def read_header(self):
        """Reads the header"""
        self.f.seek(0)
        header = {}
        header['x'] = struct.unpack('>d', self.f.read(8))[0]
        header['y'] = struct.unpack('>d', self.f.read(8))[0]
        header['z'] = struct.unpack('>d', self.f.read(8))[0]
        header['nx'] = struct.unpack('>i', self.f.read(4))[0]
        header['ny'] = struct.unpack('>i', self.f.read(4))[0]
        header['nz'] = struct.unpack('>i', self.f.read(4))[0]
        header['dx'] = struct.unpack('>d', self.f.read(8))[0]
        header['dy'] = struct.unpack('>d', self.f.read(8))[0]
        header['dz'] = struct.unpack('>d', self.f.read(8))[0]
        header['n_subgrids'] = struct.unpack('>i', self.f.read(4))[0]
        return header

    def read_subgrid_header(self, skip_bytes: int=64):
        """Reads a subgrid header at the position ``skip_bytes``"""
        self.f.seek(skip_bytes)
        sg_header = {}
        sg_header['ix'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['iy'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['iz'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['nx'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['ny'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['nz'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['rx'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['ry'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['rz'] = struct.unpack('>i', self.f.read(4))[0]
        sg_header['sg_size'] = np.prod([sg_header[n] for n in ['nx', 'ny', 'nz']])
        return sg_header

    def read_subarray(
            self,
            start_x: int,
            start_y: int,
            start_z: int=0,
            nx: int=1,
            ny: int=1,
            nz: int=None,
            z_first: bool=True
    ) -> np.typing.ArrayLike:
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
            """ Helper to clean up code at the end of this """
            x0 = np.flatnonzero(start == coords)
            x0 = 0 if not x0 else x0[0]
            x1 = np.flatnonzero(end == coords)
            x1 = None if not x1 else x1[0]
            return slice(x0, x1)

        def _get_needed_subgrids(start, end, coords):
            """ Helper function to clean up subgrid selection """
            for s, c in enumerate(coords):
                if start in c: break
            for e, c in enumerate(coords):
                if end in c: break
            return np.arange(s, e+1)

        if not nz:
            nz = self.header['nz']
        end_x = start_x + nx
        end_y = start_y + ny
        end_z = start_z + nz
        p, q, r = self.header['p'], self.header['q'], self.header['r']
        # Convert to numpy array for simpler indexing
        x_coords = np.array(self.coords['x'], dtype=object)
        y_coords = np.array(self.coords['y'], dtype=object)
        z_coords = np.array(self.coords['z'], dtype=object)

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
        full_size = (len(x_sg_coords), len(y_sg_coords), len(z_sg_coords))
        bounding_data = np.empty(full_size, dtype=np.float64)
        subgrid_iter = itertools.product(p_subgrids, q_subgrids, r_subgrids)
        for (xsg, ysg, zsg) in subgrid_iter:
            subgrid_idx = xsg + (p * ysg) + (p * q * zsg)
            # Set up the indices to insert subgrid data into the bounding data
            x0, y0, z0 = self.subgrid_start_indices[subgrid_idx]
            x0, y0, z0 = x0 - x_min, y0 - y_min, z0 - z_min
            dx, dy, dz = self.subgrid_shapes[subgrid_idx]
            x1, y1, z1 = x0 + dx, y0 + dy, z0+ dz
            bounding_data[x0:x1, y0:y1, z0:z1] = self.iloc_subgrid(subgrid_idx)

        # Now clip out the exact part from the bounding box
        clip_x = _get_final_clip(start_x, end_x, x_sg_coords)
        clip_y = _get_final_clip(start_y, end_y, y_sg_coords)
        clip_z = _get_final_clip(start_z, end_z, z_sg_coords)
        if z_first:
            ret_data = bounding_data[clip_x, clip_y, clip_z].T
        else:
            ret_data = bounding_data[clip_x, clip_y, clip_z]
        return ret_data


    def loc_subgrid(self, pp: int, qq: int, rr: int) -> np.typing.ArrayLike:
        """
        Read a subgrid given it's (pp, qq, rr) coordinate in the subgrid-grid.

        :param pp:
            Index in the p subgrid to read.
        :param qq:
            Index in the q subgrid to read.
        :param rr:
            Index in the r subgrid to read.
        :returns:
            The data from the (pp, qq, rr)'th subgrid.
        """
        p, q, r = self.header['p'], self.header['q'], self.header['r']
        subgrid_idx = pp + (p * qq) + (q * p * rr)
        return self.iloc_subgrid(subgrid_idx)

    def iloc_subgrid(self, idx: int) -> np.typing.ArrayLike:
        """
        Read a subgrid at some scalar index.

        :param idx:
            The index of the subgrid to read
        :returns:
            The data from the idx'th subgrid.
        """
        offset = self.subgrid_offsets[idx]
        shape = self.subgrid_shapes[idx]
        return self._backend_iloc_subgrid(offset, shape)

    def _backend_iloc_subgrid(
            self, offset: int, shape: Iterable[int]
    ) -> np.typing.ArrayLike:
        """
        Backend function for memory mapping data from the pfb file on disk.

        :param offset:
            The byte offset to begin reading the sugrid data at.
        :param shape:
            A tuple representing the resulting shape of the subgrid array.
        :returns:
            The data from the subgrid at ``offset` bytes into the file.
        """
        mm = np.memmap(
            self.f,
            dtype=np.float64,
            mode='r',
            offset=offset,
            shape=tuple(shape),
            order='F'
        ).byteswap()
        data = np.array(mm)
        return data

    def read_all_subgrids(
            self, mode: str='full', z_first: bool=True
    ) -> Union[Iterable[np.typing.ArrayLike], np.typing.ArrayLike]:
        """
        Read all of the subgrids in the file.

        :param mode:
            Specifies how to arange the data from the subgrids before returning.
        :param z_first:
            Whether the z dimension should be first. If true returned arrays have
            dimensions ('z', 'y', 'x') else ('x', 'y', 'z')

        :returns:
            A numpy array or iterable of numpy arrays, depending on how ``mode`` is set.
            If ``full`` the returned array will be of dimensions (nx, ny, nz).
            If ``flat`` the returned data will be a list of each fo the subgrid arrays.
            If ``tiled`` the returned data will be a numpy array with dimensions
            (p, q, r) where each index of the array contains the subgrid data which
            also will be numpy array of floats with dimensions (pp_nx, qq_ny, rr_nz) where
            each of pp_nx, qq_ny, and rr_nz are the size of the subgrid array.
        """
        if mode not in ['flat', 'tiled', 'full']:
            raise Exception('mode must be one of flat, tiled, or full')
        if mode in ['flat', 'tiled']:
            all_data = []
            for i in range(self.header['n_subgrids']):
                if z_first:
                    all_data.append(self.iloc_subgrid(i).T)
                else:
                    all_data.append(self.iloc_subgrid(i))
            if mode == 'tiled':
                if z_first:
                    tiled_shape = tuple(self.header[dim] for dim in ['r', 'q', 'p'])
                    all_data = np.array(all_data, dtype=object).reshape(tiled_shape)
                else:
                    tiled_shape = tuple(self.header[dim] for dim in ['p', 'q', 'r'])
                    all_data = np.array(all_data, dtype=object).reshape(tiled_shape)
        elif mode == 'full':
            if z_first:
                full_shape = tuple(self.header[dim] for dim in ['nz', 'ny', 'nx'])
            else:
                full_shape = tuple(self.header[dim] for dim in ['nx', 'ny', 'nz'])
            chunks = self.chunks['x'], self.chunks['y'], self.chunks['z']
            all_data = np.empty(full_shape, dtype=np.float64)
            for i in range(self.header['n_subgrids']):
                nx, ny, nz = self.subgrid_shapes[i]
                ix, iy, iz = self.subgrid_start_indices[i]
                if z_first:
                    all_data[iz:iz+nz, iy:iy+ny, ix:ix+nx] = self.iloc_subgrid(i).T
                else:
                    all_data[ix:ix+nx, iy:iy+ny, iz:iz+nz] = self.iloc_subgrid(i)
        return all_data



@jit(nopython=True)
def get_maingrid_and_remainder(nx, ny, nz, p, q, r):
    nnx = int(nx / p)
    nny = int(ny / q)
    nnz = int(nz / r)
    lx = (nx % p)
    ly = (ny % q)
    lz = (nz % r)
    return nnx, nny, nnz, lx, ly, lz


@jit(nopython=True)
def get_subgrid_loc(sel_subgrid, p, q, r):
    rr = int(np.floor(sel_subgrid / (p * q)))
    qq = int(np.floor((sel_subgrid - (rr*p*q)) / p))
    pp = int(sel_subgrid - rr * (p * q) - (qq * p))
    subgrid_loc = (pp, qq, rr)
    return subgrid_loc


@jit(nopython=True)
def subgrid_lower_left(
    nnx, nny, nnz,
    pp, qq, rr,
    lx, ly, lz
):
    ix = pp * nnx + min(pp, lx)
    iy = qq * nny + min(qq, ly)
    iz = rr * nnz + min(rr, lz)
    return ix, iy, iz


@jit(nopython=True)
def subgrid_size(
    nnx, nny, nnz,
    pp, qq, rr,
    lx, ly, lz
):
    snx = nnx if pp >= lx else nnx+1
    sny = nny if qq >= ly else nny+1
    snz = nnz if rr >= lz else nnz+1
    return snx, sny, snz


@jit(nopython=True)
def precalculate_subgrid_info(nx, ny, nz, p, q, r, n_subgrids):
    subgrid_shapes = []
    subgrid_offsets = []
    subgrid_locs = []
    subgrid_begin_idxs = []
    # Initial size and offset for first subgrid
    snx, sny, snz = 0, 0, 0
    off = 64
    for sg_num in range(n_subgrids):
        # Move past the current header and previous subgrid
        off += 36 +  (8 * (snx * sny * snz))
        subgrid_offsets.append(off)

        nnx, nny, nnz, lx, ly, lz= get_maingrid_and_remainder(nx, ny, nz, p, q, r)
        pp, qq, rr = get_subgrid_loc(sg_num, p, q, r)
        subgrid_locs.append((pp, qq, rr))

        ix, iy, iz = subgrid_lower_left(
            nnx, nny, nnz,
            pp, qq, rr,
            lx, ly, lz
        )
        subgrid_begin_idxs.append((ix, iy, iz))

        snx, sny, snz = subgrid_size(
            nnx, nny, nnz,
            pp, qq, rr,
            lx, ly, lz
        )
        subgrid_shapes.append((snx, sny, snz))
    return subgrid_offsets, subgrid_locs, subgrid_begin_idxs, subgrid_shapes
