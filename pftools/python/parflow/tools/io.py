# -*- coding: utf-8 -*-
"""io module

Helper functions to load or write files

A note on naming conventions in the python IO interfaces
as compared to the C implementation for reading/writing
pfb files:
+-------------------------+------------------+---------------------+
| Description             | Parflow C        | Parflow Python      |
+-------------------------+------------------+---------------------+
| Subgrid lengths         | P, Q, R          | p, q, r             |
| Subgrid indices         | p, q, r          | sg_p, sg_q, sg_r    |
| Size of the full grid   | NX, NY, NZ       | nx, ny, nz          |
| Size of a subgrid       | nx, ny, nz       | sg_nx, sg_ny, sg_nz |
| Cell size of the grid   | dx, dy, dz       | dx, dy, dz          |
| Lower left index        | ix, iy, iz       | ix, iy, iz          |
+-------------------------+------------------+---------------------+
"""

from functools import partial
import itertools
import json
from pathlib import Path
try:
    from numba import jit, njit
except ImportError:
    # Some systems may not have numba capabilities
    def jit(*args, **kwargs):
        """Dummy decorator, does nothing"""
        def _decorator(func):
            return func
        return _decorator

from numbers import Number
import pandas as pd
import numpy as np
import struct
from typing import Mapping, List, Union, Iterable
import yaml

from .hydrology import (
    calculate_evapotranspiration,
    calculate_overland_flow,
    calculate_overland_flow_grid,
    calculate_subsurface_storage,
    calculate_surface_storage,
    calculate_water_table_depth,
)
from .fs import get_absolute_path
from .helper import sort_dict, get_or_create_dict

try:
    from yaml import CDumper as YAMLDumper
except ImportError:
    from yaml import Dumper as YAMLDumper


def read_pfb(file: str, keys: dict=None, mode: str='full', z_first: bool=True,
             read_sg_info: bool=True):
    """
    Read a single pfb file, and return the data therein

    :param file:
        The file to read.
    :param keys:
        A set of keys for indexing subarrays of the full pfb. Optional.
        This is mainly a trick for interfacing with xarray, but the format
        of the keys is:

            ::
            {'x': {'start': start_x, 'stop': end_x},
             'y': {'start': start_y, 'stop': end_y},
             'z': {'start': start_z, 'stop': end_z}}

    :param mode:
        The mode for the reader. See ``ParflowBinaryReader::read_all_subgrids``
        for more information about what modes are available.
    :param read_sg_info:
        Precalculating subgrid information does not always work correctly for
        some files, especially velocity files. If ``True``, read subgrid info 
        directly from the pfb file (slower, but more robust)
    :return:
        An nd array containing the data from the pfb file.
    """
    with ParflowBinaryReader(file, read_sg_info=read_sg_info) as pfb:
        if not keys:
            data = pfb.read_all_subgrids(mode=mode, z_first=z_first)
        else:
            base_header = pfb.header
            start_x = keys.get('x', {}).get('start', None) or 0
            start_y = keys.get('y', {}).get('start', None) or 0
            start_z = keys.get('z', {}).get('start', None) or 0
            stop_x =  keys.get('x', {}).get('stop', None) or base_header['nx']
            stop_y =  keys.get('y', {}).get('stop', None) or base_header['ny']
            stop_z =  keys.get('z', {}).get('stop', None) or base_header['nz']
            nx = np.max([stop_x - start_x, 1])
            ny = np.max([stop_y - start_y, 1])
            nz = np.max([stop_z - start_z, 1])
            data = pfb.read_subarray(
                        start_x, start_y, start_z, nx, ny, nz, z_first=z_first)
    return data


# -----------------------------------------------------------------------------

def write_pfb(
    file, array,
    p=1, q=1, r=1,
    x=0.0, y=0.0, z=0.0,
    dx=1.0, dy=1.0, dz=1.0,
    z_first=True, dist=True,
    **kwargs
):
    """
    Write a single pfb file. The data must be a 3D numpy array with float64
    values. The number of subgrids in the saved file will be p * q * r. This
    is regardless of the number of subgrids in the PFB file loaded by the
    ParflowBinaryReader into the numpy array. Therefore, loading a file with
    ParflowBinaryReader and saving it with this method may restructure the
    file into a different number of subgrids if you change these values.

    If dist is True then also write a file with the .dist extension added to
    the file_name. The .dist file will contain one line per subgrid with the
    offset of the subgrid in the .pfb file.

    :param file:
        The name of the file to write the array to.
    :param array:
        The array to write.
    :param p:
        Number of subgrids in the x direction.
    :param q:
        Number of subgrids in the y direction.
    :param r:
        Number of subgrids in the z direction.
    :param x:
        The length of the x-axis
    :param y:
        The length of the y-axis
    :param z:
        The length of the z-axis
    :param dx:
        The spacing between cells in the x direction
    :param dy:
        The spacing between cells in the y direction
    :param dz:
        The spacing between cells in the z direction
    :param z_first:
        Whether the z-axis should be first or last.
    :param dist:
        Whether to write the distfile in addition to the pfb.
    :param kwargs:
        Extra keyword arguments, primarily to eat unnecessary
        args by passing in a dictionary with `**dict`.
    """
    if array.dtype != np.float64:
        raise ValueError(f"Arrays written to pfb must be of type np.float64!"
                          + " Found {array.dtype} instead!")

    if len(array.shape) == 3:
        if z_first:
                nz, ny, nx = array.shape
        else:
            nx, ny, nz = array.shape
    elif len(array.shape) == 2:
        nz = 1
        ny, nx = array.shape
        array = np.expand_dims(array, axis=0 if z_first else -1)
    else:
        raise ValueError("Array must be 2 or 3 dimensional to write to pfb!")

    n_subgrids = p * q * r
    # All the subgrid info here
    sg_offs, sg_locs, sg_starts, sg_shapes = precalculate_subgrid_info(
        nx, ny, nz, p, q, r
    )

    with open(file, 'wb') as f:
        # Write the file header
        f.write(struct.pack('>d', float(x)))
        f.write(struct.pack('>d', float(y)))
        f.write(struct.pack('>d', float(z)))
        f.write(struct.pack('>i', int(nx)))
        f.write(struct.pack('>i', int(ny)))
        f.write(struct.pack('>i', int(nz)))
        f.write(struct.pack('>d', float(dx)))
        f.write(struct.pack('>d', float(dy)))
        f.write(struct.pack('>d', float(dz)))
        f.write(struct.pack('>i', int(n_subgrids)))

        # loop over subgrids:
        for off, loc, start, shape in zip(sg_offs, sg_locs, sg_starts, sg_shapes):
            # Write the subgrid header
            for sgh in itertools.chain(start, shape, [1,1,1]):
                f.write(struct.pack('>i', int(sgh)))

            mm = np.memmap(
                file,
                dtype=np.float64,
                mode='r+',
                offset=off,
                shape=tuple(shape),
                order='F'
            )
            # Gather shapes & extents
            s_ix, s_iy, s_iz = start
            n_ix, n_iy, n_iz = shape
            if z_first:
                mm[:] = array[s_iz:s_iz+n_iz,
                              s_iy:s_iy+n_iy,
                              s_ix:s_ix+n_ix].T.byteswap()
            else:
                mm[:] = array[s_ix:s_ix+n_ix,
                              s_iy:s_iy+n_iy,
                              s_iz:s_iz+n_iz].byteswap()
            # Seek to offset + the size of the subgrid
            f.seek(off + 8 * np.prod(shape))

    # Create the .dist file if requested
    if dist:
        write_dist(file, sg_offs)


def write_dist(file, sg_offs):
    """
    Write a distfile.

    :param file:
        The path of the file to be written.
    :param sg_offs:
        The subgrid offsets.
    """
    with open(file + ".dist", "w+") as dist_fp:
        for i, s in enumerate(sg_offs):
            # Need to account for header bytes
            real_off = s - 100 if i == 0 else s - 36
            dist_fp.write(f'{real_off}\n')


# -----------------------------------------------------------------------------

def read_pfb_sequence(
    file_seq: Iterable[str],
    keys=None,
    z_first: bool=True,
    z_is: str='z',
    read_sg_info: bool=False
):
    """
    An efficient wrapper to read a sequence of pfb files. This
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
    :param read_sg_info:
        Precalculating subgrid information does not always work correctly for
        some files, especially velocity files. If ``True``, read subgrid info 
        directly from the pfb file (slower, but more robust)
    
    :return:
        An nd array containing the data from the files.
    """
    # Filter out unique files only
    file_seq = sorted(list(set(file_seq)))
    with ParflowBinaryReader(file_seq[0], read_sg_info=read_sg_info) as pfb_init:
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
        start_x = keys.get('x', {}).get('start', None) or 0
        start_y = keys.get('y', {}).get('start', None) or 0
        start_z = keys.get(z_is, {}).get('start', None) or 0
        stop_x =  keys.get('x', {}).get('stop', None) or base_header['nx']
        stop_y =  keys.get('y', {}).get('stop', None) or base_header['ny']
        stop_z =  keys.get(z_is, {}).get('stop', None) or base_header['nz']
        nx = np.max([stop_x - start_x, 1])
        ny = np.max([stop_y - start_y, 1])
        nz = np.max([stop_z - start_z, 1])

    n_seq = len(file_seq)
    if z_first:
        seq_size = (len(file_seq), nz, ny, nx)
    else:
        seq_size = (len(file_seq), nx, ny, nz)
    pfb_seq = np.empty(seq_size, dtype=np.float64)
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
                subseq_data = pfb.read_all_subgrids(mode='full', z_first=z_first)
            else:
                subseq_data = pfb.read_subarray(
                        start_x, start_y, start_z, nx, ny, nz, z_first=z_first)
            pfb_seq[i, :, : ,:] = subseq_data
    if z_is == 'time':
        if z_first:
            pfb_seq = np.concatenate(pfb_seq, axis=0)
        else:
            pfb_seq = np.concatenate(pfb_seq, axis=-1)
    return pfb_seq


# -----------------------------------------------------------------------------

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
    :param read_sg_info:
        Whether or not to read subgrid information directly from the pfb file.
        Precalculating subgrid information does not always work correctly for
        some files, especially velocity files. This reads the subgrid offset 
        bytes, subgrid indices and subgrid shapes, then computes the subgrid 
        coordinates subgrid locations, and chunk sizes as normal
        
    """

    def __init__(
        self,
        file: str,
        precompute_subgrid_info: bool=True,
        p: int=None,
        q: int=None,
        r: int=None,
        header: Mapping[str, Number]=None,
        read_sg_info: bool=False
    ):
        self.filename = file
        self.f = open(self.filename, 'rb')
        if not header:
            self.header = self.read_header()
        else:
            self.header = header

        if np.all([p, q, r]):
            self.header['p'] = p
            self.header['q'] = q
            self.header['r'] = r

        self.read_subgrid_info()
            
    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.f.close()

    def compute_subgrid_info(self):
        """ Computes the subgrid information """
        try:
            sg_offs, sg_locs, sg_starts, sg_shapes = precalculate_subgrid_info(
                self.header['nx'],
                self.header['ny'],
                self.header['nz'],
                self.header['p'],
                self.header['q'],
                self.header['r']
            )
        except:
            raise ValueError(self.header)
        self.subgrid_offsets = np.array(sg_offs)
        self.subgrid_locations = np.array(sg_locs)
        self.subgrid_start_indices = np.array(sg_starts)
        self.subgrid_shapes = np.array(sg_shapes)
        self.chunks = self._compute_chunks()
        self.coords = self._compute_coords()

    def read_subgrid_info(self):
        """ Read the header for each subgrid directly from the pfb file, rather 
        than calculating it via precalculate_subgrid_info """
        
        sg_shapes = []
        sg_offs = []
        sg_locs = []
        sg_starts = []

        off = 64
        for sg_num in range(self.header['n_subgrids']):
            # Read and move past the current subgrid header
            sg_head = self.read_subgrid_header(off)
            off += 36 
            
            sg_starts.append([sg_head['ix'], sg_head['iy'], sg_head['iz']])
            sg_shapes.append([sg_head['nx'], sg_head['ny'], sg_head['nz']])
            sg_offs.append(off)
            
            # Finally, move past the current subgrid before next iteration
            off += sg_head['sg_size']*8

        # Calculate p, q, r from subgrid shapes
        p, q, r = 0, 0, 0
        x, y, z = 0, 0, 0
        
        for shape in sg_shapes:
            if x == self.header['nx']:
                break
            p = p + 1
            x = x + shape[0]
            
        for shape in sg_shapes[::p]:
            if y == self.header['ny']:
                break
            q = q + 1
            y = y + shape[1]

        for shape in sg_shapes[::p * q]:
            if z == self.header['nz']:
                break
            r = r + 1
            z = z + shape[2]

        self.header['p'] = p
        self.header['q'] = q
        self.header['r'] = r
            
        for sg_num in range(self.header['n_subgrids']):            
            # Calculate subgrid locs instead of reading from file
            sg_p, sg_q, sg_r = get_subgrid_loc(sg_num, self.header['p'], 
                                                       self.header['q'], 
                                                       self.header['r'])
            sg_locs.append((sg_p, sg_q, sg_r))
        
        
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
        header_len = 9

        # Apologies, this is kind of ugly, but faster than using struct
        (sg_header['ix'],
         sg_header['iy'],
         sg_header['iz'],
         sg_header['nx'],
         sg_header['ny'],
         sg_header['nz'],
         sg_header['rx'],
         sg_header['ry'],
         sg_header['rz']) = np.memmap(self.f,
                                      dtype=np.int32,
                                      offset=skip_bytes,
                                      mode='r',
                                      shape=(header_len,),
                                      order='F').byteswap()
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
            """ Helper to clean up code at the end of this """
            x0 = np.flatnonzero(start == coords)
            x0 = 0 if not x0 else x0[0]
            x1 = np.flatnonzero(end == coords)
            x1 = None if x1 is None or len(x1) == 0 else x1[0]
            return slice(x0, x1)

        def _get_needed_subgrids(start, end, coords):
            """ Helper function to clean up subgrid selection """
            for s, c in enumerate(coords):
                if start in c: break
            for e, c in enumerate(coords):
                if end in c: break
            return np.arange(s, e+1)

        if not start_x:
            start_x = 0
        if not start_y:
            start_y = 0
        if not start_z:
            start_z = 0
        if not nx:
            nx = self.header['nx']
        if not ny:
            ny = self.header['ny']
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


    def loc_subgrid(self, sg_p: int, sg_q: int, sg_r: int) -> np.ndarray:
        """
        Read a subgrid given it's (sg_p, sg_q, sg_r) coordinate in the subgrid-grid.

        :param sg_p:
            Index in the p subgrid to read.
        :param sg_q:
            Index in the q subgrid to read.
        :param sg_r:
            Index in the r subgrid to read.
        :returns:
            The data from the (sg_p, sg_q, sg_r)'th subgrid.
        """
        p, q, r = self.header['p'], self.header['q'], self.header['r']
        subgrid_idx = sg_p + (p * sg_q) + (q * p * sg_r)
        return self.iloc_subgrid(subgrid_idx)

    def iloc_subgrid(self, idx: int) -> np.ndarray:
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
    ) -> np.ndarray:
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
    ) -> Union[Iterable[np.ndarray], np.ndarray]:
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
            also will be numpy array of floats with dimensions (sg_nx, sg_ny, sg_nz) where
            each of sg_nx, sg_ny, and sg_nz are the size of the subgrid array.
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


# -----------------------------------------------------------------------------

@jit(nopython=True)
def get_maingrid_and_remainder(nx, ny, nz, p, q, r):
    """
    Determines the sizes of the subgrids. Maingrid
    sizes are simprm_ny the integer value of the number
    of cells divided by the number of subgrids along
    each axis. The remainder is the modulus.

    :param nx:
        The length of the array along the x axis.
    :param ny:
        The length of the array along the y axis.
    :param nz:
        The length of the array along the z axis.
    :param p:
        The number of subgrids along the x axis.
    :param q:
        The number of subgrids along the y axis.
    :param r:
        The number of subgrids along the z axis.
    """
    mg_nx = int(nx / p)
    mg_ny = int(ny / q)
    mg_nz = int(nz / r)
    rm_nx = (nx % p)
    rm_ny = (ny % q)
    rm_nz = (nz % r)
    return mg_nx, mg_ny, mg_nz, rm_nx, rm_ny, rm_nz


# -----------------------------------------------------------------------------

@jit(nopython=True)
def get_subgrid_loc(sel_subgrid, p, q, r):
    """
    Translate an integer subgrid to the location in 3d subgrid space.

    :param sel_subgrid:
        The scalar index of the subgrid of interest.
    :param p:
        The number of subgrids along the x axis.
    :param q:
        The number of subgrids along the y axis.
    :param r:
        The number of subgrids along the z axis.
    """
    sg_r = int(np.floor(sel_subgrid / (p * q)))
    sg_q = int(np.floor((sel_subgrid - (sg_r*p*q)) / p))
    sg_p = int(sel_subgrid - sg_r * (p * q) - (sg_q * p))
    subgrid_loc = (sg_p, sg_q, sg_r)
    return subgrid_loc


# -----------------------------------------------------------------------------

@jit(nopython=True)
def subgrid_lower_left(
    mg_nx, mg_ny, mg_nz,
    sg_p, sg_q, sg_r,
    rm_nx, rm_ny, rm_nz
):
    """
    Get the index of the lower left corner of a subgrid.

    :param mg_nx:
        The standard length of a subgrid on the x-axis
    :param mg_ny:
        The standard length of a subgrid on the y-axis
    :param mg_nz:
        The standard length of a subgrid on the z-axis
    :param sg_p:
        The index of the subgrid on the x-axis.
    :param sg_q:
        The index of the subgrid on the y-axis.
    :param sg_r:
        The index of the subgrid on the z-axis.
    :param rm_nx:
        Remainder from the maingrid calculation on the x-axis
    :param rm_ny:
        Remainder from the maingrid calculation on the y-axis
    :param rm_nz:
        Remainder from the maingrid calculation on the z-axis
    """
    ix = sg_p * mg_nx + min(sg_p, rm_nx)
    iy = sg_q * mg_ny + min(sg_q, rm_ny)
    iz = sg_r * mg_nz + min(sg_r, rm_nz)
    return ix, iy, iz


# -----------------------------------------------------------------------------

@jit(nopython=True)
def subgrid_size(
    mg_nx, mg_ny, mg_nz,
    sg_p, sg_q, sg_r,
    rm_nx, rm_ny, rm_nz
):
    """
    Get the size of a subgrid

    :param mg_nx:
        The standard length of a subgrid on the x-axis
    :param mg_ny:
        The standard length of a subgrid on the y-axis
    :param mg_nz:
        The standard length of a subgrid on the z-axis
    :param sg_p:
        The index of the subgrid on the x-axis.
    :param sg_q:
        The index of the subgrid on the y-axis.
    :param sg_r:
        The index of the subgrid on the z-axis.
    :param rm_nx:
        Remainder from the maingrid calculation on the x-axis
    :param rm_ny:
        Remainder from the maingrid calculation on the y-axis
    :param rm_nz:
        Remainder from the maingrid calculation on the z-axis
    """
    sg_nx = mg_nx if sg_p >= rm_nx else mg_nx+1
    sg_ny = mg_ny if sg_q >= rm_ny else mg_ny+1
    sg_nz = mg_nz if sg_r >= rm_nz else mg_nz+1
    return sg_nx, sg_ny, sg_nz


# -----------------------------------------------------------------------------

@jit(nopython=True)
def precalculate_subgrid_info(nx, ny, nz, p, q, r):
    """
    Computes all necessary subgrid information to index and read
    or write a pfb file.

    :param nx:
        Number of cells along the x-axis.
    :param ny:
        Number of cells along the y-axis.
    :param nz:
        Number of cells along the z-axis.
    :param p:
        Number of subgrids along the x-axis.
    :param q:
        Number of subgrids along the y-axis.
    :param r:
        Number of subgrids along the z-axis.

    :return:
        A tuple of arrays, (
            subgrid_offsets,
            subgid_locations,
            subgrid_lower_left_indices,
            subgrid_shapes
        )
    """
    subgrid_shapes = []
    subgrid_offsets = []
    subgrid_locs = []
    subgrid_begin_idxs = []
    # Initial size and offset for first subgrid
    sg_nx, sg_ny, sg_nz = 0, 0, 0
    n_subgrids = p * q * r
    off = 64
    for sg_num in range(n_subgrids):
        # Move past the current header and previous subgrid
        off += 36 +  (8 * (sg_nx * sg_ny * sg_nz))
        subgrid_offsets.append(off)

        (mg_nx, mg_ny, mg_nz,
         rm_nx, rm_ny, rm_nz) = get_maingrid_and_remainder(nx, ny, nz, p, q, r)
        sg_p, sg_q, sg_r = get_subgrid_loc(sg_num, p, q, r)
        subgrid_locs.append((sg_p, sg_q, sg_r))

        ix, iy, iz = subgrid_lower_left(
            mg_nx, mg_ny, mg_nz,
            sg_p, sg_q, sg_r,
            rm_nx, rm_ny, rm_nz
        )
        subgrid_begin_idxs.append((ix, iy, iz))

        sg_nx, sg_ny, sg_nz = subgrid_size(
            mg_nx, mg_ny, mg_nz,
            sg_p, sg_q, sg_r,
            rm_nx, rm_ny, rm_nz
        )
        subgrid_shapes.append((sg_nx, sg_ny, sg_nz))
    return subgrid_offsets, subgrid_locs, subgrid_begin_idxs, subgrid_shapes


# -----------------------------------------------------------------------------

def load_patch_matrix_from_image_file(file_name, color_to_patch=None,
                                      fall_back_id=0):
    import imageio

    im = imageio.imread(file_name)
    height, width, color = im.shape
    matrix = np.zeros((height, width), dtype=np.int16)
    if color_to_patch is None:
        for j in range(height):
            for i in range(width):
                if im[j, i, 0] != 255:
                    matrix[j, i] = 1
    else:
        size1 = set()
        size2 = set()
        size3 = set()
        colors = []

        def _to_key(c, num):
            return ','.join([f'{c[i]}' for i in range(num)])

        to_key_1 = partial(_to_key, num=1)
        to_key_2 = partial(_to_key, num=2)
        to_key_3 = partial(_to_key, num=3)

        for key, value in color_to_patch.items():
            hex_color = key.lstrip('#')
            color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            colors.append((color, value))
            size1.add(to_key_1(color))
            size2.add(to_key_2(color))
            size3.add(to_key_3(color))

        to_key = None
        if len(colors) == len(size3):
            to_key = to_key_3
        if len(colors) == len(size2):
            to_key = to_key_2
        if len(colors) == len(size1):
            to_key = to_key_1

        print(f'Sizes: colors({len(colors)}), 1({len(size1)}), '
              f'2({len(size2)}), 3({len(size3)})')

        if to_key is None:
            raise Exception('You have duplicate colors')

        fast_map = {}
        for color_patch in colors:
            fast_map[to_key(color_patch[0])] = color_patch[1]

        for j in range(height):
            for i in range(width):
                key = to_key(im[j, i])
                try:
                    matrix[j, i] = fast_map[key]
                except Exception:
                    matrix[j, i] = fall_back_id

    return np.flip(matrix, 0)


# -----------------------------------------------------------------------------

def load_patch_matrix_from_asc_file(file_name):
    ncols = -1
    nrows = -1
    in_header = True
    nb_line_to_skip = 0
    with open(file_name) as f:
        while in_header:
            line = f.readline()
            try:
                int(line)
                in_header = False
            except Exception:
                key, value = line.split()
                if key == 'ncols':
                    ncols = int(value)
                if key == 'nrows':
                    nrows = int(value)
                nb_line_to_skip += 1

    matrix = np.loadtxt(file_name, skiprows=nb_line_to_skip, dtype=np.int16)
    matrix.shape = (nrows, ncols)

    return np.flip(matrix, 0)


# -----------------------------------------------------------------------------

def load_patch_matrix_from_sa_file(file_name):
    i_size = -1
    j_size = -1
    k_size = -1
    with open(file_name) as f:
        i_size, j_size, k_size = map(int, f.readline().split())

    matrix = np.loadtxt(file_name, skiprows=1, dtype=np.int16)
    matrix.shape = (j_size, i_size)
    return matrix


# -----------------------------------------------------------------------------

def write_patch_matrix_as_asc(matrix, file_name, xllcorner=0.0, yllcorner=0.0,
                              cellsize=1.0, NODATA_value=0, **kwargs):
    """Write asc for pfsol"""
    height, width = matrix.shape
    with open(file_name, 'w') as out:
        out.write(f'ncols          {width}\n')
        out.write(f'nrows          {height}\n')
        out.write(f'xllcorner      {xllcorner}\n')
        out.write(f'yllcorner      {yllcorner}\n')
        out.write(f'cellsize       {cellsize}\n')
        out.write(f'NODATA_value   {NODATA_value}\n')
        # asc are vertically flipped
        for j in range(height):
            for i in range(width):
                out.write(f'{matrix[height - j - 1, i]}\n')


# -----------------------------------------------------------------------------

def write_patch_matrix_as_sa(matrix, file_name, **kwargs):
    """Write asc for pfsol"""
    nrows, ncols = matrix.shape
    with open(file_name, 'w') as out:
        out.write(f'{ncols} {nrows} 1\n')
        it = np.nditer(matrix)
        for value in it:
            out.write(f'{value}\n')


# -----------------------------------------------------------------------------

def write_dict_as_pfidb(dict_obj, file_name):
    """Write a Python dict in a pfidb format inside the provided file_name
    """
    with open(file_name, 'w') as out:
        out.write(f'{len(dict_obj)}\n')
        for key in dict_obj:
            out.write(f'{len(key)}\n')
            out.write(f'{key}\n')
            value = dict_obj[key]
            out.write(f'{len(str(value))}\n')
            out.write(f'{str(value)}\n')


# -----------------------------------------------------------------------------

def write_dict_as_yaml(dict_obj, file_name):
    """Write a Python dict in a pfidb format inside the provided file_name
    """
    yaml_obj = {}
    overriden_keys = {}
    for key, value in dict_obj.items():
        keys_path = key.split('.')
        get_or_create_dict(
            yaml_obj, keys_path[:-1], overriden_keys)[keys_path[-1]] = value

    # Push value back to yaml
    for key, value in overriden_keys.items():
        keys_path = key.split('.')
        value_obj = get_or_create_dict(yaml_obj, keys_path, {})
        value_obj['_value_'] = value

    output = yaml.dump(sort_dict(yaml_obj), Dumper=YAMLDumper)
    Path(file_name).write_text(output)


# -----------------------------------------------------------------------------

def write_dict_as_json(dict_obj, file_name):
    """Write a Python dict in a json format inside the provided file_name
    """
    Path(file_name).write_text(json.dumps(dict_obj, indent=2))


# -----------------------------------------------------------------------------

def write_dict(dict_obj, file_name):
    """Write a Python dict into a file_name using the extension to
    determine its format.
    """
    # Always write a sorted dictionary
    sorted_dict = sort_dict(dict_obj)

    ext = Path(file_name).suffix[1:].lower()
    if ext in ['yaml', 'yml']:
        write_dict_as_yaml(sorted_dict, file_name)
    elif ext == 'pfidb':
        write_dict_as_pfidb(sorted_dict, file_name)
    elif ext == 'json':
        write_dict_as_json(sorted_dict, file_name)
    else:
        raise Exception(f'Could not find writer for {file_name}')


# -----------------------------------------------------------------------------

def to_native_type(string):
    """Converting a string to a value in native format.
    Used for converting .pfidb files
    """
    types_to_try = [int, float]
    for t in types_to_try:
        try:
            return t(string)
        except ValueError:
            pass

    # Handle boolean type
    lower_str = string.lower()
    if lower_str in ['true', 'false']:
        return lower_str[0] == 't'

    return string


# -----------------------------------------------------------------------------

def read_pfidb(file_path):
    """Load pfidb file into a Python dict
    """
    result_dict = {}
    action = 'nb_lines'  # nb_lines, size, string
    size = 0
    key = ''
    value = ''
    string_type_count = 0
    full_path = get_absolute_path(file_path)

    with open(full_path, 'r') as input_file:
        for line in input_file:
            if action == 'string':
                if string_type_count % 2 == 0:
                    key = line[:size]
                else:
                    value = line[:size]
                    result_dict[key] = to_native_type(value)
                string_type_count += 1
                action = 'size'

            elif action == 'size':
                size = int(line)
                action = 'string'

            elif action == 'nb_lines':
                action = 'size'

    return result_dict


# -----------------------------------------------------------------------------

def read_yaml(file_path):
    """Load yaml file into a Python dict
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    return yaml.safe_load(path.read_text())


# -----------------------------------------------------------------------------

def _read_clmin(file_name):
    """function to load in drv_clmin.dat files

       Args:
           - file_name: name of drv_clmin.dat file

       Returns:
           dictionary of key/value pairs of variables in file
    """
    clm_vars = {}
    with open(file_name, 'r') as rf:
        for line in rf:
            # skip if first 15 are empty or exclamation
            if line and line[0].islower():
                first_word = line.split()[0]
                if len(first_word) > 15:
                    clm_vars[first_word[:14]] = first_word[15:]
                else:
                    clm_vars[first_word] = line.split()[1]

    return clm_vars


# -----------------------------------------------------------------------------

def _read_vegm(file_name):
    """function to load in drv_vegm.dat files

       Args:
           - file_name: name of drv_vegm.dat file

       Returns:
           3D numpy array for domain, with 3rd dimension defining each column
           in the vegm.dat file except for x/y
    """
    # Assume first two lines are comments and use generic column names
    df = pd.read_csv(file_name, delim_whitespace=True, skiprows=2, header=None)
    df.columns = [f'c{i}' for i in range(df.shape[1])]

    # Number of columns and rows determined by last line of file
    nx = int(df.iloc[-1]['c0'])
    ny = int(df.iloc[-1]['c1'])
    # Don't use 'x' and 'y' columns
    feature_cols = df.columns[2:]
    # Stack everything into (ny, nx, n_features)
    vegm_array = np.stack(
        [df[c].values.reshape((ny, nx)) for c in feature_cols], axis=-1
    )
    return vegm_array


# -----------------------------------------------------------------------------

def _read_vegp(file_name):
    """function to load in drv_vegp.dat files

       Args:
           - file_name: name of drv_vegp.dat file

       Returns:
           Dictionary with keys as variables and values as lists of parameter
           values for each of the 18 land cover types
    """
    vegp_data = {}
    current_var = None
    with open(file_name, 'r') as rf:
        for line in rf:
            if not line or line[0] == '!':
                continue

            split = line.split()
            if current_var is not None:
                vegp_data[current_var] = [to_native_type(i) for i in split]
                current_var = None
            elif line[0].islower():
                current_var = split[0]

    return vegp_data


# -----------------------------------------------------------------------------

def read_clm(file_name, type='clmin'):
    type_map = {
        'clmin': _read_clmin,
        'vegm': _read_vegm,
        'vegp': _read_vegp
    }

    if type not in type_map:
        raise Exception(f'Unknown clm type: {type}')

    return type_map[type](get_absolute_path(file_name))


# -----------------------------------------------------------------------------


class DataAccessor:
    """Helper for extracting numpy array from a given run"""

    def __init__(self, run, selector=None):
        """Create DataAccessor from a Run instance"""
        self._run = run
        self._name = run.get_name()
        self._selector = selector
        self._t_padding = 5
        self._time = None
        self._ts = None
        # CLM
        self._forcing_time = 0
        self._process_id = 0
        # Initialize time
        self.time = 0

    # ---------------------------------------------------------------------------

    def _pfb_to_array(self, file_path):
        return read_pfb(get_absolute_path(file_path))

    # ---------------------------------------------------------------------------
    # time
    # ---------------------------------------------------------------------------

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t):
        self._time = int(t)
        self._ts = f'{self._time:0>{self._t_padding}}'

    @property
    def times(self):
        t0 = self._run.TimingInfo.StartCount
        t_start = self._run.TimingInfo.StartTime
        t_end = self._run.TimingInfo.StopTime
        t_step = self._run.TimeStep.Value
        t = t0 + t_start
        time_values = []
        while t <= t_end:
            time_values.append(int(t))
            t += t_step

        return time_values

    # ---------------------------------------------------------------------------
    # forcing time
    # ---------------------------------------------------------------------------

    @property
    def forcing_time(self):
        return self._forcing_time

    @forcing_time.setter
    def forcing_time(self, t):
        self._forcing_time = int(t)

    # ---------------------------------------------------------------------------
    # Process id
    # ---------------------------------------------------------------------------

    @property
    def process_id(self):
        return self._process_id

    @process_id.setter
    def process_id(self, t):
        self._process_id = int(t)

    # ---------------------------------------------------------------------------
    # Region selector
    # ---------------------------------------------------------------------------

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, selector):
        self._selector = selector

    # ---------------------------------------------------------------------------
    # Grid information
    # ---------------------------------------------------------------------------

    @property
    def shape(self):
        # FIXME do something with selector
        return (
            self._run.ComputationalGrid.NZ,
            self._run.ComputationalGrid.NY,
            self._run.ComputationalGrid.NX
        )

    @property
    def dx(self):
        return self._run.ComputationalGrid.DX

    @property
    def dy(self):
        return self._run.ComputationalGrid.DY

    @property
    def dz(self):
        if self._run.Solver.Nonlinear.VariableDz:
            assert self._run.dzScale.Type == 'nzList'
            dz_scale = []
            for i in range(self._run.dzScale.nzListNumber):
                dz_scale.append(self._run.Cell[str(i)]['dzScale']['Value'])
            dz_scale = np.array(dz_scale)
        else:
            dz_scale = np.ones((self._run.ComputationalGrid.NZ,))

        dz_values = dz_scale * self._run.ComputationalGrid.DZ
        return dz_values

    # ---------------------------------------------------------------------------
    # Mannings Roughness Coef
    # ---------------------------------------------------------------------------

    @property
    def mannings(self):
        return self._pfb_to_array(f'{self._name}.out.mannings.pfb')

    # ---------------------------------------------------------------------------
    # Mask
    # ---------------------------------------------------------------------------

    @property
    def mask(self):
        return self._pfb_to_array(f'{self._name}.out.mask.pfb')

    # ---------------------------------------------------------------------------
    # Slopes X Y
    # ---------------------------------------------------------------------------

    @property
    def slope_x(self):
        if self._run.TopoSlopesX.FileName is None:
            return self._pfb_to_array(f'{self._name}.out.slope_x.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopesX.FileName)

    @property
    def slope_y(self):
        if self._run.TopoSlopesY.FileName is None:
            return self._pfb_to_array(f'{self._name}.out.slope_y.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopesY.FileName)

    # ---------------------------------------------------------------------------
    # Elevation
    # ---------------------------------------------------------------------------

    @property
    def elevation(self):
        if self._run.TopoSlopes.Elevation.FileName is None:
            return self._pfb_to_array(f'{self._name}.DEM.pfb')
        else:
            return self._pfb_to_array(self._run.TopoSlopes.Elevation.FileName)

    # ---------------------------------------------------------------------------
    # Computed Porosity
    # ---------------------------------------------------------------------------

    @property
    def computed_porosity(self):
        return self._pfb_to_array(f'{self._name}.out.porosity.pfb')

    # ---------------------------------------------------------------------------
    # Computed Permeability
    # ---------------------------------------------------------------------------

    @property
    def computed_permeability_x(self):
        return self._pfb_to_array(f'{self._name}.out.perm_x.pfb')

    @property
    def computed_permeability_y(self):
        return self._pfb_to_array(f'{self._name}.out.perm_y.pfb')

    @property
    def computed_permeability_z(self):
        return self._pfb_to_array(f'{self._name}.out.perm_z.pfb')

    # ---------------------------------------------------------------------------
    # Pressures
    # ---------------------------------------------------------------------------

    @property
    def pressure_initial_condition(self):
        press_type = self._run.ICPressure.Type
        if press_type == 'PFBFile':
            geom_name = self._run.ICPressure.GeomNames
            if len(geom_name) > 1:
                msg = f'ICPressure.GeomNames are set to {geom_name}'
                raise Exception(msg)
            file_name = self._run.Geom[geom_name[0]].ICPressure.FileName
            return self._pfb_to_array(file_name)
        else:
            # HydroStaticPatch, ... ?
            msg = f'Initial pressure of type {press_type} is not supported'
            raise Exception(msg)

    # ---------------------------------------------------------------------------

    @property
    def pressure_boundary_conditions(self):
        # Extract all BC names (bc[{patch_name}__{cycle_name}] = value)
        bc = {}
        patch_names = []

        # Handle patch names
        main_name = self._run.Domain.GeomName
        all_names = self._run.Geom[main_name].Patches
        patch_names.extend(all_names)

        # Extract cycle names for each patch
        for p_name in patch_names:
            cycle_name = self._run.Patch[p_name].BCPressure.Cycle
            cycle_names = self._run.Cycle[cycle_name].Names
            for c_name in cycle_names:
                key = f'{p_name}__{c_name}'
                bc[key] = self._run.Patch[p_name].BCPressure[c_name].Value

        return bc

    # ---------------------------------------------------------------------------

    @property
    def pressure(self):
        file_name = get_absolute_path(f'{self._name}.out.press.{self._ts}.pfb')
        return self._pfb_to_array(file_name)

    # ---------------------------------------------------------------------------
    # Saturations
    # ---------------------------------------------------------------------------

    @property
    def saturation(self):
        file_name = get_absolute_path(f'{self._name}.out.satur.{self._ts}.pfb')
        return self._pfb_to_array(file_name)

    # ---------------------------------------------------------------------------
    # Specific storage
    # ---------------------------------------------------------------------------

    @property
    def specific_storage(self):
        return self._pfb_to_array(f'{self._name}.out.specific_storage.pfb')

    # ---------------------------------------------------------------------------
    # Evapotranspiration
    # ---------------------------------------------------------------------------

    @property
    def et(self):
        if self._run.Solver.PrintCLM:
            # Read ET from CLM output
            return self.clm_output('qflx_evap_tot')
        else:
            # Assert that one and only one of Solver.EvapTransFile or Solver.EvapTransFileTransient is set
            assert self._run.Solver.EvapTransFile != self._run.Solver.EvapTransFileTransient, \
                'Only one of Solver.EvapTrans.FileName, Solver.EvapTransFileTransient can be set in order to ' \
                'calculate evapotranspiration'

            if self._run.Solver.EvapTransFile:
                # Read steady-state flux file
                et_data = self._pfb_to_array(self._run.Solver.EvapTrans.FileName)
            else:
                # Read current timestep from series of flux PFB files
                et_data = self._pfb_to_array(f'{self._run.Solver.EvapTrans.FileName}.{self._ts}.pfb')

        return calculate_evapotranspiration(et_data, self.dx, self.dy, self.dz)

    # ---------------------------------------------------------------------------
    # Overland Flow
    # ---------------------------------------------------------------------------

    def overland_flow(self, flow_method='OverlandKinematic', epsilon=1e-5):
        return calculate_overland_flow(self.pressure, self.slope_x, self.slope_y, self.mannings,
                                       self.dx, self.dy, flow_method=flow_method, epsilon=epsilon, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Overland Flow Grid
    # ---------------------------------------------------------------------------

    def overland_flow_grid(self, flow_method='OverlandKinematic', epsilon=1e-5):
        return calculate_overland_flow_grid(self.pressure, self.slope_x, self.slope_y, self.mannings,
                                            self.dx, self.dy, flow_method=flow_method, epsilon=epsilon, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Subsurface Storage
    # ---------------------------------------------------------------------------

    @property
    def subsurface_storage(self):
        return calculate_subsurface_storage(self.computed_porosity, self.pressure, self.saturation,
                                            self.specific_storage, self.dx, self.dy, self.dz, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Surface Storage
    # ---------------------------------------------------------------------------

    @property
    def surface_storage(self):
        return calculate_surface_storage(self.pressure, self.dx, self.dy, mask=self.mask)

    # ---------------------------------------------------------------------------
    # Water Table Depth
    # ---------------------------------------------------------------------------

    @property
    def wtd(self):
        return calculate_water_table_depth(self.pressure, self.saturation, self.dz)

    # ---------------------------------------------------------------------------
    # CLM
    # ---------------------------------------------------------------------------

    def _clm_output_filepath(self, directory, prefix, ext):
        file_name = f'{prefix}.{self._ts}.{ext}.{self._process_id}'
        base_path = f'{self._run.Solver.CLM.CLMFileDir}/{directory}'
        return get_absolute_path(f'{base_path}/{file_name}')

    def _clm_output_bin(self, field, dtype):
        fp = self._clm_output_filepath(field, field, 'bin')
        return np.fromfile(fp, dtype=dtype, count=-1, sep='', offset=0)

    def clm_output(self, field, layer=-1):
        assert self._run.Solver.PrintCLM, 'CLM output must be enabled'
        assert field in self.clm_output_variables, f'Unrecognized variable {field}'

        if self._run.Solver.CLM.SingleFile:
            file_name = f'{self._name}.out.clm_output.{self._ts}.C.pfb'
            arr = self._pfb_to_array(f'{file_name}')

            nz = arr.shape[0]
            nz_expected = len(self.clm_output_variables) + self._run.Solver.CLM.RootZoneNZ - 1
            assert nz == nz_expected, f'Unexpected shape of CLM output, expected {nz_expected}, got {nz}'

            i = self.clm_output_variables.index(field)
            if field == 't_soil':
                if layer < 0:
                    i = layer
                else:
                    i += layer

            arr = arr[i, :, :]
        else:
            file_name = f'{self._name}.out.{field}.{self._ts}.pfb'
            arr = self._pfb_to_array(f'{file_name}')

            if field == 't_soil':
                nz = arr.shape[0]
                assert nz == self._run.Solver.CLM.RootZoneNZ, f'Unexpected shape of CLM output, expected ' \
                                                              f'{self._run.Solver.CLM.RootZoneNZ}, got {nz}'
                arr = arr[layer, :, :]

        if arr.ndim == 3:
            arr = np.squeeze(arr, axis=0)
        return arr

    @property
    def clm_output_variables(self):
        return ('eflx_lh_tot',
                'eflx_lwrad_out',
                'eflx_sh_tot',
                'eflx_soil_grnd',
                'qflx_evap_tot',
                'qflx_evap_grnd',
                'qflx_evap_soi',
                'qflx_evap_veg',
                'qflx_tran_veg',
                'qflx_infl',
                'swe_out',
                't_grnd',
                'qflx_qirr',
                't_soil')

    @property
    def clm_output_diagnostics(self):
        return self._clm_output_filepath('diag_out', 'diagnostics', 'dat')

    @property
    def clm_output_eflx_lh_tot(self):
        return self._clm_output_bin('eflx_lh_tot', float)

    @property
    def clm_output_eflx_lwrad_out(self):
        return self._clm_output_bin('eflx_lwrad_out', float)

    @property
    def clm_output_eflx_sh_tot(self):
        return self._clm_output_bin('eflx_sh_tot', float)

    @property
    def clm_output_eflx_soil_grnd(self):
        return self._clm_output_bin('eflx_soil_grnd', float)

    @property
    def clm_output_qflx_evap_grnd(self):
        return self._clm_output_bin('qflx_evap_grnd', float)

    @property
    def clm_output_qflx_evap_soi(self):
        return self._clm_output_bin('qflx_evap_soi', float)

    @property
    def clm_output_qflx_evap_tot(self):
        return self._clm_output_bin('qflx_evap_tot', float)

    @property
    def clm_output_qflx_evap_veg(self):
        return self._clm_output_bin('qflx_evap_veg', float)

    @property
    def clm_output_qflx_infl(self):
        return self._clm_output_bin('qflx_infl', float)

    @property
    def clm_output_qflx_top_soil(self):
        return self._clm_output_bin('qflx_top_soil', float)

    @property
    def clm_output_qflx_tran_veg(self):
        return self._clm_output_bin('qflx_tran_veg', float)

    @property
    def clm_output_swe_out(self):
        return self._clm_output_bin('swe_out', float)

    @property
    def clm_output_t_grnd(self):
        return self._clm_output_bin('t_grnd', float)

    def clm_forcing(self, name):
        time_slice = self._run.Solver.CLM.MetFileNT
        prefix = self._run.Solver.CLM.MetFileName
        directory = self._run.Solver.CLM.MetFilePath
        file_index = int(self._forcing_time / time_slice)
        t0 = f'{file_index * time_slice + 1:0>6}'
        t1 = f'{(file_index + 1) * time_slice:0>6}'
        file_name = get_absolute_path(
            f'{directory}/{prefix}.{name}.{t0}_to_{t1}.pfb')

        return self._pfb_to_array(file_name)[self._forcing_time % time_slice]

    @property
    def clm_forcing_dswr(self):
        """Downward Visible or Short-Wave radiation [W/m2]"""
        return self.clm_forcing('DSWR')

    @property
    def clm_forcing_dlwr(self):
        """Downward Infa-Red or Long-Wave radiation [W/m2]"""
        return self.clm_forcing('DLWR')

    @property
    def clm_forcing_apcp(self):
        """Precipitation rate [mm/s]"""
        return self.clm_forcing('APCP')

    @property
    def clm_forcing_temp(self):
        """Air temperature [K]"""
        return self.clm_forcing('Temp')

    @property
    def clm_forcing_ugrd(self):
        """West-to-East or U-component of wind [m/s]"""
        return self.clm_forcing('UGRD')

    @property
    def clm_forcing_vgrd(self):
        """South-to-North or V-component of wind [m/s]"""
        return self.clm_forcing('VGRD')

    @property
    def clm_forcing_press(self):
        """Atmospheric Pressure [pa]"""
        return self.clm_forcing('Press')

    @property
    def clm_forcing_spfh(self):
        """Water-vapor specific humidity [kg/kg]"""
        return self.clm_forcing('SPFH')

    def _clm_map(self, root):
        if root.Type == 'Constant':
            return root.Value

        if root.Type == 'Linear':
            return (root.Min, root.Max)

        if root.Type == 'PFBFile':
            return self._pfb_to_array(root.FileName)

        return None

    def clm_map_land_fraction(self, name):
        root = self._run.Solver.CLM.Vegetation.Map.LandFrac[name]
        return self._clm_map(root)

    @property
    def clm_map_latitude(self):
        root = self._run.Solver.CLM.Vegetation.Map.Latitude
        return self._clm_map(root)

    @property
    def clm_map_longitude(self):
        root = self._run.Solver.CLM.Vegetation.Map.Longitude
        return self._clm_map(root)

    @property
    def clm_map_sand(self):
        root = self._run.Solver.CLM.Vegetation.Map.Sand
        return self._clm_map(root)

    @property
    def clm_map_clay(self):
        root = self._run.Solver.CLM.Vegetation.Map.Clay
        return self._clm_map(root)

    @property
    def clm_map_color(self):
        root = self._run.Solver.CLM.Vegetation.Map.Color
        return self._clm_map(root)
