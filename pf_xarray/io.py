import itertools
from numba import jit
import numpy as np
import struct


def read_pfb(file, mode='full'):
    pfb = ParflowBinaryReader(file)
    data = pfb.read_all_subgrids(mode=mode)
    pfb.close()
    return data


def read_stack_of_pfbs(file_seq):
    pfb_init = ParflowBinaryReader(file_seq[0])
    base_header = pfb_init.header
    base_sg_offsets = pfb_init.subgrid_offsets
    base_sg_locations = pfb_init.subgrid_locations
    base_sg_indices = pfb_init.subgrid_start_indices
    base_sg_shapes = pfb_init.subgrid_shapes
    pfb_init.close()
    stack_size= (len(file_seq), base_header['nx'], base_header['ny'], base_header['nz'])
    pfb_stack = np.empty(stack_size, dtype=np.float64)
    for i, f in enumerate(file_seq):
        pfb = ParflowBinaryReader(f, precompute_subgrid_info=False, header=base_header)
        pfb.subgrid_offsets = base_sg_offsets
        pfb.subgrid_locations = base_sg_locations
        pfb.subgrid_start_indices = base_sg_indices
        pfb.subgrid_shapes = base_sg_shapes
        pfb_stack[i, :, : ,:] = pfb.read_all_subgrids(mode='full')
        pfb.close()
    return pfb_stack


class ParflowBinaryReader:

    def __init__(self, file, precompute_subgrid_info=True, p=None, q=None, r=None, header=None):
        self.filename = file
        self.f = open(self.filename, 'rb')
        if not header:
            self.header = self.read_header()
        else:
            self.header = header
            p = self.header.get('p', p)
            q = self.header.get('q', q)
            r = self.header.get('r', r)

        # If p, q, and r aren't given we can precompute them
        if not np.all([p, q, r]):
            eps = 1 - 1e-6
            first_sg_head = self.read_subgrid_header()
            self.header['p'] = int((self.header['nx'] / first_sg_head['nx']) + eps)
            self.header['q'] = int((self.header['ny'] / first_sg_head['ny']) + eps)
            self.header['r'] = int((self.header['nz'] / first_sg_head['nz']) + eps)

        if precompute_subgrid_info:
            self.compute_subgrid_info()

    def close(self):
        self.f.close()

    def compute_subgrid_info(self):
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

    def _compute_chunks(self):
        p, q, r = self.header['p'], self.header['q'], self.header['r'],
        x_chunks = tuple(self.subgrid_shapes[:,0][0:p].flatten())
        y_chunks = tuple(self.subgrid_shapes[:,1][0:p*q:p].flatten())
        z_chunks = tuple(self.subgrid_shapes[:,2][0:p*q*r:p*q].flatten())
        return {'x': x_chunks, 'y': y_chunks, 'z': z_chunks}

    def _compute_coords(self):
        coords = {'x': [], 'y': [], 'z': []}
        for c in ['x', 'y', 'z']:
            chunk_start = 0
            for chunk in self.chunks[c]:
                coords[c].append(np.arange(chunk_start, chunk_start + chunk))
                chunk_start += chunk
        return coords

    def read_header(self):
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

    def read_subgrid_header(self, skip_bytes=64):
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

    def read_subarray(self, start_x, start_y, start_z=0, nx=1, ny=1, nz=None):
        """
        mirroring parflowio loadclipofdata
        determine which subgrid start is in
        determine which subgrid end is in
        determine padding from the subgrids

        As an example of what needs to happen here is
        the following image:
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
        """
        def _get_final_clip(start, end, coords):
            """ Helper to clean up code at the end of this """
            x0 = np.flatnonzero(start == coords)
            x0 = 0 if not x0 else x0[0]
            x1 = np.flatnonzero(end == coords)
            x1 = -1 if not x1 else x1[0]
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
        return bounding_data[clip_x, clip_y, clip_z]


    def loc_subgrid(self, pp, qq, rr):
        p, q, r = self.header['p'], self.header['q'], self.header['r']
        subgrid_idx = pp + (p * qq) + (q * rr)
        return self.iloc_subgrid(subgrid_idx)

    def iloc_subgrid(self, idx):
        offset = self.subgrid_offsets[idx]
        shape = self.subgrid_shapes[idx]
        return self._backend_iloc_subgrid(offset, shape)

    def _backend_iloc_subgrid(self, offset, shape):
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

    def read_all_subgrids(self, mode='full'):
        if mode not in ['flat', 'tiled', 'full']:
            raise Exception('mode must be one of flat, tiled, or full')
        if mode in ['flat', 'tiled']:
            all_data = []
            for i in range(self.header['n_subgrids']):
                all_data.append(self.iloc_subgrid(i))
            if mode == 'tiled':
                tiled_shape = tuple(self.header[dim] for dim in ['p', 'q', 'r'])
                all_data = np.array(all_data, dtype=object).reshape(tiled_shape)
        elif mode == 'full':
            full_shape = tuple(self.header[dim] for dim in ['nx', 'ny', 'nz'])
            chunks = self.chunks['x'], self.chunks['y'], self.chunks['z']
            all_data = np.empty(full_shape, dtype=np.float64)
            for i in range(self.header['n_subgrids']):
                nx, ny, nz = self.subgrid_shapes[i]
                ix, iy, iz = self.subgrid_start_indices[i]
                all_data[ix:ix+nx, iy:iy+ny, iz:iz+nz] = self.iloc_subgrid(i)
        return all_data

@jit(nopython=True)
def get_maingrid_and_remainder(nx, ny, nz, p, q, r):
    nnx = int(np.ceil(nx / p))
    nny = int(np.ceil(ny / q))
    nnz = int(np.ceil(nz / r))
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
    ix = max(0, pp * (nnx-1) + min(pp, lx))
    iy = max(0, qq * (nny-1) + min(qq, ly))
    iz = max(0, rr * (nnz-1) + min(rr, lz))
    return ix, iy, iz

@jit(nopython=True)
def subgrid_size(
    nnx, nny, nnz,
    pp, qq, rr,
    lx, ly, lz
):
    snx = nnx-1 if pp >= max(lx, 1) else nnx
    sny = nny-1 if qq >= max(ly, 1) else nny
    snz = nnz-1 if rr >= max(lz, 1) else nnz
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
