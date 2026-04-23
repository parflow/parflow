import struct

import numpy as np


class Header:
    """ Header class generated during pfb files reading """
    def __init__(self, pfb_filename: str):
        with open(pfb_filename, "rb") as src:
            src.seek(0)
            self.x = struct.unpack(">d", src.read(8))[0]
            self.y = struct.unpack(">d", src.read(8))[0]
            self.z = struct.unpack(">d", src.read(8))[0]
            self.nx = struct.unpack(">i", src.read(4))[0]
            self.ny = struct.unpack(">i", src.read(4))[0]
            self.nz = struct.unpack(">i", src.read(4))[0]
            self.dx = struct.unpack(">d", src.read(8))[0]
            self.dy = struct.unpack(">d", src.read(8))[0]
            self.dz = struct.unpack(">d", src.read(8))[0]
            self.n_subgrids = struct.unpack(">i", src.read(4))[0]

    @property
    def west(self) -> float:
        return self.x + self.dx / 2

    @property
    def east(self) -> float:
        return self.west + (self.nx - 1) * self.dx

    @property
    def south(self) -> float:
        return self.y + self.dy / 2

    @property
    def north(self) -> float:
        return self.south + (self.ny - 1) * self.dy

    def get_centered_x(self) -> np.ndarray:
        return self.west + np.arange(self.nx) * self.dx

    def get_centered_y(self) -> np.ndarray:
        return self.south + np.arange(self.ny) * self.dy

    def get_z(self):
        return self.z + np.arange(self.nz) * self.dz

    def get_shape(self, z_first=True):
        if z_first:
            return self.nz, self.ny, self.nx
        return self.nx, self.ny, self.nz
