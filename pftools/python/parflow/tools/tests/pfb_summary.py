"""
    File: pfb_summary.py

    USAGE:
        python pfb_summary.py <pfb_file>

    DESCRIPTION:
        Peeks into a parflow .pfb file and display a summary of the file.
        This prints to stdout a summary of the .pfb file. It prints the file header from the first 64 bytes.
        Then prints the subgrid headers of the first 2 subgrids and the last subgrid.

        The purpose of this utility is to assist with debugging so you can view a summary of a PFB file.
"""

from ctypes import cdll
import sys
import os
import struct
import numpy as np


def pfb_summary(filename):
    s = PFBSummary(file_name=filename)
    s.print_summary()


class PFBSummary:
    def __init__(self, fp=None, file_name=None, header=None):
        self.fp = fp
        self.file_name = file_name
        self.header = header

    def close(self):
        if self.fp is not None:
            try:
                self.fp.close()
                self.fp = None
            except Exception as e:
                pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.fp is not None:
            try:
                self.fp.close()
                self.fp = None
            except Exception as e:
                pass

    def run(self):
        """Read the command line arguments and print summary of the PFB File."""

        if len(sys.argv) < 2:
            print("Error: PFB File name is required")
            sys.exit(-1)
        file_name = sys.argv[1]
        if not os.path.exists(file_name):
            print(f"File '{file_name}' does not exist.")
            sys.exit(-1)
        self.file_name = file_name
        self.print_summary()

    def print_summary(self):
        self.open_pfb(self.file_name)
        print(self.header)

        checksum = 0
        first_cell = 0
        printed_dots = False
        number_of_subgrids = int(self.header.get("n_subgrids", 0))
        for i in range(0, number_of_subgrids):
            subgrid_header = self.read_subgrid_header()
            data = self.read_subgrid_data(subgrid_header)
            checksum = checksum + float(np.sum(data))
            if i == 0:
                first_cell = data[0][0][0]
            if i < 2 or i >= number_of_subgrids - 1:
                print()
                print(f"Subgrid #{i}")
                print(subgrid_header)
                print(data)
            elif not printed_dots:
                print()
                print("... more subgrids ...")
                printed_dots = True
        print()
        print("Checksum of all subgrid values")
        print(checksum)
        print()
        print("Value of first cell")
        print(first_cell)

    def open_pfb(self, file_name):
        """Read the .pfb file_name and print the summary."""

        self.fp = open(file_name, "rb")
        self.file_name = file_name
        self.read_header()
        return self.header

    def read_header(self):
        """Read the pfb file header into self.header."""

        self.fp.seek(0)
        self.header = {}
        self.header["x"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["y"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["z"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["nx"] = struct.unpack(">i", self.fp.read(4))[0]
        self.header["ny"] = struct.unpack(">i", self.fp.read(4))[0]
        self.header["nz"] = struct.unpack(">i", self.fp.read(4))[0]
        self.header["dx"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["dy"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["dz"] = struct.unpack(">d", self.fp.read(8))[0]
        self.header["n_subgrids"] = struct.unpack(">i", self.fp.read(4))[0]

    def read_subgrid_header(self):
        """Read the subgrid header from the file and return the header as a dict."""

        subgrid_header = {}
        subgrid_header["ix"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["iy"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["iz"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["nx"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["ny"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["nz"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["rx"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["ry"] = struct.unpack(">i", self.fp.read(4))[0]
        subgrid_header["rz"] = struct.unpack(">i", self.fp.read(4))[0]
        return subgrid_header

    def read_subgrid_data(self, subgrid_header):
        """Read the data of the subgrid. Returns a numpy array mapped to the subgrid data."""
        ix = subgrid_header["ix"]
        iy = subgrid_header["iy"]
        iz = subgrid_header["iz"]
        nx = subgrid_header["nx"]
        ny = subgrid_header["ny"]
        nz = subgrid_header["nz"]

        offset = self.fp.tell()
        shape = [nz, ny, nx]
        data = np.memmap(
            self.file_name,
            dtype=np.float64,
            mode="r",
            offset=offset,
            shape=tuple(shape),
            order="F",
        ).byteswap()
        offset = offset + nx * ny * nz * 8
        self.fp.seek(offset)
        return data


if __name__ == "__main__":
    main = PFBSummary()
    main.run()
