"""
    Unit test for the pf_xarray class.
    This verifies methods with the pf_xarray tools module of parflow.
    It uses existing test data .pfb files in the parflow repository for testing.
"""

import sys
import os
import unittest
import numpy as np
import xarray as xr
import tempfile

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(rootdir)
from parflow.tools.pf_backend import ParflowBackendEntrypoint
from parflow import ParflowBinaryReader, read_pfb_sequence, read_pfb, write_pfb
from pfb_summary import PFBSummary

EXAMPLE_PFB_FILE_PATH_0 = f"{rootdir}/tools/tests/data/forsyth5.out.press.00000.pfb"
EXAMPLE_PFB_FILE_PATH_1 = f"{rootdir}/tools/tests/data/forsyth5.out.press.00001.pfb"
TEMP_DIRECTORY = "/tmp/parflow_xarray"


class TestPFXArray(unittest.TestCase):
    def test_open_dataset(self):
        """Test reading a pfb file using xarray open_dataset."""

        ds = xr.open_dataset(
            EXAMPLE_PFB_FILE_PATH_0, name="forsyth5.out.press", engine="parflow"
        )

        # Verify the sizes of the dimensions from the loaded xarray
        self.assertEqual(3, len(ds.dims))
        self.assertEqual(21, ds.dims["z"])
        self.assertEqual(46, ds.dims["x"])
        self.assertEqual(46, ds.dims["y"])
        # Verify the checkum of values in the data array is what is expected
        da = ds.to_array()
        self.assertEqual(-444360000, int(np.sum(da)))
        # Verify that the shape of the data set is 1 variable with the expected 3D dimensions
        self.assertEqual((1, 21, 46, 46), da.shape)

    def test_read_pfb_file_list(self):
        """Test reading a list of pfb files."""

        pfb_file_list = [EXAMPLE_PFB_FILE_PATH_0, EXAMPLE_PFB_FILE_PATH_1]
        da = read_pfb_sequence(pfb_file_list)

        # Verify that the shape of the data set is 2 time rows with the expected 3D dimensions
        self.assertEqual((2, 21, 46, 46), da.shape)
        # Verify the sizes of the dimensions from the loaded xarray
        # Verify the checkum of values in the data array is what is expected
        self.assertEqual(-887197545, int(np.sum(da)))

    def test_write_pfb_file(self):
        """Test writing a single pfb file."""

        with ParflowBinaryReader(EXAMPLE_PFB_FILE_PATH_0) as pfb:
            da = pfb.read_all_subgrids()
            header = pfb.header
            header["p"] = 8
            header["q"] = 5
            header["r"] = 1
            header["n_subgrids"] = header["p"] * header["q"] * header["r"]

            with tempfile.TemporaryDirectory() as TEMP_DIRECTORY:
                file_name = f"{TEMP_DIRECTORY}/example0.pfb"
                write_pfb(file_name, da, **header)
                total = 0
                with PFBSummary() as s:
                    header = s.open_pfb(file_name)
                    self.assertEqual(46, header.get("nx"))
                    self.assertEqual(46, header.get("ny"))
                    self.assertEqual(21, header.get("nz"))
                    number_of_subgrids = int(header.get("n_subgrids", 0))
                    self.assertEqual(40, number_of_subgrids)
                    for i in range(0, number_of_subgrids):
                        sg_header = s.read_subgrid_header()
                        data = s.read_subgrid_data(sg_header)
                        total = total + data.sum()
                        if i == 1:
                            self.assertEqual(1, sg_header.get("ix"))
                            self.assertEqual(0, sg_header.get("iy"))
                            self.assertEqual(0, sg_header.get("iz"))
                            self.assertEqual(6, sg_header.get("nx"))
                            self.assertEqual(10, sg_header.get("ny"))
                            self.assertEqual(21, sg_header.get("nz"))
                self.assertEqual(-444360000.0, int(total))

    def test_local(self):
        # Set up the array that I know values for
        da = np.zeros((1, 4, 4))
        da[:, 0, :] += 1
        da[:, :, 0] += 1
        header = {}
        header["nx"] = 4
        header["ny"] = 4
        header["nz"] = 1
        header["p"] = 2
        header["q"] = 2
        header["r"] = 1
        with tempfile.TemporaryDirectory() as TEMP_DIRECTORY:
            file_name = f"{TEMP_DIRECTORY}/local.pfb"
            write_pfb(file_name, da, **header, dist=True)
            with ParflowBinaryReader(file_name) as s:
                # Check the header
                header = s.header
                self.assertEqual(4, header.get("nx"))
                self.assertEqual(4, header.get("ny"))
                self.assertEqual(1, header.get("nz"))

                # Check the subgrid header
                sg_header = s.read_subgrid_header()
                self.assertEqual(2, sg_header.get("nx"))
                self.assertEqual(2, sg_header.get("ny"))
                self.assertEqual(1, sg_header.get("nz"))

                # Check the subgrid data
                data = s.iloc_subgrid(0)
                checksum = da[:, 0:2, 0:2]
                self.assertEqual(data.sum(), checksum.sum())

                # Check all datapoints
                data = s.read_all_subgrids()
                self.assertEqual(0, np.sum(da - data))

    def test_empty_array(self):
        da = np.arange(0, 0, dtype=np.float64)
        da.resize([0, 0, 0])
        header = {}
        header["p"] = 1
        header["q"] = 1
        header["r"] = 1
        with tempfile.TemporaryDirectory() as TEMP_DIRECTORY:
            file_name = f"{TEMP_DIRECTORY}/empty.pfb"
            write_pfb(file_name, da, **header)
            with PFBSummary() as s:
                header = s.open_pfb(file_name)
                self.assertEqual(0, header.get("nx"))
                self.assertEqual(0, header.get("ny"))
                self.assertEqual(0, header.get("nz"))
                self.assertEqual(0, header.get("nz"))
                number_of_subgrids = int(header.get("n_subgrids", 0))
                self.assertEqual(1, number_of_subgrids)


if __name__ == "__main__":
    unittest.main()
