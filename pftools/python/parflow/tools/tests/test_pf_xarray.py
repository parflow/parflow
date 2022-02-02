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
rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(rootdir)
from pf_xarray.pf_backend import ParflowBackendEntrypoint
from pf_xarray.io import ParflowBinaryReader, read_stack_of_pfbs, read_pfb, write_pfb
from pfb_summary import PFBSummary

EXAMPLE_PFB_FILE_PATH_0 = f"{rootdir}/pf_xarray/tests/data/forsyth5.out.press.00000.pfb"
EXAMPLE_PFB_FILE_PATH_1 = f"{rootdir}/pf_xarray/tests/data/forsyth5.out.press.00001.pfb"
TEMP_DIRECTORY = "/tmp/parflow_xarray"


class TestPFXArray(unittest.TestCase):
    def test_open_dataset(self):
        """Test reading a pfb file using xarray open_dataset."""

        ds = xr.open_dataset(
            EXAMPLE_PFB_FILE_PATH_0, name='forsyth5.out.press', engine=ParflowBackendEntrypoint)

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

        pfb_file_list = [
            EXAMPLE_PFB_FILE_PATH_0,
            EXAMPLE_PFB_FILE_PATH_1
        ]
        da = read_stack_of_pfbs(pfb_file_list)

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
            header['p'] = '8'
            header['q'] = '5'
            header['z'] = '1'
            if not os.path.exists(TEMP_DIRECTORY):
                os.makedirs(TEMP_DIRECTORY)
            file_name = f'{TEMP_DIRECTORY}/example0.pfb'
            if os.path.exists(file_name):
                os.remove(file_name)
            write_pfb(file_name, da, header=header)
            if not os.path.exists(file_name):
                self.fail(f"PFB File '{file_name}' was not created.")
            total = 0
            with PFBPeek() as s:
                header = s.open_pfb(file_name)
                self.assertEqual(46, header.get('nx'))
                self.assertEqual(46, header.get('ny'))
                self.assertEqual(21, header.get('nz'))
                number_of_subgrids = int(header.get('n_subgrids', 0))
                self.assertEqual(40, number_of_subgrids)
                for i in range(0, number_of_subgrids):
                    sg_header = s.read_subgrid_header()
                    data = s.read_subgrid_data(sg_header)
                    total = total + data.sum()
                    if i == 1:
                        self.assertEqual(6, sg_header.get('ix'))
                        self.assertEqual(0, sg_header.get('iy'))
                        self.assertEqual(1, sg_header.get('iz'))
                        self.assertEqual(6, sg_header.get('nx'))
                        self.assertEqual(10, sg_header.get('ny'))
                        self.assertEqual(21, sg_header.get('nz'))
            self.assertEqual(-444360000.0, int(total))
            os.remove(file_name)

    def test_local(self):
        da = np.arange(1, 25, dtype=np.float64)
        da.resize([2, 4, 3])
        header = {}
        header['nx'] = '3'
        header['ny'] = '4'
        header['nz'] = '2'
        header['p'] = '2'
        header['q'] = '2'
        header['r'] = '1'
        if not os.path.exists(TEMP_DIRECTORY):
            os.makedirs(TEMP_DIRECTORY)
        file_name = f'{TEMP_DIRECTORY}/local.pfb'
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(file_name + '.dist'):
            os.remove(file_name + '.dist')
        write_pfb(file_name, da, header=header, dist=True)
        if not os.path.exists(file_name):
            self.fail(f"PFB File '{file_name}' was not created.")
        if not os.path.exists(file_name + '.dist'):
            self.fail(f"PFB File '{file_name}.dist' was not created.")
        with PFBPeek() as s:
            header = s.open_pfb(file_name)
            self.assertEqual(3, header.get('nx'))
            self.assertEqual(4, header.get('ny'))
            self.assertEqual(2, header.get('nz'))
            sg_header = s.read_subgrid_header()
            data = s.read_subgrid_data(sg_header)
            checksum = data.sum()
            self.assertEqual(72, checksum)
            self.assertEqual(2, sg_header.get('nx'))
            self.assertEqual(2, sg_header.get('ny'))
            self.assertEqual(2, sg_header.get('nz'))
            sg_header = s.read_subgrid_header()
            data = s.read_subgrid_data(sg_header)
            self.assertEqual(1, sg_header.get('nx'))
            self.assertEqual(2, sg_header.get('ny'))
            self.assertEqual(2, sg_header.get('nz'))
            self.assertEqual(2, sg_header.get('ix'))
            self.assertEqual(0, sg_header.get('iy'))
            self.assertEqual(0, sg_header.get('iz'))
            checksum = data.sum()
            self.assertEqual(42, checksum)
            self.assertEqual(3, data[0][0][0])
        os.remove(file_name + '.dist')
        os.remove(file_name)

    def test_empty_array(self):
        da = np.arange(0, 0, dtype=np.float64)
        da.resize([0, 0, 0])
        header = {}
        header['p'] = '1'
        header['q'] = '1'
        header['r'] = '1'
        if not os.path.exists(TEMP_DIRECTORY):
            os.makedirs(TEMP_DIRECTORY)
        file_name = f'{TEMP_DIRECTORY}/empty.pfb'
        if os.path.exists(file_name):
            os.remove(file_name)
        write_pfb(file_name, da, header=header)
        if not os.path.exists(file_name):
            self.fail(f"PFB File '{file_name}' was not created.")
        with PFBPeek() as s:
            header = s.open_pfb(file_name)
            self.assertEqual(0, header.get('nx'))
            self.assertEqual(0, header.get('ny'))
            self.assertEqual(0, header.get('nz'))
            self.assertEqual(0, header.get('nz'))
            number_of_subgrids = int(header.get('n_subgrids', 0))
            self.assertEqual(1, number_of_subgrids)
        os.remove(file_name)

if __name__ == '__main__':
    unittest.main()
