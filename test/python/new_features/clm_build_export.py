from pathlib import Path

import numpy as np

from parflow import Run
from parflow.tools.builders import CLMImporter, DomainBuilder
from parflow.tools.export import CLMExporter
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb, read_clm

clmin_file = "../../input/drv_clmin.dat.old"
vegm_file = "../../tcl/clm/drv_vegm.dat"
vegp_file = "../../tcl/clm/drv_vegp.dat"


# ---------------------------------------------------------
# Testing clm data key setting with DomainBuilder
# ---------------------------------------------------------


def test_clmin_key_setting(clm):
    clm_key_settings = {
        "StartDate": "2020-01-01",
        "StartTime": "00-00-00",
        "EndDate": "2020-12-31",
        "EndTime": "23-59-59",
        "metf1d": "narr_1hr.sc3.txt",
        "outf1d": "washita.output.txt",
        "poutf1d": "test.out.dat",
        "rstf": "washita.rst.",
    }
    DomainBuilder(clm).clm_input(**clm_key_settings)

    clm_timing = clm.Solver.CLM.Input.Timing
    clm_files = clm.Solver.CLM.Input.File
    assert clm_timing.StartYear == "2020"
    assert clm_timing.EndHour == "23"
    assert clm_timing.EndMinute == "59"
    assert clm_files.MetInput == clm_key_settings["metf1d"]
    assert clm_files.Output == clm_key_settings["outf1d"]
    assert clm_files.ParamOutput == clm_key_settings["poutf1d"]
    assert clm_files.ActiveRestart == clm_key_settings["rstf"]


# ---------------------------------------------------------
# Testing clmin file setting with CLMImporter
# ---------------------------------------------------------


def test_clmin_file_setting(clm):
    input = clm.Solver.CLM.Input

    # Prove that some of these got changed
    input.Domain.MaxTiles = 3
    input.InitCond.SnowCover = 2
    input.Forcing.WindObsHeight = 5

    # Will generate warnings for keys that CLM doesn't need
    CLMImporter(clm).input_file(clmin_file)
    verify_clmin_data(clm)


# ---------------------------------------------------------
# Testing vegm file setting with CLMImporter
# ---------------------------------------------------------


def test_vegm_file_setting(clm):
    veg_params = clm.Solver.CLM.Vegetation.Parameters

    try:
        # This should produce an exception since we haven't set the
        # land items yet.
        CLMImporter(clm).map_file(vegm_file)
    except Exception as e:
        assert str(e) == "Land cover items are not set"

    # Now set the land cover items and try again
    veg_params.LandNames = veg_params.LandNames

    CLMImporter(clm).map_file(vegm_file)
    verify_vegm_data(clm)


# ---------------------------------------------------------
# Testing vegp file setting with CLMImporter
# ---------------------------------------------------------


def test_vegp_file_setting(clm):
    CLMImporter(clm).parameters_file(vegp_file)
    verify_vegp_data(clm)


# ---------------------------------------------------------
# Testing exporters
# ---------------------------------------------------------


def test_clmin_round_trip(clm):
    path = Path(get_absolute_path("drv_clmin.dat"))
    if path.exists():
        path.unlink()

    CLMExporter(clm).write_input()

    new_clm = Run("clm", __file__)
    CLMImporter(new_clm).input_file(path)
    verify_clmin_data(new_clm)


def test_vegm_round_trip(clm):
    path = Path(get_absolute_path("drv_vegm.dat"))
    if path.exists():
        path.unlink()

    CLMExporter(clm).write_map()

    new_clm = Run("clm", __file__)
    veg_params = new_clm.Solver.CLM.Vegetation.Parameters

    # Set the land cover items
    veg_params.LandNames = veg_params.LandNames

    CLMImporter(new_clm).map_file(path)
    verify_vegm_data(new_clm)


def test_vegp_round_trip(clm):
    path = Path(get_absolute_path("drv_vegp.dat"))
    if path.exists():
        path.unlink()

    CLMExporter(clm).write_parameters()

    new_clm = Run("clm", __file__)
    veg_params = new_clm.Solver.CLM.Vegetation.Parameters

    # Set the land cover items
    veg_params.LandNames = veg_params.LandNames

    CLMImporter(new_clm).parameters_file(path)
    verify_vegp_data(new_clm)


def test_fewer_land_names(clm):
    vegm_file = Path(get_absolute_path("drv_vegm.dat"))
    vegp_file = Path(get_absolute_path("drv_vegp.dat"))
    for path in (vegm_file, vegp_file):
        if path.exists():
            path.unlink()

    CLMExporter(clm).write_map().write_parameters()

    new_clm = Run("clm", __file__)
    veg_params = new_clm.Solver.CLM.Vegetation.Parameters
    veg_map = new_clm.Solver.CLM.Vegetation.Map

    # Set only two land cover items
    veg_params.LandNames = "forest_en forest_eb"

    CLMImporter(new_clm).map_file(vegm_file).parameters_file(vegp_file)

    assert veg_params.LandNames == ["forest_en", "forest_eb"]
    assert "water" not in veg_params.keys()
    assert "water" not in veg_map.keys()

    veg_map.LandFrac.forest_en.Value = 0.6
    veg_map.LandFrac.forest_eb.Value = 0.4
    veg_map.LandFrac.forest_en.Type = "Constant"
    veg_map.LandFrac.forest_eb.Type = "Constant"

    new_clm.ComputationalGrid.NX = 5
    new_clm.ComputationalGrid.NY = 5

    CLMExporter(new_clm).write_map().write_parameters()

    vegm_data = read_clm(vegm_file, type="vegm")
    vegp_data = read_clm(vegp_file, type="vegp")

    # These should be 0.6 and 0.4
    shape = vegm_data.shape[:2]
    assert np.array_equal(vegm_data[:, :, 5], np.full(shape, 0.6))
    assert np.array_equal(vegm_data[:, :, 6], np.full(shape, 0.4))

    # All of the vegm data after the 2 land types should be zero
    last_land_ind = 6
    zeros = np.zeros(shape)
    for i in range(last_land_ind + 1, vegm_data.shape[2]):
        assert np.array_equal(vegm_data[:, :, i], zeros)

    # All of the vegp data after the second land type should be equal
    # to the default
    default_vegp = CLMExporter(new_clm)._default_vegp_values
    for key, val in vegp_data.items():
        for i in range(2, len(val)):
            assert val[i] == default_vegp[key]

    # Make sure at least one of the other values is correct
    key = "z0m"
    ind = 1
    val = 2.2
    assert vegp_data[key][ind] != default_vegp[key]
    assert vegp_data[key][ind] == val


def verify_clmin_data(clm):
    input = clm.Solver.CLM.Input
    timing = input.Timing

    assert timing.StartYear == 1998
    assert timing.StartMonth == 9
    assert timing.StartDay == 1
    assert timing.EndYear == 2001
    assert input.Domain.MaxTiles == 1
    assert input.InitCond.SnowCover == 0
    assert input.Forcing.WindObsHeight == 10


def verify_vegm_data(clm):
    veg_map = clm.Solver.CLM.Vegetation.Map

    shape = (5, 5)
    properties = ["Latitude", "Longitude", "Sand", "Clay", "Color", "LandFrac"]

    # Check to see that all matrices got set
    for key in veg_map.keys():
        item = veg_map[key]

        if key == "LandFrac":
            for land_frac_name in item.keys():
                land_frac = item[land_frac_name]
                assert land_frac.Type in ("PFBFile", "Constant")
                if land_frac.Type == "PFBFile":
                    array = read_pfb(land_frac.FileName).squeeze()
                    assert array.shape == shape
        else:
            assert item.Type in ("PFBFile", "Constant")
            if item.Type == "PFBFile":
                array = read_pfb(item.FileName).squeeze()
                assert array.shape == shape

    # Check a few individual ones
    const_values = {
        "Latitude": 34.75,
        "Longitude": -98.138,
        "Sand": 0.16,
        "Clay": 0.265,
        "Color": 2,
    }
    for name, val in const_values.items():
        assert veg_map[name].Type == "Constant"
        assert veg_map[name].Value == val

    const_land_fracs = {
        "forest_en": 0.0,
        "grasslands": 1.0,
    }
    for name, val in const_land_fracs.items():
        assert veg_map.LandFrac[name].Type == "Constant"
        assert veg_map.LandFrac[name].Value == val


def verify_vegp_data(clm):
    veg_params = clm.Solver.CLM.Vegetation.Parameters

    # Check that a few of these were set
    check_list = {
        "forest_en": {"WaterType": 1, "MaxLAI": 6.0, "MinLAI": 5.0, "DispHeight": 11.0},
        "water": {
            "WaterType": 3,
            "StemAI": 2.0,
            "RoughLength": 0.002,
            "StemTransNir": -99.0,
        },
    }
    for name, item in check_list.items():
        for key, val in item.items():
            assert veg_params[name][key] == val


if __name__ == "__main__":
    clm = Run("clm", __file__)

    clm.ComputationalGrid.NX = 5
    clm.ComputationalGrid.NY = 5

    test_clmin_key_setting(clm)
    test_clmin_file_setting(clm)
    test_vegm_file_setting(clm)
    test_vegp_file_setting(clm)

    # Test export then loading the file back in
    test_clmin_round_trip(clm)
    test_vegm_round_trip(clm)
    test_vegp_round_trip(clm)

    # Make sure things are working for fewer land names
    test_fewer_land_names(clm)
