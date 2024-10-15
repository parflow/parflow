import json
from pathlib import Path

import numpy as np

from parflow import Run
from parflow.tools.builders import VegParamBuilder
from parflow.tools.fs import get_absolute_path
from parflow.tools.export import CLMExporter
from parflow.tools.io import read_clm


def verify_clmin_data(data):
    assert data

    path = get_absolute_path("$PF_SRC/test/correct_output/clmin_data.ref.json")
    with open(path, "r") as rf:
        ref_data = json.load(rf)

    assert data == ref_data


def verify_vegm_data(data):
    # Test the shape and a few data points
    assert data.shape == (5, 5, 23)
    assert data[1, 1, 14] == 1
    assert data[3, 4, 1] == -98.138
    assert data[1, 2, 3] == 0.265

    # Now test the whole array
    path = get_absolute_path("$PF_SRC/test/correct_output/drv_vegm.ref.npy")
    ref_data = np.load(path)
    assert np.array_equal(data, ref_data)


def verify_vegp_data(data):
    assert data

    path = get_absolute_path("$PF_SRC/test/correct_output/vegp_data.ref.json")
    with open(path, "r") as rf:
        ref_data = json.load(rf)

    assert data == ref_data


def verify_short_vegp_data(exporter, data):
    assert data

    # Make sure all data past the cutoff was set to the default
    cutoff = 4
    defaults = exporter._default_vegp_values
    for key, vals in data.items():
        assert all(x == defaults[key] for x in vals[cutoff:])


clm = Run("clm", __file__)

# ---------------------------------------------------------
# Testing clm data readers
# ---------------------------------------------------------

# Reading drv_clmin.dat into dictionary
# using old file that has more variables than CLM currently needs
clmin_data = read_clm("$PF_SRC/test/input/drv_clmin.dat.old", type="clmin")
verify_clmin_data(clmin_data)
print(f"clmin_data = {clmin_data}")

# Reading drv_vegm.dat into 3D array
vegm_data = read_clm("$PF_SRC/test/tcl/clm/drv_vegm.dat", type="vegm")
verify_vegm_data(vegm_data)
print(f"vegm_data.shape = {vegm_data.shape}")

# Reading drv_vegp.dat into dictionary
vegp_data = read_clm("$PF_SRC/test/tcl/clm/drv_vegp.dat", type="vegp")
verify_vegp_data(vegp_data)
print(f"vegp_data = {vegp_data}")

# ---------------------------------------------------------
# Testing clm data writers
# ---------------------------------------------------------

# Remove any leftover driver files...
paths = [
    "drv_clmin.dat",
    "drv_vegm.dat",
    "drv_vegp.dat",
]
for path in paths:
    path = Path(get_absolute_path(f"$PF_SRC/test/python/new_features/{path}"))
    if path.exists():
        path.unlink()

CLMExporter(clm).write_input().write_map(vegm_data).write_parameters(vegp_data)

# Re-read the exported files back in to ensure they are the same
new_clmin_data = read_clm("drv_clmin.dat", type="clmin")
new_vegm_data = read_clm("drv_vegm.dat", type="vegm")
new_vegp_data = read_clm("drv_vegp.dat", type="vegp")

# FIXME: because some CLM keys are not yet registered in the pf-keys yaml
# files, the new clm data does not match.
# assert new_clmin_data == clmin_data
assert np.array_equal(new_vegm_data, vegm_data)
assert new_vegp_data == vegp_data

clm.Solver.CLM.Vegetation.Parameters.LandNames = (
    "forest_en forest_eb forest_dn forest_db"
)

# Make the data list short and ensure default values get filled
for val in vegp_data.values():
    while len(val) > 4:
        val.pop()

CLMExporter(clm).write_parameters(vegp_data)
new_vegp_data = read_clm("drv_vegp.dat", type="vegp")
verify_short_vegp_data(CLMExporter(clm), new_vegp_data)

VegParamBuilder(clm).load_default_properties().print()

clm.Solver.CLM.Vegetation.Map.Clay.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.Clay.Value = 0.264
