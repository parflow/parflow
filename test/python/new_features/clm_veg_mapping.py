from pathlib import Path
import sys

import numpy as np

from parflow import Run
from parflow.tools.builders import VegParamBuilder
from parflow.tools.export import CLMExporter
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_clm, write_pfb

clm = Run("clm", __file__)

# ---------------------------------------------------------
# Computational grid keys
# ---------------------------------------------------------

clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 1000.0
clm.ComputationalGrid.DY = 1000.0
clm.ComputationalGrid.DZ = 0.5

clm.ComputationalGrid.NX = 5
clm.ComputationalGrid.NY = 5
clm.ComputationalGrid.NZ = 10

# ---------------------------------------------------------
# Testing clm data key setting
# ---------------------------------------------------------

# TODO: add to DomainBuilder
clm.Solver.CLM.Vegetation.Parameters.LandNames = (
    "forest_en forest_eb forest_dn forest_db"
)

clm.Solver.CLM.Vegetation.Map.Latitude.Type = "Linear"
clm.Solver.CLM.Vegetation.Map.Latitude.Min = 34.750
clm.Solver.CLM.Vegetation.Map.Latitude.Max = 35.750

# TODO: add parsing of PFBFiles
# clm.Solver.CLM.Vegetation.Map.Longitude.Type = 'PFBFile'
# clm.Solver.CLM.Vegetation.Map.Longitude.FileName = 'longitude_mapping.pfb'

clm.Solver.CLM.Vegetation.Map.Longitude.Type = "Linear"
clm.Solver.CLM.Vegetation.Map.Longitude.Min = -99.00
clm.Solver.CLM.Vegetation.Map.Longitude.Max = -98.00

clm.Solver.CLM.Vegetation.Map.Sand.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.Sand.Value = 0.16

clm.Solver.CLM.Vegetation.Map.Clay.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.Clay.Value = 0.265

clm.Solver.CLM.Vegetation.Map.Color.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.Color.Value = 2

# ---------------------------------------------------------
# Setting land use fractions
# ---------------------------------------------------------

clm.Solver.CLM.Vegetation.Map.LandFrac.forest_en.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.LandFrac.forest_en.Value = 0.0

forest_eb_mat = np.zeros((clm.ComputationalGrid.NX, clm.ComputationalGrid.NY))
forest_eb_mat[1, :] = 1.0
file_name = "forest_eb_mat.pfb"
write_pfb(get_absolute_path(file_name), forest_eb_mat)

clm.Solver.CLM.Vegetation.Map.LandFrac.forest_eb.Type = "PFBFile"
clm.Solver.CLM.Vegetation.Map.LandFrac.forest_eb.FileName = file_name

clm.Solver.CLM.Vegetation.Map.LandFrac.forest_dn.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.LandFrac.forest_dn.Value = 0.5

clm.Solver.CLM.Vegetation.Map.LandFrac.forest_db.Type = "Constant"
clm.Solver.CLM.Vegetation.Map.LandFrac.forest_db.Value = 0.0

# ---------------------------------------------------------
# Testing clm data reader for veg mapping
# ---------------------------------------------------------

# Reading drv_vegm.dat into 3D array
vegm_data = read_clm("../../tcl/clm/drv_vegm.dat", type="vegm")
if not vegm_data[1, 1, 14] == 1:
    sys.exit(1)

# ---------------------------------------------------------
# Testing clm data writers
# ---------------------------------------------------------

path = get_absolute_path("drv_vegm.dat")
if Path(path).exists():
    Path(path).unlink()

CLMExporter(clm).write_map()
