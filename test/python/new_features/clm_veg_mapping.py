from parflow import Run
from parflow.tools.builders import VegParamBuilder
from parflow.tools.export import CLMExporter
from parflow.tools.io import read_clm
import sys
import numpy as np

clm = Run("clm", __file__)

#---------------------------------------------------------
# Computational grid keys
#---------------------------------------------------------

clm.ComputationalGrid.Lower.X = 0.0
clm.ComputationalGrid.Lower.Y = 0.0
clm.ComputationalGrid.Lower.Z = 0.0

clm.ComputationalGrid.DX = 1000.
clm.ComputationalGrid.DY = 1000.
clm.ComputationalGrid.DZ = 0.5

clm.ComputationalGrid.NX = 5
clm.ComputationalGrid.NY = 5
clm.ComputationalGrid.NZ = 10

#---------------------------------------------------------
# Testing clm data key setting
#---------------------------------------------------------

# TODO: add to DomainBuilder
clm.Solver.CLM.Vegetation.Parameters.LandNames = 'forest_en forest_eb forest_dn forest_db'

clm.Solver.CLM.Vegetation.Map.Latitude.Type = 'Linear'
clm.Solver.CLM.Vegetation.Map.Latitude.Min = 34.750
clm.Solver.CLM.Vegetation.Map.Latitude.Max = 35.750

# TODO: add parsing of PFBFiles
# clm.Solver.CLM.Vegetation.Map.Longitude.Type = 'PFBFile'
# clm.Solver.CLM.Vegetation.Map.Longitude.FileName = 'longitude_mapping.pfb'

clm.Solver.CLM.Vegetation.Map.Longitude.Type = 'Linear'
clm.Solver.CLM.Vegetation.Map.Longitude.Min = -99.00
clm.Solver.CLM.Vegetation.Map.Longitude.Max = -98.00

clm.Solver.CLM.Vegetation.Map.Sand.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.Sand.Value = 0.16

clm.Solver.CLM.Vegetation.Map.Clay.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.Clay.Value = 0.265

clm.Solver.CLM.Vegetation.Map.Color.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.Color.Value = 2

#---------------------------------------------------------
# Setting land use fractions
#---------------------------------------------------------

clm.Solver.CLM.Vegetation.Map.forest_en.LandFrac.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.forest_en.LandFrac.Value = 0.0

forest_eb_mat = np.zeros((clm.ComputationalGrid.NX, clm.ComputationalGrid.NY))
forest_eb_mat[1, :] = 1.0

clm.Solver.CLM.Vegetation.Map.forest_eb.LandFrac.Type = 'Matrix'
clm.Solver.CLM.Vegetation.Map.forest_eb.LandFrac.Matrix = forest_eb_mat

clm.Solver.CLM.Vegetation.Map.forest_dn.LandFrac.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.forest_dn.LandFrac.Value = 0.5

clm.Solver.CLM.Vegetation.Map.forest_db.LandFrac.Type = 'Constant'
clm.Solver.CLM.Vegetation.Map.forest_db.LandFrac.Value = 0.0

#---------------------------------------------------------
# Testing clm data reader for veg mapping
#---------------------------------------------------------

# Reading drv_vegm.dat into 3D array
vegm_data = read_clm('../../tcl/clm/drv_vegm.dat', type='vegm')
if not vegm_data[1, 1, 14] == 1:
    sys.exit(1)

# ---------------------------------------------------------
# Testing clm data writers
# ---------------------------------------------------------

CLMExporter(clm) \
    .export_drv_vegm()




