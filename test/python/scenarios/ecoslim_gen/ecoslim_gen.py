# ecoslim_gen.py
# example script to guide the outputs necessary for EcoSLIM

from parflow import Run
from parflow.builders import EcoSlimBuilder

ecoslim_gen = Run("ecoslim_gen", __file__)

# EcoSLIM needs the following outputs from ParFlow (** = optional):
# - velocity (x)
# - velocity (y)
# - velocity (z)
ecoslim_gen.Solver.PrintVelocities = True

# - saturation
ecoslim_gen.Solver.PrintSaturation = True

# - porosity
ecoslim_gen.Solver.PrintSubsurf = True

# - DEM .pfb file **

# - indicator .pfb file **

# - evapotranspiration files (out.evaptrans.filenumber.pfb) **
ecoslim_gen.Solver.PrintEvapTrans = True

# - CLM files (.C.pfb) **
ecoslim_gen.Solver.PrintCLM = True

EcoSlimBuilder.run()
EcoSlimBuilder.key_add()


