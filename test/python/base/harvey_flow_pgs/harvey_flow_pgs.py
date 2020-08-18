# this runs the Cape Cod site flow case for the Harvey and Garabedian bacterial 
# injection experiment from Maxwell, et al, 2007.

#
# Import the ParFlow TCL package
#
from parflow import Run
harvey_flow_pgs = Run("harvey_flow_pgs", __file__)


#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
harvey_flow_pgs.FileVersion = 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

harvey_flow_pgs.Process.Topology.P = 1
harvey_flow_pgs.Process.Topology.Q = 1
harvey_flow_pgs.Process.Topology.R = 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
harvey_flow_pgs.ComputationalGrid.Lower.X = 0.0
harvey_flow_pgs.ComputationalGrid.Lower.Y = 0.0
harvey_flow_pgs.ComputationalGrid.Lower.Z = 0.0

harvey_flow_pgs.ComputationalGrid.DX = 0.34
harvey_flow_pgs.ComputationalGrid.DY = 0.34
harvey_flow_pgs.ComputationalGrid.DZ = 0.038

harvey_flow_pgs.ComputationalGrid.NX = 50
harvey_flow_pgs.ComputationalGrid.NY = 30
harvey_flow_pgs.ComputationalGrid.NZ = 100

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
harvey_flow_pgs.GeomInput.Names = 'domain_input upper_aquifer_input lower_aquifer_input'


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
harvey_flow_pgs.GeomInput.domain_input.InputType = 'Box'
harvey_flow_pgs.GeomInput.domain_input.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
harvey_flow_pgs.Geom.domain.Lower.X = 0.0
harvey_flow_pgs.Geom.domain.Lower.Y = 0.0
harvey_flow_pgs.Geom.domain.Lower.Z = 0.0

harvey_flow_pgs.Geom.domain.Upper.X = 17.0
harvey_flow_pgs.Geom.domain.Upper.Y = 10.2
harvey_flow_pgs.Geom.domain.Upper.Z = 3.8

harvey_flow_pgs.Geom.domain.Patches = 'left right front back bottom top'

#-----------------------------------------------------------------------------
# Upper Aquifer Geometry Input
#-----------------------------------------------------------------------------
harvey_flow_pgs.GeomInput.upper_aquifer_input.InputType = 'Box'
harvey_flow_pgs.GeomInput.upper_aquifer_input.GeomName = 'upper_aquifer'

#-----------------------------------------------------------------------------
# Upper Aquifer Geometry
#-----------------------------------------------------------------------------
harvey_flow_pgs.Geom.upper_aquifer.Lower.X = 0.0
harvey_flow_pgs.Geom.upper_aquifer.Lower.Y = 0.0
harvey_flow_pgs.Geom.upper_aquifer.Lower.Z = 1.5
#pfset Geom.upper_aquifer.Lower.Z                        0.0

harvey_flow_pgs.Geom.upper_aquifer.Upper.X = 17.0
harvey_flow_pgs.Geom.upper_aquifer.Upper.Y = 10.2
harvey_flow_pgs.Geom.upper_aquifer.Upper.Z = 3.8

#-----------------------------------------------------------------------------
# Lower Aquifer Geometry Input
#-----------------------------------------------------------------------------
harvey_flow_pgs.GeomInput.lower_aquifer_input.InputType = 'Box'
harvey_flow_pgs.GeomInput.lower_aquifer_input.GeomName = 'lower_aquifer'

#-----------------------------------------------------------------------------
# Lower Aquifer Geometry
#-----------------------------------------------------------------------------
harvey_flow_pgs.Geom.lower_aquifer.Lower.X = 0.0
harvey_flow_pgs.Geom.lower_aquifer.Lower.Y = 0.0
harvey_flow_pgs.Geom.lower_aquifer.Lower.Z = 0.0

harvey_flow_pgs.Geom.lower_aquifer.Upper.X = 17.0
harvey_flow_pgs.Geom.lower_aquifer.Upper.Y = 10.2
harvey_flow_pgs.Geom.lower_aquifer.Upper.Z = 1.5


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
harvey_flow_pgs.Geom.Perm.Names = 'upper_aquifer lower_aquifer'
# we open a file, in this case from PEST to set upper and lower kg and sigma
#
# fileId = [open stats4.txt r 0600]
kgu = 83
varu = 0.31
kgl = 87.3
varl = 0.22
# close $fileId


## we use the parallel turning bands formulation in ParFlow to simulate
## GRF for upper and lower aquifer
##
harvey_flow_pgs.Geom.upper_aquifer.Perm.Seed = 33335
harvey_flow_pgs.Geom.lower_aquifer.Perm.Seed = 31315

harvey_flow_pgs.Geom.upper_aquifer.Perm.LambdaX = 3.60
harvey_flow_pgs.Geom.upper_aquifer.Perm.LambdaY = 3.60
harvey_flow_pgs.Geom.upper_aquifer.Perm.LambdaZ = 0.19
harvey_flow_pgs.Geom.upper_aquifer.Perm.GeomMean = 112.00

harvey_flow_pgs.Geom.upper_aquifer.Perm.Sigma = 1.0
harvey_flow_pgs.Geom.upper_aquifer.Perm.Sigma = 0.48989794
harvey_flow_pgs.Geom.upper_aquifer.Perm.NumLines = 150
harvey_flow_pgs.Geom.upper_aquifer.Perm.MaxSearchRad = 4
harvey_flow_pgs.Geom.upper_aquifer.Perm.RZeta = 5.0
harvey_flow_pgs.Geom.upper_aquifer.Perm.KMax = 100.0000001
harvey_flow_pgs.Geom.upper_aquifer.Perm.DelK = 0.2
harvey_flow_pgs.Geom.upper_aquifer.Perm.Seed = 33333
harvey_flow_pgs.Geom.upper_aquifer.Perm.LogNormal = 'Log'
harvey_flow_pgs.Geom.upper_aquifer.Perm.StratType = 'Bottom'

harvey_flow_pgs.Geom.lower_aquifer.Perm.LambdaX = 3.60
harvey_flow_pgs.Geom.lower_aquifer.Perm.LambdaY = 3.60
harvey_flow_pgs.Geom.lower_aquifer.Perm.LambdaZ = 0.19

harvey_flow_pgs.Geom.lower_aquifer.Perm.GeomMean = 77.0
harvey_flow_pgs.Geom.lower_aquifer.Perm.Sigma = 1.0
harvey_flow_pgs.Geom.lower_aquifer.Perm.Sigma = 0.48989794
harvey_flow_pgs.Geom.lower_aquifer.Perm.MaxSearchRad = 4
harvey_flow_pgs.Geom.lower_aquifer.Perm.NumLines = 150
harvey_flow_pgs.Geom.lower_aquifer.Perm.RZeta = 5.0
harvey_flow_pgs.Geom.lower_aquifer.Perm.KMax = 100.0000001
harvey_flow_pgs.Geom.lower_aquifer.Perm.DelK = 0.2
harvey_flow_pgs.Geom.lower_aquifer.Perm.Seed = 33333
harvey_flow_pgs.Geom.lower_aquifer.Perm.LogNormal = 'Log'
harvey_flow_pgs.Geom.lower_aquifer.Perm.StratType = 'Bottom'

harvey_flow_pgs.Geom.upper_aquifer.Perm.Seed = 1
harvey_flow_pgs.Geom.upper_aquifer.Perm.MaxNPts = 70
harvey_flow_pgs.Geom.upper_aquifer.Perm.MaxCpts = 20

harvey_flow_pgs.Geom.lower_aquifer.Perm.Seed = 1
harvey_flow_pgs.Geom.lower_aquifer.Perm.MaxNPts = 70
harvey_flow_pgs.Geom.lower_aquifer.Perm.MaxCpts = 20

harvey_flow_pgs.Geom.lower_aquifer.Perm.Type = "TurnBands"
harvey_flow_pgs.Geom.upper_aquifer.Perm.Type = "TurnBands"

# uncomment the lines below to run parallel gaussian instead
# of parallel turning bands

# harvey_flow_pgs.Geom.lower_aquifer.Perm.Type = 'ParGauss'
# harvey_flow_pgs.Geom.upper_aquifer.Perm.Type = 'ParGauss'

#pfset lower aqu and upper aq stats to pest/read in values

harvey_flow_pgs.Geom.upper_aquifer.Perm.GeomMean = kgu
harvey_flow_pgs.Geom.upper_aquifer.Perm.Sigma = varu

harvey_flow_pgs.Geom.lower_aquifer.Perm.GeomMean = kgl
harvey_flow_pgs.Geom.lower_aquifer.Perm.Sigma = varl


harvey_flow_pgs.Perm.TensorType = 'TensorByGeom'

harvey_flow_pgs.Geom.Perm.TensorByGeom.Names = 'domain'

harvey_flow_pgs.Geom.domain.Perm.TensorValX = 1.0
harvey_flow_pgs.Geom.domain.Perm.TensorValY = 1.0
harvey_flow_pgs.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

harvey_flow_pgs.SpecificStorage.Type = 'Constant'
harvey_flow_pgs.SpecificStorage.GeomNames = ''
harvey_flow_pgs.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

harvey_flow_pgs.Phase.Names = 'water'

harvey_flow_pgs.Phase.water.Density.Type = 'Constant'
harvey_flow_pgs.Phase.water.Density.Value = 1.0

harvey_flow_pgs.Phase.water.Viscosity.Type = 'Constant'
harvey_flow_pgs.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
harvey_flow_pgs.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

harvey_flow_pgs.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

harvey_flow_pgs.TimingInfo.BaseUnit = 1.0
harvey_flow_pgs.TimingInfo.StartCount = -1
harvey_flow_pgs.TimingInfo.StartTime = 0.0
harvey_flow_pgs.TimingInfo.StopTime = 0.0
harvey_flow_pgs.TimingInfo.DumpInterval = -1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

harvey_flow_pgs.Geom.Porosity.GeomNames = 'domain'

harvey_flow_pgs.Geom.domain.Porosity.Type = 'Constant'
harvey_flow_pgs.Geom.domain.Porosity.Value = 0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
harvey_flow_pgs.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
harvey_flow_pgs.Phase.water.Mobility.Type = 'Constant'
harvey_flow_pgs.Phase.water.Mobility.Value = 1.0


#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
harvey_flow_pgs.Wells.Names = ''


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
harvey_flow_pgs.Cycle.Names = 'constant'
harvey_flow_pgs.Cycle.constant.Names = 'alltime'
harvey_flow_pgs.Cycle.constant.alltime.Length = 1
harvey_flow_pgs.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
harvey_flow_pgs.BCPressure.PatchNames = 'left right front back bottom top'

harvey_flow_pgs.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
harvey_flow_pgs.Patch.left.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.left.BCPressure.RefGeom = 'domain'
harvey_flow_pgs.Patch.left.BCPressure.RefPatch = 'bottom'
harvey_flow_pgs.Patch.left.BCPressure.alltime.Value = 10.0

harvey_flow_pgs.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
harvey_flow_pgs.Patch.right.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.right.BCPressure.RefGeom = 'domain'
harvey_flow_pgs.Patch.right.BCPressure.RefPatch = 'bottom'
harvey_flow_pgs.Patch.right.BCPressure.alltime.Value = 9.97501

harvey_flow_pgs.Patch.front.BCPressure.Type = 'FluxConst'
harvey_flow_pgs.Patch.front.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.front.BCPressure.alltime.Value = 0.0

harvey_flow_pgs.Patch.back.BCPressure.Type = 'FluxConst'
harvey_flow_pgs.Patch.back.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.back.BCPressure.alltime.Value = 0.0

harvey_flow_pgs.Patch.bottom.BCPressure.Type = 'FluxConst'
harvey_flow_pgs.Patch.bottom.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.bottom.BCPressure.alltime.Value = 0.0

harvey_flow_pgs.Patch.top.BCPressure.Type = 'FluxConst'
harvey_flow_pgs.Patch.top.BCPressure.Cycle = 'constant'
harvey_flow_pgs.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

harvey_flow_pgs.TopoSlopesX.Type = 'Constant'
harvey_flow_pgs.TopoSlopesX.GeomNames = 'domain'

harvey_flow_pgs.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

harvey_flow_pgs.TopoSlopesY.Type = 'Constant'
harvey_flow_pgs.TopoSlopesY.GeomNames = 'domain'

harvey_flow_pgs.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

harvey_flow_pgs.Mannings.Type = 'Constant'
harvey_flow_pgs.Mannings.GeomNames = 'domain'
harvey_flow_pgs.Mannings.Geom.domain.Value = 0.

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

harvey_flow_pgs.PhaseSources.water.Type = 'Constant'
harvey_flow_pgs.PhaseSources.water.GeomNames = 'domain'
harvey_flow_pgs.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
#  Solver Impes  
#-----------------------------------------------------------------------------
harvey_flow_pgs.Solver.MaxIter = 50
harvey_flow_pgs.Solver.AbsTol = 1E-10
harvey_flow_pgs.Solver.Drop = 1E-15

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

harvey_flow_pgs.run()
