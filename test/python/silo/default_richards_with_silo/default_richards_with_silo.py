#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
from parflow import Run
default_richards_with_silo = Run("default_richards_with_silo", __file__)

# Examples of compression options for SILO
# Note compression only works for HDF5
#pfset SILO.Filetype "HDF5"
#pfset SILO.CompressionOptions "METHOD=GZIP"
#pfset SILO.CompressionOptions "METHOD=SZIP"
#pfset SILO.CompressionOptions "METHOD=FPZIP"
#pfset SILO.CompressionOptions "ERRMODE=FALLBACK METHOD=GZIP"

default_richards_with_silo.FileVersion = 4

default_richards_with_silo.Process.Topology.P = 1
default_richards_with_silo.Process.Topology.Q = 1
default_richards_with_silo.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
default_richards_with_silo.ComputationalGrid.Lower.X = -10.0
default_richards_with_silo.ComputationalGrid.Lower.Y = 10.0
default_richards_with_silo.ComputationalGrid.Lower.Z = 1.0

default_richards_with_silo.ComputationalGrid.DX = 8.8888888888888893
default_richards_with_silo.ComputationalGrid.DY = 10.666666666666666
default_richards_with_silo.ComputationalGrid.DZ = 1.0

default_richards_with_silo.ComputationalGrid.NX = 10
default_richards_with_silo.ComputationalGrid.NY = 10
default_richards_with_silo.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
default_richards_with_silo.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
default_richards_with_silo.GeomInput.domain_input.InputType = 'Box'
default_richards_with_silo.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
default_richards_with_silo.Geom.domain.Lower.X = -10.0
default_richards_with_silo.Geom.domain.Lower.Y = 10.0
default_richards_with_silo.Geom.domain.Lower.Z = 1.0

default_richards_with_silo.Geom.domain.Upper.X = 150.0
default_richards_with_silo.Geom.domain.Upper.Y = 170.0
default_richards_with_silo.Geom.domain.Upper.Z = 9.0

default_richards_with_silo.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
default_richards_with_silo.GeomInput.background_input.InputType = 'Box'
default_richards_with_silo.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
default_richards_with_silo.Geom.background.Lower.X = -99999999.0
default_richards_with_silo.Geom.background.Lower.Y = -99999999.0
default_richards_with_silo.Geom.background.Lower.Z = -99999999.0

default_richards_with_silo.Geom.background.Upper.X = 99999999.0
default_richards_with_silo.Geom.background.Upper.Y = 99999999.0
default_richards_with_silo.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
default_richards_with_silo.GeomInput.source_region_input.InputType = 'Box'
default_richards_with_silo.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
default_richards_with_silo.Geom.source_region.Lower.X = 65.56
default_richards_with_silo.Geom.source_region.Lower.Y = 79.34
default_richards_with_silo.Geom.source_region.Lower.Z = 4.5

default_richards_with_silo.Geom.source_region.Upper.X = 74.44
default_richards_with_silo.Geom.source_region.Upper.Y = 89.99
default_richards_with_silo.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
default_richards_with_silo.GeomInput.concen_region_input.InputType = 'Box'
default_richards_with_silo.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
default_richards_with_silo.Geom.concen_region.Lower.X = 60.0
default_richards_with_silo.Geom.concen_region.Lower.Y = 80.0
default_richards_with_silo.Geom.concen_region.Lower.Z = 4.0

default_richards_with_silo.Geom.concen_region.Upper.X = 80.0
default_richards_with_silo.Geom.concen_region.Upper.Y = 100.0
default_richards_with_silo.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
default_richards_with_silo.Geom.Perm.Names = 'background'

default_richards_with_silo.Geom.background.Perm.Type = 'Constant'
default_richards_with_silo.Geom.background.Perm.Value = 4.0

default_richards_with_silo.Perm.TensorType = 'TensorByGeom'

default_richards_with_silo.Geom.Perm.TensorByGeom.Names = 'background'

default_richards_with_silo.Geom.background.Perm.TensorValX = 1.0
default_richards_with_silo.Geom.background.Perm.TensorValY = 1.0
default_richards_with_silo.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

default_richards_with_silo.SpecificStorage.Type = 'Constant'
default_richards_with_silo.SpecificStorage.GeomNames = 'domain'
default_richards_with_silo.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

default_richards_with_silo.Phase.Names = 'water'

default_richards_with_silo.Phase.water.Density.Type = 'Constant'
default_richards_with_silo.Phase.water.Density.Value = 1.0

default_richards_with_silo.Phase.water.Viscosity.Type = 'Constant'
default_richards_with_silo.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
default_richards_with_silo.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
default_richards_with_silo.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

default_richards_with_silo.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

default_richards_with_silo.TimingInfo.BaseUnit = 1.0
default_richards_with_silo.TimingInfo.StartCount = 0
default_richards_with_silo.TimingInfo.StartTime = 0.0
default_richards_with_silo.TimingInfo.StopTime = 0.010
default_richards_with_silo.TimingInfo.DumpInterval = -1
default_richards_with_silo.TimeStep.Type = 'Constant'
default_richards_with_silo.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

default_richards_with_silo.Geom.Porosity.GeomNames = 'background'

default_richards_with_silo.Geom.background.Porosity.Type = 'Constant'
default_richards_with_silo.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
default_richards_with_silo.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

default_richards_with_silo.Phase.RelPerm.Type = 'VanGenuchten'
default_richards_with_silo.Phase.RelPerm.GeomNames = 'domain'
default_richards_with_silo.Geom.domain.RelPerm.Alpha = 0.005
default_richards_with_silo.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

default_richards_with_silo.Phase.Saturation.Type = 'VanGenuchten'
default_richards_with_silo.Phase.Saturation.GeomNames = 'domain'
default_richards_with_silo.Geom.domain.Saturation.Alpha = 0.005
default_richards_with_silo.Geom.domain.Saturation.N = 2.0
default_richards_with_silo.Geom.domain.Saturation.SRes = 0.2
default_richards_with_silo.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
default_richards_with_silo.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
default_richards_with_silo.Cycle.Names = 'constant'
default_richards_with_silo.Cycle.constant.Names = 'alltime'
default_richards_with_silo.Cycle.constant.alltime.Length = 1
default_richards_with_silo.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
default_richards_with_silo.BCPressure.PatchNames = 'left right front back bottom top'

default_richards_with_silo.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
default_richards_with_silo.Patch.left.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.left.BCPressure.RefGeom = 'domain'
default_richards_with_silo.Patch.left.BCPressure.RefPatch = 'bottom'
default_richards_with_silo.Patch.left.BCPressure.alltime.Value = 5.0

default_richards_with_silo.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
default_richards_with_silo.Patch.right.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.right.BCPressure.RefGeom = 'domain'
default_richards_with_silo.Patch.right.BCPressure.RefPatch = 'bottom'
default_richards_with_silo.Patch.right.BCPressure.alltime.Value = 3.0

default_richards_with_silo.Patch.front.BCPressure.Type = 'FluxConst'
default_richards_with_silo.Patch.front.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.front.BCPressure.alltime.Value = 0.0

default_richards_with_silo.Patch.back.BCPressure.Type = 'FluxConst'
default_richards_with_silo.Patch.back.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.back.BCPressure.alltime.Value = 0.0

default_richards_with_silo.Patch.bottom.BCPressure.Type = 'FluxConst'
default_richards_with_silo.Patch.bottom.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.bottom.BCPressure.alltime.Value = 0.0

default_richards_with_silo.Patch.top.BCPressure.Type = 'FluxConst'
default_richards_with_silo.Patch.top.BCPressure.Cycle = 'constant'
default_richards_with_silo.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

default_richards_with_silo.TopoSlopesX.Type = 'Constant'
default_richards_with_silo.TopoSlopesX.GeomNames = 'domain'

default_richards_with_silo.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

default_richards_with_silo.TopoSlopesY.Type = 'Constant'
default_richards_with_silo.TopoSlopesY.GeomNames = 'domain'

default_richards_with_silo.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

default_richards_with_silo.Mannings.Type = 'Constant'
default_richards_with_silo.Mannings.GeomNames = 'domain'
default_richards_with_silo.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

default_richards_with_silo.ICPressure.Type = 'HydroStaticPatch'
default_richards_with_silo.ICPressure.GeomNames = 'domain'
default_richards_with_silo.Geom.domain.ICPressure.Value = 3.0
default_richards_with_silo.Geom.domain.ICPressure.RefGeom = 'domain'
default_richards_with_silo.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

default_richards_with_silo.PhaseSources.water.Type = 'Constant'
default_richards_with_silo.PhaseSources.water.GeomNames = 'background'
default_richards_with_silo.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

default_richards_with_silo.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
default_richards_with_silo.Solver = 'Richards'
default_richards_with_silo.Solver.MaxIter = 5

default_richards_with_silo.Solver.Nonlinear.MaxIter = 10
default_richards_with_silo.Solver.Nonlinear.ResidualTol = 1e-9
default_richards_with_silo.Solver.Nonlinear.EtaChoice = 'EtaConstant'
default_richards_with_silo.Solver.Nonlinear.EtaValue = 1e-5
default_richards_with_silo.Solver.Nonlinear.UseJacobian = True
default_richards_with_silo.Solver.Nonlinear.DerivativeEpsilon = 1e-2

default_richards_with_silo.Solver.Linear.KrylovDimension = 10

default_richards_with_silo.Solver.Linear.Preconditioner = 'PFMGOctree'
# default_richards_with_silo.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
# default_richards_with_silo.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100


default_richards_with_silo.Solver.WriteSiloSubsurfData = True
default_richards_with_silo.Solver.WriteSiloPressure = True
default_richards_with_silo.Solver.WriteSiloSaturation = True
default_richards_with_silo.Solver.WriteSiloConcentration = True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

default_richards_with_silo.run()
