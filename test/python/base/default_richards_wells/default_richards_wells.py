#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
from parflow import Run
default_richards_wells = Run("default_richards_wells", __file__)

# set runname default_richards_wells

# Examples of compression options for SILO
# Note compression only works for HDF5
#pfset SILO.Filetype "HDF5"
#pfset SILO.CompressionOptions "METHOD=GZIP"
#pfset SILO.CompressionOptions "METHOD=SZIP"
#pfset SILO.CompressionOptions "METHOD=FPZIP"
#pfset SILO.CompressionOptions "ERRMODE=FALLBACK METHOD=GZIP"

default_richards_wells.FileVersion = 4

default_richards_wells.Process.Topology.P = 1
default_richards_wells.Process.Topology.Q = 1
default_richards_wells.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
default_richards_wells.ComputationalGrid.Lower.X = -10.0
default_richards_wells.ComputationalGrid.Lower.Y = 10.0
default_richards_wells.ComputationalGrid.Lower.Z = 1.0

default_richards_wells.ComputationalGrid.DX = 8.8888888888888893
default_richards_wells.ComputationalGrid.DY = 10.666666666666666
default_richards_wells.ComputationalGrid.DZ = 1.0

default_richards_wells.ComputationalGrid.NX = 10
default_richards_wells.ComputationalGrid.NY = 10
default_richards_wells.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
default_richards_wells.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
default_richards_wells.GeomInput.domain_input.InputType = 'Box'
default_richards_wells.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
default_richards_wells.Geom.domain.Lower.X = -10.0
default_richards_wells.Geom.domain.Lower.Y = 10.0
default_richards_wells.Geom.domain.Lower.Z = 1.0

default_richards_wells.Geom.domain.Upper.X = 150.0
default_richards_wells.Geom.domain.Upper.Y = 170.0
default_richards_wells.Geom.domain.Upper.Z = 9.0

default_richards_wells.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
default_richards_wells.GeomInput.background_input.InputType = 'Box'
default_richards_wells.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
default_richards_wells.Geom.background.Lower.X = -99999999.0
default_richards_wells.Geom.background.Lower.Y = -99999999.0
default_richards_wells.Geom.background.Lower.Z = -99999999.0

default_richards_wells.Geom.background.Upper.X = 99999999.0
default_richards_wells.Geom.background.Upper.Y = 99999999.0
default_richards_wells.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
default_richards_wells.GeomInput.source_region_input.InputType = 'Box'
default_richards_wells.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
default_richards_wells.Geom.source_region.Lower.X = 65.56
default_richards_wells.Geom.source_region.Lower.Y = 79.34
default_richards_wells.Geom.source_region.Lower.Z = 4.5

default_richards_wells.Geom.source_region.Upper.X = 74.44
default_richards_wells.Geom.source_region.Upper.Y = 89.99
default_richards_wells.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
default_richards_wells.GeomInput.concen_region_input.InputType = 'Box'
default_richards_wells.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
default_richards_wells.Geom.concen_region.Lower.X = 60.0
default_richards_wells.Geom.concen_region.Lower.Y = 80.0
default_richards_wells.Geom.concen_region.Lower.Z = 4.0

default_richards_wells.Geom.concen_region.Upper.X = 80.0
default_richards_wells.Geom.concen_region.Upper.Y = 100.0
default_richards_wells.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
default_richards_wells.Geom.Perm.Names = 'background'

default_richards_wells.Geom.background.Perm.Type = 'Constant'
default_richards_wells.Geom.background.Perm.Value = 4.0

default_richards_wells.Perm.TensorType = 'TensorByGeom'

default_richards_wells.Geom.Perm.TensorByGeom.Names = 'background'

default_richards_wells.Geom.background.Perm.TensorValX = 1.0
default_richards_wells.Geom.background.Perm.TensorValY = 1.0
default_richards_wells.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

default_richards_wells.SpecificStorage.Type = 'Constant'
default_richards_wells.SpecificStorage.GeomNames = 'domain'
default_richards_wells.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

default_richards_wells.Phase.Names = 'water'

default_richards_wells.Phase.water.Density.Type = 'Constant'
default_richards_wells.Phase.water.Density.Value = 1.0

default_richards_wells.Phase.water.Viscosity.Type = 'Constant'
default_richards_wells.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
default_richards_wells.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
default_richards_wells.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

default_richards_wells.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

default_richards_wells.TimingInfo.BaseUnit = 1.0
default_richards_wells.TimingInfo.StartCount = 0
default_richards_wells.TimingInfo.StartTime = 0.0
default_richards_wells.TimingInfo.StopTime = 0.010
default_richards_wells.TimingInfo.DumpInterval = -1
default_richards_wells.TimeStep.Type = 'Constant'
default_richards_wells.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

default_richards_wells.Geom.Porosity.GeomNames = 'background'

default_richards_wells.Geom.background.Porosity.Type = 'Constant'
default_richards_wells.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
default_richards_wells.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

default_richards_wells.Phase.RelPerm.Type = 'VanGenuchten'
default_richards_wells.Phase.RelPerm.GeomNames = 'domain'
default_richards_wells.Geom.domain.RelPerm.Alpha = 0.005
default_richards_wells.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

default_richards_wells.Phase.Saturation.Type = 'VanGenuchten'
default_richards_wells.Phase.Saturation.GeomNames = 'domain'
default_richards_wells.Geom.domain.Saturation.Alpha = 0.005
default_richards_wells.Geom.domain.Saturation.N = 2.0
default_richards_wells.Geom.domain.Saturation.SRes = 0.2
default_richards_wells.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
default_richards_wells.Cycle.Names = 'constant'
default_richards_wells.Cycle.constant.Names = 'alltime'
default_richards_wells.Cycle.constant.alltime.Length = 1
default_richards_wells.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

default_richards_wells.Wells.Names = 'pumping_well'
default_richards_wells.Wells.pumping_well.InputType = 'Vertical'
default_richards_wells.Wells.pumping_well.Action = 'Extraction'
default_richards_wells.Wells.pumping_well.Type = 'Pressure'
default_richards_wells.Wells.pumping_well.X = 0
default_richards_wells.Wells.pumping_well.Y = 80
default_richards_wells.Wells.pumping_well.ZUpper = 3.0
default_richards_wells.Wells.pumping_well.ZLower = 2.00
default_richards_wells.Wells.pumping_well.Method = 'Standard'
default_richards_wells.Wells.pumping_well.Cycle = 'constant'
default_richards_wells.Wells.pumping_well.alltime.Pressure.Value = 0.5
default_richards_wells.Wells.pumping_well.alltime.Saturation.water.Value = 1.0

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
default_richards_wells.BCPressure.PatchNames = 'left right front back bottom top'

default_richards_wells.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
default_richards_wells.Patch.left.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.left.BCPressure.RefGeom = 'domain'
default_richards_wells.Patch.left.BCPressure.RefPatch = 'bottom'
default_richards_wells.Patch.left.BCPressure.alltime.Value = 5.0

default_richards_wells.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
default_richards_wells.Patch.right.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.right.BCPressure.RefGeom = 'domain'
default_richards_wells.Patch.right.BCPressure.RefPatch = 'bottom'
default_richards_wells.Patch.right.BCPressure.alltime.Value = 5.0

default_richards_wells.Patch.front.BCPressure.Type = 'FluxConst'
default_richards_wells.Patch.front.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.front.BCPressure.alltime.Value = 0.0

default_richards_wells.Patch.back.BCPressure.Type = 'FluxConst'
default_richards_wells.Patch.back.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.back.BCPressure.alltime.Value = 0.0

default_richards_wells.Patch.bottom.BCPressure.Type = 'FluxConst'
default_richards_wells.Patch.bottom.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.bottom.BCPressure.alltime.Value = 0.0

default_richards_wells.Patch.top.BCPressure.Type = 'FluxConst'
default_richards_wells.Patch.top.BCPressure.Cycle = 'constant'
default_richards_wells.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

default_richards_wells.TopoSlopesX.Type = 'Constant'
default_richards_wells.TopoSlopesX.GeomNames = 'domain'

default_richards_wells.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

default_richards_wells.TopoSlopesY.Type = 'Constant'
default_richards_wells.TopoSlopesY.GeomNames = 'domain'

default_richards_wells.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

default_richards_wells.Mannings.Type = 'Constant'
default_richards_wells.Mannings.GeomNames = 'domain'
default_richards_wells.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

default_richards_wells.ICPressure.Type = 'HydroStaticPatch'
default_richards_wells.ICPressure.GeomNames = 'domain'
default_richards_wells.Geom.domain.ICPressure.Value = 5.0
default_richards_wells.Geom.domain.ICPressure.RefGeom = 'domain'
default_richards_wells.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

default_richards_wells.PhaseSources.water.Type = 'Constant'
default_richards_wells.PhaseSources.water.GeomNames = 'background'
default_richards_wells.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

default_richards_wells.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
default_richards_wells.Solver = 'Richards'
default_richards_wells.Solver.MaxIter = 5

default_richards_wells.Solver.Nonlinear.MaxIter = 10
default_richards_wells.Solver.Nonlinear.ResidualTol = 1e-9
default_richards_wells.Solver.Nonlinear.EtaChoice = 'EtaConstant'
default_richards_wells.Solver.Nonlinear.EtaValue = 1e-5
default_richards_wells.Solver.Nonlinear.UseJacobian = True
default_richards_wells.Solver.Nonlinear.DerivativeEpsilon = 1e-2

default_richards_wells.Solver.Linear.KrylovDimension = 10

default_richards_wells.Solver.Linear.Preconditioner = 'MGSemi'
# default_richards_wells.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
# default_richards_wells.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100


#pfset Solver.WriteSiloSubsurfData True
#pfset Solver.WriteSiloPressure True
#pfset Solver.WriteSiloSaturation True
#pfset Solver.WriteSiloConcentration True

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


default_richards_wells.run()
