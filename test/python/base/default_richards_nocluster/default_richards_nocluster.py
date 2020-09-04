#-----------------------------------------------------------------------------
#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.
#-----------------------------------------------------------------------------

from parflow import Run

default_richards_nocluster = Run("default_richards_nocluster", __file__)

#---------------------------------------------------------

default_richards_nocluster.FileVersion = 4

default_richards_nocluster.UseClustering = False

default_richards_nocluster.Process.Topology.P = 1
default_richards_nocluster.Process.Topology.Q = 1
default_richards_nocluster.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

default_richards_nocluster.ComputationalGrid.Lower.X = -10.0
default_richards_nocluster.ComputationalGrid.Lower.Y = 10.0
default_richards_nocluster.ComputationalGrid.Lower.Z = 1.0

default_richards_nocluster.ComputationalGrid.DX = 8.8888888888888893
default_richards_nocluster.ComputationalGrid.DY = 10.666666666666666
default_richards_nocluster.ComputationalGrid.DZ = 1.0

default_richards_nocluster.ComputationalGrid.NX = 10
default_richards_nocluster.ComputationalGrid.NY = 10
default_richards_nocluster.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

default_richards_nocluster.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------

default_richards_nocluster.GeomInput.domain_input.InputType = 'Box'
default_richards_nocluster.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------

default_richards_nocluster.Geom.domain.Lower.X = -10.0
default_richards_nocluster.Geom.domain.Lower.Y = 10.0
default_richards_nocluster.Geom.domain.Lower.Z = 1.0

default_richards_nocluster.Geom.domain.Upper.X = 150.0
default_richards_nocluster.Geom.domain.Upper.Y = 170.0
default_richards_nocluster.Geom.domain.Upper.Z = 9.0

default_richards_nocluster.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------

default_richards_nocluster.GeomInput.background_input.InputType = 'Box'
default_richards_nocluster.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------

default_richards_nocluster.Geom.background.Lower.X = -99999999.0
default_richards_nocluster.Geom.background.Lower.Y = -99999999.0
default_richards_nocluster.Geom.background.Lower.Z = -99999999.0

default_richards_nocluster.Geom.background.Upper.X = 99999999.0
default_richards_nocluster.Geom.background.Upper.Y = 99999999.0
default_richards_nocluster.Geom.background.Upper.Z = 99999999.0

#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------

default_richards_nocluster.GeomInput.source_region_input.InputType = 'Box'
default_richards_nocluster.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------

default_richards_nocluster.Geom.source_region.Lower.X = 65.56
default_richards_nocluster.Geom.source_region.Lower.Y = 79.34
default_richards_nocluster.Geom.source_region.Lower.Z = 4.5

default_richards_nocluster.Geom.source_region.Upper.X = 74.44
default_richards_nocluster.Geom.source_region.Upper.Y = 89.99
default_richards_nocluster.Geom.source_region.Upper.Z = 5.5

#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------

default_richards_nocluster.GeomInput.concen_region_input.InputType = 'Box'
default_richards_nocluster.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------

default_richards_nocluster.Geom.concen_region.Lower.X = 60.0
default_richards_nocluster.Geom.concen_region.Lower.Y = 80.0
default_richards_nocluster.Geom.concen_region.Lower.Z = 4.0

default_richards_nocluster.Geom.concen_region.Upper.X = 80.0
default_richards_nocluster.Geom.concen_region.Upper.Y = 100.0
default_richards_nocluster.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

default_richards_nocluster.Geom.Perm.Names = 'background'

default_richards_nocluster.Geom.background.Perm.Type = 'Constant'
default_richards_nocluster.Geom.background.Perm.Value = 4.0

default_richards_nocluster.Perm.TensorType = 'TensorByGeom'

default_richards_nocluster.Geom.Perm.TensorByGeom.Names = 'background'

default_richards_nocluster.Geom.background.Perm.TensorValX = 1.0
default_richards_nocluster.Geom.background.Perm.TensorValY = 1.0
default_richards_nocluster.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

default_richards_nocluster.SpecificStorage.Type = 'Constant'
default_richards_nocluster.SpecificStorage.GeomNames = 'domain'
default_richards_nocluster.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

default_richards_nocluster.Phase.Names = 'water'

default_richards_nocluster.Phase.water.Density.Type = 'Constant'
default_richards_nocluster.Phase.water.Density.Value = 1.0

default_richards_nocluster.Phase.water.Viscosity.Type = 'Constant'
default_richards_nocluster.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

default_richards_nocluster.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

default_richards_nocluster.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

default_richards_nocluster.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

default_richards_nocluster.TimingInfo.BaseUnit = 1.0
default_richards_nocluster.TimingInfo.StartCount = 0
default_richards_nocluster.TimingInfo.StartTime = 0.0
default_richards_nocluster.TimingInfo.StopTime = 0.010
default_richards_nocluster.TimingInfo.DumpInterval = -1
default_richards_nocluster.TimeStep.Type = 'Constant'
default_richards_nocluster.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

default_richards_nocluster.Geom.Porosity.GeomNames = 'background'

default_richards_nocluster.Geom.background.Porosity.Type = 'Constant'
default_richards_nocluster.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

default_richards_nocluster.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

default_richards_nocluster.Phase.RelPerm.Type = 'VanGenuchten'
default_richards_nocluster.Phase.RelPerm.GeomNames = 'domain'
default_richards_nocluster.Geom.domain.RelPerm.Alpha = 0.005
default_richards_nocluster.Geom.domain.RelPerm.N = 2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

default_richards_nocluster.Phase.Saturation.Type = 'VanGenuchten'
default_richards_nocluster.Phase.Saturation.GeomNames = 'domain'
default_richards_nocluster.Geom.domain.Saturation.Alpha = 0.005
default_richards_nocluster.Geom.domain.Saturation.N = 2.0
default_richards_nocluster.Geom.domain.Saturation.SRes = 0.2
default_richards_nocluster.Geom.domain.Saturation.SSat = 0.99

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

default_richards_nocluster.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

default_richards_nocluster.Cycle.Names = 'constant'
default_richards_nocluster.Cycle.constant.Names = 'alltime'
default_richards_nocluster.Cycle.constant.alltime.Length = 1
default_richards_nocluster.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

default_richards_nocluster.BCPressure.PatchNames = 'left right front back bottom top'

default_richards_nocluster.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
default_richards_nocluster.Patch.left.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.left.BCPressure.RefGeom = 'domain'
default_richards_nocluster.Patch.left.BCPressure.RefPatch = 'bottom'
default_richards_nocluster.Patch.left.BCPressure.alltime.Value = 5.0

default_richards_nocluster.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
default_richards_nocluster.Patch.right.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.right.BCPressure.RefGeom = 'domain'
default_richards_nocluster.Patch.right.BCPressure.RefPatch = 'bottom'
default_richards_nocluster.Patch.right.BCPressure.alltime.Value = 3.0

default_richards_nocluster.Patch.front.BCPressure.Type = 'FluxConst'
default_richards_nocluster.Patch.front.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.front.BCPressure.alltime.Value = 0.0

default_richards_nocluster.Patch.back.BCPressure.Type = 'FluxConst'
default_richards_nocluster.Patch.back.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.back.BCPressure.alltime.Value = 0.0

default_richards_nocluster.Patch.bottom.BCPressure.Type = 'FluxConst'
default_richards_nocluster.Patch.bottom.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.bottom.BCPressure.alltime.Value = 0.0

default_richards_nocluster.Patch.top.BCPressure.Type = 'FluxConst'
default_richards_nocluster.Patch.top.BCPressure.Cycle = 'constant'
default_richards_nocluster.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

default_richards_nocluster.TopoSlopesX.Type = 'Constant'
default_richards_nocluster.TopoSlopesX.GeomNames = 'domain'
default_richards_nocluster.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

default_richards_nocluster.TopoSlopesY.Type = 'Constant'
default_richards_nocluster.TopoSlopesY.GeomNames = 'domain'
default_richards_nocluster.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

default_richards_nocluster.Mannings.Type = 'Constant'
default_richards_nocluster.Mannings.GeomNames = 'domain'
default_richards_nocluster.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

default_richards_nocluster.ICPressure.Type = 'HydroStaticPatch'
default_richards_nocluster.ICPressure.GeomNames = 'domain'
default_richards_nocluster.Geom.domain.ICPressure.Value = 3.0
default_richards_nocluster.Geom.domain.ICPressure.RefGeom = 'domain'
default_richards_nocluster.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

default_richards_nocluster.PhaseSources.water.Type = 'Constant'
default_richards_nocluster.PhaseSources.water.GeomNames = 'background'
default_richards_nocluster.PhaseSources.water.Geom.background.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

default_richards_nocluster.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

default_richards_nocluster.Solver = 'Richards'
default_richards_nocluster.Solver.MaxIter = 5

default_richards_nocluster.Solver.Nonlinear.MaxIter = 10
default_richards_nocluster.Solver.Nonlinear.ResidualTol = 1e-9
default_richards_nocluster.Solver.Nonlinear.EtaChoice = 'EtaConstant'
default_richards_nocluster.Solver.Nonlinear.EtaValue = 1e-5
default_richards_nocluster.Solver.Nonlinear.UseJacobian = True
default_richards_nocluster.Solver.Nonlinear.DerivativeEpsilon = 1e-2

default_richards_nocluster.Solver.Linear.KrylovDimension = 10

default_richards_nocluster.Solver.Linear.Preconditioner = 'PFMG'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

default_richards_nocluster.run()
