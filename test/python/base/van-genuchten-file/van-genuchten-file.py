#  This runs the basic default_richards test case.
#  This run, as written in this input file, should take
#  3 nonlinear iterations.

#
# Import the ParFlow TCL package
#
from parflow import Run
van_genuchten_file = Run("van_genuchten_file", __file__)

van_genuchten_file.FileVersion = 4

van_genuchten_file.Process.Topology.P = 1
van_genuchten_file.Process.Topology.Q = 1
van_genuchten_file.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
van_genuchten_file.ComputationalGrid.Lower.X = -10.0
van_genuchten_file.ComputationalGrid.Lower.Y = 10.0
van_genuchten_file.ComputationalGrid.Lower.Z = 1.0

van_genuchten_file.ComputationalGrid.DX = 8.8888888888888893
van_genuchten_file.ComputationalGrid.DY = 10.666666666666666
van_genuchten_file.ComputationalGrid.DZ = 1.0

van_genuchten_file.ComputationalGrid.NX = 10
van_genuchten_file.ComputationalGrid.NY = 10
van_genuchten_file.ComputationalGrid.NZ = 8

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
van_genuchten_file.GeomInput.Names = 'domain_input background_input source_region_input concen_region_input'


#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
van_genuchten_file.GeomInput.domain_input.InputType = 'Box'
van_genuchten_file.GeomInput.domain_input.GeomName = 'domain'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
van_genuchten_file.Geom.domain.Lower.X = -10.0
van_genuchten_file.Geom.domain.Lower.Y = 10.0
van_genuchten_file.Geom.domain.Lower.Z = 1.0

van_genuchten_file.Geom.domain.Upper.X = 150.0
van_genuchten_file.Geom.domain.Upper.Y = 170.0
van_genuchten_file.Geom.domain.Upper.Z = 9.0

van_genuchten_file.Geom.domain.Patches = 'left right front back bottom top'

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
van_genuchten_file.GeomInput.background_input.InputType = 'Box'
van_genuchten_file.GeomInput.background_input.GeomName = 'background'

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
van_genuchten_file.Geom.background.Lower.X = -99999999.0
van_genuchten_file.Geom.background.Lower.Y = -99999999.0
van_genuchten_file.Geom.background.Lower.Z = -99999999.0

van_genuchten_file.Geom.background.Upper.X = 99999999.0
van_genuchten_file.Geom.background.Upper.Y = 99999999.0
van_genuchten_file.Geom.background.Upper.Z = 99999999.0


#---------------------------------------------------------
# Source_Region Geometry Input
#---------------------------------------------------------
van_genuchten_file.GeomInput.source_region_input.InputType = 'Box'
van_genuchten_file.GeomInput.source_region_input.GeomName = 'source_region'

#---------------------------------------------------------
# Source_Region Geometry
#---------------------------------------------------------
van_genuchten_file.Geom.source_region.Lower.X = 65.56
van_genuchten_file.Geom.source_region.Lower.Y = 79.34
van_genuchten_file.Geom.source_region.Lower.Z = 4.5

van_genuchten_file.Geom.source_region.Upper.X = 74.44
van_genuchten_file.Geom.source_region.Upper.Y = 89.99
van_genuchten_file.Geom.source_region.Upper.Z = 5.5


#---------------------------------------------------------
# Concen_Region Geometry Input
#---------------------------------------------------------
van_genuchten_file.GeomInput.concen_region_input.InputType = 'Box'
van_genuchten_file.GeomInput.concen_region_input.GeomName = 'concen_region'

#---------------------------------------------------------
# Concen_Region Geometry
#---------------------------------------------------------
van_genuchten_file.Geom.concen_region.Lower.X = 60.0
van_genuchten_file.Geom.concen_region.Lower.Y = 80.0
van_genuchten_file.Geom.concen_region.Lower.Z = 4.0

van_genuchten_file.Geom.concen_region.Upper.X = 80.0
van_genuchten_file.Geom.concen_region.Upper.Y = 100.0
van_genuchten_file.Geom.concen_region.Upper.Z = 6.0

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
van_genuchten_file.Geom.Perm.Names = 'background'

van_genuchten_file.Geom.background.Perm.Type = 'Constant'
van_genuchten_file.Geom.background.Perm.Value = 4.0

van_genuchten_file.Perm.TensorType = 'TensorByGeom'

van_genuchten_file.Geom.Perm.TensorByGeom.Names = 'background'

van_genuchten_file.Geom.background.Perm.TensorValX = 1.0
van_genuchten_file.Geom.background.Perm.TensorValY = 1.0
van_genuchten_file.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

van_genuchten_file.SpecificStorage.Type = 'Constant'
van_genuchten_file.SpecificStorage.GeomNames = 'domain'
van_genuchten_file.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

van_genuchten_file.Phase.Names = 'water'

van_genuchten_file.Phase.water.Density.Type = 'Constant'
van_genuchten_file.Phase.water.Density.Value = 1.0

van_genuchten_file.Phase.water.Viscosity.Type = 'Constant'
van_genuchten_file.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
van_genuchten_file.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
van_genuchten_file.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

van_genuchten_file.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

van_genuchten_file.TimingInfo.BaseUnit = 1.0
van_genuchten_file.TimingInfo.StartCount = 0
van_genuchten_file.TimingInfo.StartTime = 0.0
van_genuchten_file.TimingInfo.StopTime = 0.010
van_genuchten_file.TimingInfo.DumpInterval = -1
van_genuchten_file.TimeStep.Type = 'Constant'
van_genuchten_file.TimeStep.Value = 0.001

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

van_genuchten_file.Geom.Porosity.GeomNames = 'background'

van_genuchten_file.Geom.background.Porosity.Type = 'Constant'
van_genuchten_file.Geom.background.Porosity.Value = 1.0

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
van_genuchten_file.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

van_genuchten_file.Phase.RelPerm.Type = 'VanGenuchten'
van_genuchten_file.Phase.RelPerm.GeomNames = 'domain'

# pfset Geom.domain.RelPerm.Alpha        0.005
# pfset Geom.domain.RelPerm.N            2.0

van_genuchten_file.Phase.RelPerm.VanGenuchten.File = 1
van_genuchten_file.Geom.domain.RelPerm.Alpha.Filename = 'van_genuchten_alpha.pfb'
van_genuchten_file.Geom.domain.RelPerm.N.Filename = 'van_genuchten_n.pfb'

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

van_genuchten_file.Phase.Saturation.Type = 'VanGenuchten'
van_genuchten_file.Phase.Saturation.GeomNames = 'domain'

van_genuchten_file.Geom.domain.Saturation.Alpha = 0.005
van_genuchten_file.Geom.domain.Saturation.N = 2.0
van_genuchten_file.Geom.domain.Saturation.SRes = 0.2
van_genuchten_file.Geom.domain.Saturation.SSat = 0.99

van_genuchten_file.Phase.Saturation.VanGenuchten.File = 1
van_genuchten_file.Geom.domain.Saturation.Alpha.Filename = 'van_genuchten_alpha.pfb'
van_genuchten_file.Geom.domain.Saturation.N.Filename = 'van_genuchten_n.pfb'
van_genuchten_file.Geom.domain.Saturation.SRes.Filename = 'van_genuchten_sr.pfb'
van_genuchten_file.Geom.domain.Saturation.SSat.Filename = 'van_genuchten_ssat.pfb'


#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
van_genuchten_file.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
van_genuchten_file.Cycle.Names = 'constant'
van_genuchten_file.Cycle.constant.Names = 'alltime'
van_genuchten_file.Cycle.constant.alltime.Length = 1
van_genuchten_file.Cycle.constant.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
van_genuchten_file.BCPressure.PatchNames = 'left right front back bottom top'

van_genuchten_file.Patch.left.BCPressure.Type = 'DirEquilRefPatch'
van_genuchten_file.Patch.left.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.left.BCPressure.RefGeom = 'domain'
van_genuchten_file.Patch.left.BCPressure.RefPatch = 'bottom'
van_genuchten_file.Patch.left.BCPressure.alltime.Value = 5.0

van_genuchten_file.Patch.right.BCPressure.Type = 'DirEquilRefPatch'
van_genuchten_file.Patch.right.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.right.BCPressure.RefGeom = 'domain'
van_genuchten_file.Patch.right.BCPressure.RefPatch = 'bottom'
van_genuchten_file.Patch.right.BCPressure.alltime.Value = 3.0

van_genuchten_file.Patch.front.BCPressure.Type = 'FluxConst'
van_genuchten_file.Patch.front.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.front.BCPressure.alltime.Value = 0.0

van_genuchten_file.Patch.back.BCPressure.Type = 'FluxConst'
van_genuchten_file.Patch.back.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.back.BCPressure.alltime.Value = 0.0

van_genuchten_file.Patch.bottom.BCPressure.Type = 'FluxConst'
van_genuchten_file.Patch.bottom.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.bottom.BCPressure.alltime.Value = 0.0

van_genuchten_file.Patch.top.BCPressure.Type = 'FluxConst'
van_genuchten_file.Patch.top.BCPressure.Cycle = 'constant'
van_genuchten_file.Patch.top.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

van_genuchten_file.TopoSlopesX.Type = 'Constant'
van_genuchten_file.TopoSlopesX.GeomNames = 'domain'

van_genuchten_file.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

van_genuchten_file.TopoSlopesY.Type = 'Constant'
van_genuchten_file.TopoSlopesY.GeomNames = 'domain'

van_genuchten_file.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

van_genuchten_file.Mannings.Type = 'Constant'
van_genuchten_file.Mannings.GeomNames = 'domain'
van_genuchten_file.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

van_genuchten_file.ICPressure.Type = 'HydroStaticPatch'
van_genuchten_file.ICPressure.GeomNames = 'domain'
van_genuchten_file.Geom.domain.ICPressure.Value = 3.0
van_genuchten_file.Geom.domain.ICPressure.RefGeom = 'domain'
van_genuchten_file.Geom.domain.ICPressure.RefPatch = 'bottom'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

van_genuchten_file.PhaseSources.water.Type = 'Constant'
van_genuchten_file.PhaseSources.water.GeomNames = 'background'
van_genuchten_file.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

van_genuchten_file.KnownSolution = 'NoKnownSolution'


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
van_genuchten_file.Solver = 'Richards'
van_genuchten_file.Solver.MaxIter = 5

van_genuchten_file.Solver.Nonlinear.MaxIter = 10
van_genuchten_file.Solver.Nonlinear.ResidualTol = 1e-9
van_genuchten_file.Solver.Nonlinear.EtaChoice = 'EtaConstant'
van_genuchten_file.Solver.Nonlinear.EtaValue = 1e-5
van_genuchten_file.Solver.Nonlinear.UseJacobian = True
van_genuchten_file.Solver.Nonlinear.DerivativeEpsilon = 1e-2

van_genuchten_file.Solver.Linear.KrylovDimension = 10

van_genuchten_file.Solver.Linear.Preconditioner = 'PFMG'
# van_genuchten_file.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
# van_genuchten_file.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


van_genuchten_file.run()
