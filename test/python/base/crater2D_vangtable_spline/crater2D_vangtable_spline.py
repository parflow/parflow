#  This is a 2D crater problem w/ time varying input and topography
#    Reed Maxwell, 11/06
#     
#      

#
# Import the ParFlow TCL package
#
from parflow import Run
crater2D_vangtable_spline = Run("crater2D_vangtable_spline", __file__)

# set runname  crater2D_vangtable_spline

#---------------------------------------------------------
# Controls for the VanG curves used later.
#---------------------------------------------------------
#set VG_points 0
VG_points = 20000
VG_alpha = 1.0
VG_N = 2.0

crater2D_vangtable_spline.FileVersion = 4

crater2D_vangtable_spline.Process.Topology.P = 1
crater2D_vangtable_spline.Process.Topology.Q = 1
crater2D_vangtable_spline.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
crater2D_vangtable_spline.ComputationalGrid.Lower.X = 0.0
crater2D_vangtable_spline.ComputationalGrid.Lower.Y = 0.0
crater2D_vangtable_spline.ComputationalGrid.Lower.Z = 0.0

crater2D_vangtable_spline.ComputationalGrid.NX = 100
crater2D_vangtable_spline.ComputationalGrid.NY = 1
crater2D_vangtable_spline.ComputationalGrid.NZ = 100

UpperX = 400
UpperY = 1.0
UpperZ = 200

LowerX = crater2D_vangtable_spline.ComputationalGrid.Lower.X
LowerY = crater2D_vangtable_spline.ComputationalGrid.Lower.Y
LowerZ = crater2D_vangtable_spline.ComputationalGrid.Lower.Z

NX = crater2D_vangtable_spline.ComputationalGrid.NX
NY = crater2D_vangtable_spline.ComputationalGrid.NY
NZ = crater2D_vangtable_spline.ComputationalGrid.NZ

crater2D_vangtable_spline.ComputationalGrid.DX = (UpperX - LowerX) / NX
crater2D_vangtable_spline.ComputationalGrid.DY = (UpperY - LowerY) / NY
crater2D_vangtable_spline.ComputationalGrid.DZ = (UpperZ - LowerZ) / NZ

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
Zones = "zone1 zone2 zone3above4 zone3left4 zone3right4 zone3below4 zone4"

crater2D_vangtable_spline.GeomInput.Names = f'solidinput {Zones} background'

crater2D_vangtable_spline.GeomInput.solidinput.InputType = 'SolidFile'
crater2D_vangtable_spline.GeomInput.solidinput.GeomNames = 'domain'
crater2D_vangtable_spline.GeomInput.solidinput.FileName = 'crater2D.pfsol'

crater2D_vangtable_spline.GeomInput.zone1.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone1.GeomName = 'zone1'

crater2D_vangtable_spline.Geom.zone1.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone1.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone1.Lower.Z = 0.0
crater2D_vangtable_spline.Geom.zone1.Upper.X = 400.0
crater2D_vangtable_spline.Geom.zone1.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone1.Upper.Z = 200.0

crater2D_vangtable_spline.GeomInput.zone2.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone2.GeomName = 'zone2'

crater2D_vangtable_spline.Geom.zone2.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone2.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone2.Lower.Z = 60.0
crater2D_vangtable_spline.Geom.zone2.Upper.X = 200.0
crater2D_vangtable_spline.Geom.zone2.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone2.Upper.Z = 80.0

crater2D_vangtable_spline.GeomInput.zone3above4.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone3above4.GeomName = 'zone3above4'

crater2D_vangtable_spline.Geom.zone3above4.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone3above4.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone3above4.Lower.Z = 180.0
crater2D_vangtable_spline.Geom.zone3above4.Upper.X = 200.0
crater2D_vangtable_spline.Geom.zone3above4.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone3above4.Upper.Z = 200.0

crater2D_vangtable_spline.GeomInput.zone3left4.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone3left4.GeomName = 'zone3left4'

crater2D_vangtable_spline.Geom.zone3left4.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone3left4.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone3left4.Lower.Z = 190.0
crater2D_vangtable_spline.Geom.zone3left4.Upper.X = 100.0
crater2D_vangtable_spline.Geom.zone3left4.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone3left4.Upper.Z = 200.0

crater2D_vangtable_spline.GeomInput.zone3right4.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone3right4.GeomName = 'zone3right4'

crater2D_vangtable_spline.Geom.zone3right4.Lower.X = 30.0
crater2D_vangtable_spline.Geom.zone3right4.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone3right4.Lower.Z = 90.0
crater2D_vangtable_spline.Geom.zone3right4.Upper.X = 80.0
crater2D_vangtable_spline.Geom.zone3right4.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone3right4.Upper.Z = 100.0

crater2D_vangtable_spline.GeomInput.zone3below4.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone3below4.GeomName = 'zone3below4'

crater2D_vangtable_spline.Geom.zone3below4.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone3below4.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone3below4.Lower.Z = 0.0
crater2D_vangtable_spline.Geom.zone3below4.Upper.X = 400.0
crater2D_vangtable_spline.Geom.zone3below4.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone3below4.Upper.Z = 20.0

crater2D_vangtable_spline.GeomInput.zone4.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.zone4.GeomName = 'zone4'

crater2D_vangtable_spline.Geom.zone4.Lower.X = 0.0
crater2D_vangtable_spline.Geom.zone4.Lower.Y = 0.0
crater2D_vangtable_spline.Geom.zone4.Lower.Z = 100.0
crater2D_vangtable_spline.Geom.zone4.Upper.X = 300.0
crater2D_vangtable_spline.Geom.zone4.Upper.Y = 1.0
crater2D_vangtable_spline.Geom.zone4.Upper.Z = 150.0

crater2D_vangtable_spline.GeomInput.background.InputType = 'Box'
crater2D_vangtable_spline.GeomInput.background.GeomName = 'background'

crater2D_vangtable_spline.Geom.background.Lower.X = -99999999.0
crater2D_vangtable_spline.Geom.background.Lower.Y = -99999999.0
crater2D_vangtable_spline.Geom.background.Lower.Z = -99999999.0
crater2D_vangtable_spline.Geom.background.Upper.X = 99999999.0
crater2D_vangtable_spline.Geom.background.Upper.Y = 99999999.0
crater2D_vangtable_spline.Geom.background.Upper.Z = 99999999.0

crater2D_vangtable_spline.Geom.domain.Patches = 'infiltration z_upper x_lower y_lower \
    x_upper y_upper z_lower'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
crater2D_vangtable_spline.Geom.Perm.Names = Zones



crater2D_vangtable_spline.Geom.zone1.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone1.Perm.Value = 9.1496

crater2D_vangtable_spline.Geom.zone2.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone2.Perm.Value = 5.4427

crater2D_vangtable_spline.Geom.zone3above4.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3above4.Perm.Value = 4.8033

crater2D_vangtable_spline.Geom.zone3left4.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3left4.Perm.Value = 4.8033

crater2D_vangtable_spline.Geom.zone3right4.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3right4.Perm.Value = 4.8033

crater2D_vangtable_spline.Geom.zone3below4.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3below4.Perm.Value = 4.8033

crater2D_vangtable_spline.Geom.zone4.Perm.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone4.Perm.Value = .48033

crater2D_vangtable_spline.Perm.TensorType = 'TensorByGeom'

crater2D_vangtable_spline.Geom.Perm.TensorByGeom.Names = 'background'

crater2D_vangtable_spline.Geom.background.Perm.TensorValX = 1.0
crater2D_vangtable_spline.Geom.background.Perm.TensorValY = 1.0
crater2D_vangtable_spline.Geom.background.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.SpecificStorage.Type = 'Constant'
crater2D_vangtable_spline.SpecificStorage.GeomNames = 'domain'
crater2D_vangtable_spline.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Phase.Names = 'water'

crater2D_vangtable_spline.Phase.water.Density.Type = 'Constant'
crater2D_vangtable_spline.Phase.water.Density.Value = 1.0

crater2D_vangtable_spline.Phase.water.Viscosity.Type = 'Constant'
crater2D_vangtable_spline.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Contaminants.Names = ''


#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Geom.Retardation.GeomNames = ''


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.TimingInfo.BaseUnit = 1.0
crater2D_vangtable_spline.TimingInfo.StartCount = 0
crater2D_vangtable_spline.TimingInfo.StartTime = 0.0
crater2D_vangtable_spline.TimingInfo.StopTime = 20.0
crater2D_vangtable_spline.TimingInfo.DumpInterval = 10.0
crater2D_vangtable_spline.TimeStep.Type = 'Constant'
crater2D_vangtable_spline.TimeStep.Value = 10.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Geom.Porosity.GeomNames = Zones

crater2D_vangtable_spline.Geom.zone1.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone1.Porosity.Value = 0.3680

crater2D_vangtable_spline.Geom.zone2.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone2.Porosity.Value = 0.3510

crater2D_vangtable_spline.Geom.zone3above4.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3above4.Porosity.Value = 0.3250

crater2D_vangtable_spline.Geom.zone3left4.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3left4.Porosity.Value = 0.3250

crater2D_vangtable_spline.Geom.zone3right4.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3right4.Porosity.Value = 0.3250

crater2D_vangtable_spline.Geom.zone3below4.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone3below4.Porosity.Value = 0.3250

crater2D_vangtable_spline.Geom.zone4.Porosity.Type = 'Constant'
crater2D_vangtable_spline.Geom.zone4.Porosity.Value = 0.3250

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.Phase.RelPerm.Type = 'VanGenuchten'
crater2D_vangtable_spline.Phase.RelPerm.GeomNames = Zones

crater2D_vangtable_spline.Geom.zone1.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone1.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone1.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone1.RelPerm.MinPressureHead = -300

crater2D_vangtable_spline.Geom.zone2.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone2.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone2.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone2.RelPerm.MinPressureHead = -300


crater2D_vangtable_spline.Geom.zone3above4.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3above4.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone3above4.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone3above4.RelPerm.MinPressureHead = -300

crater2D_vangtable_spline.Geom.zone3left4.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3left4.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone3left4.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone3left4.RelPerm.MinPressureHead = -300

crater2D_vangtable_spline.Geom.zone3right4.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3right4.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone3right4.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone3right4.RelPerm.MinPressureHead = -300

crater2D_vangtable_spline.Geom.zone3below4.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3below4.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone3below4.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone3below4.RelPerm.MinPressureHead = -300

crater2D_vangtable_spline.Geom.zone4.RelPerm.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone4.RelPerm.N = VG_N
crater2D_vangtable_spline.Geom.zone4.RelPerm.NumSamplePoints = VG_points
crater2D_vangtable_spline.Geom.zone4.RelPerm.MinPressureHead = -300

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

crater2D_vangtable_spline.Phase.Saturation.Type = 'VanGenuchten'
crater2D_vangtable_spline.Phase.Saturation.GeomNames = Zones

crater2D_vangtable_spline.Geom.zone1.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone1.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone1.Saturation.SRes = 0.2771
crater2D_vangtable_spline.Geom.zone1.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone2.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone2.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone2.Saturation.SRes = 0.2806
crater2D_vangtable_spline.Geom.zone2.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone3above4.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3above4.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone3above4.Saturation.SRes = 0.2643
crater2D_vangtable_spline.Geom.zone3above4.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone3left4.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3left4.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone3left4.Saturation.SRes = 0.2643
crater2D_vangtable_spline.Geom.zone3left4.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone3right4.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3right4.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone3right4.Saturation.SRes = 0.2643
crater2D_vangtable_spline.Geom.zone3right4.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone3below4.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone3below4.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone3below4.Saturation.SRes = 0.2643
crater2D_vangtable_spline.Geom.zone3below4.Saturation.SSat = 1.0

crater2D_vangtable_spline.Geom.zone4.Saturation.Alpha = VG_alpha
crater2D_vangtable_spline.Geom.zone4.Saturation.N = VG_N
crater2D_vangtable_spline.Geom.zone4.Saturation.SRes = 0.2643
crater2D_vangtable_spline.Geom.zone4.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
crater2D_vangtable_spline.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
crater2D_vangtable_spline.Cycle.Names = 'constant onoff'
crater2D_vangtable_spline.Cycle.constant.Names = 'alltime'
crater2D_vangtable_spline.Cycle.constant.alltime.Length = 1
crater2D_vangtable_spline.Cycle.constant.Repeat = -1

crater2D_vangtable_spline.Cycle.onoff.Names = 'on off'
crater2D_vangtable_spline.Cycle.onoff.on.Length = 10
crater2D_vangtable_spline.Cycle.onoff.off.Length = 90
crater2D_vangtable_spline.Cycle.onoff.Repeat = -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
crater2D_vangtable_spline.BCPressure.PatchNames = crater2D_vangtable_spline.Geom.domain.Patches

crater2D_vangtable_spline.Patch.infiltration.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.infiltration.BCPressure.Cycle = 'onoff'
crater2D_vangtable_spline.Patch.infiltration.BCPressure.on.Value = -0.10
crater2D_vangtable_spline.Patch.infiltration.BCPressure.off.Value = 0.0

crater2D_vangtable_spline.Patch.x_lower.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.x_lower.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.x_lower.BCPressure.alltime.Value = 0.0

crater2D_vangtable_spline.Patch.y_lower.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.y_lower.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.y_lower.BCPressure.alltime.Value = 0.0

crater2D_vangtable_spline.Patch.z_lower.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.z_lower.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.z_lower.BCPressure.alltime.Value = 0.0

crater2D_vangtable_spline.Patch.x_upper.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.x_upper.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.x_upper.BCPressure.alltime.Value = 0.0

crater2D_vangtable_spline.Patch.y_upper.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.y_upper.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.y_upper.BCPressure.alltime.Value = 0.0

crater2D_vangtable_spline.Patch.z_upper.BCPressure.Type = 'FluxConst'
crater2D_vangtable_spline.Patch.z_upper.BCPressure.Cycle = 'constant'
crater2D_vangtable_spline.Patch.z_upper.BCPressure.alltime.Value = 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

crater2D_vangtable_spline.TopoSlopesX.Type = 'Constant'
crater2D_vangtable_spline.TopoSlopesX.GeomNames = 'domain'

crater2D_vangtable_spline.TopoSlopesX.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

crater2D_vangtable_spline.TopoSlopesY.Type = 'Constant'
crater2D_vangtable_spline.TopoSlopesY.GeomNames = 'domain'

crater2D_vangtable_spline.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

crater2D_vangtable_spline.Mannings.Type = 'Constant'
crater2D_vangtable_spline.Mannings.GeomNames = 'domain'
crater2D_vangtable_spline.Mannings.Geom.domain.Value = 0.

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

crater2D_vangtable_spline.ICPressure.Type = 'HydroStaticPatch'
crater2D_vangtable_spline.ICPressure.GeomNames = 'domain'

crater2D_vangtable_spline.Geom.domain.ICPressure.Value = 1.0
crater2D_vangtable_spline.Geom.domain.ICPressure.RefPatch = 'z_lower'
crater2D_vangtable_spline.Geom.domain.ICPressure.RefGeom = 'domain'

crater2D_vangtable_spline.Geom.infiltration.ICPressure.Value = 10.0
crater2D_vangtable_spline.Geom.infiltration.ICPressure.RefPatch = 'infiltration'
crater2D_vangtable_spline.Geom.infiltration.ICPressure.RefGeom = 'domain'

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.PhaseSources.water.Type = 'Constant'
crater2D_vangtable_spline.PhaseSources.water.GeomNames = 'background'
crater2D_vangtable_spline.PhaseSources.water.Geom.background.Value = 0.0


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
crater2D_vangtable_spline.Solver = 'Richards'
crater2D_vangtable_spline.Solver.MaxIter = 10000

crater2D_vangtable_spline.Solver.Nonlinear.MaxIter = 15
crater2D_vangtable_spline.Solver.Nonlinear.ResidualTol = 1e-9
crater2D_vangtable_spline.Solver.Nonlinear.StepTol = 1e-9
crater2D_vangtable_spline.Solver.Nonlinear.EtaValue = 1e-5
crater2D_vangtable_spline.Solver.Nonlinear.UseJacobian = True
crater2D_vangtable_spline.Solver.Nonlinear.DerivativeEpsilon = 1e-7

crater2D_vangtable_spline.Solver.Linear.KrylovDimension = 25
crater2D_vangtable_spline.Solver.Linear.MaxRestarts = 10

crater2D_vangtable_spline.Solver.Linear.Preconditioner = 'MGSemi'
crater2D_vangtable_spline.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
crater2D_vangtable_spline.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 100

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

crater2D_vangtable_spline.run()
