#  This runs the infiltration experiment using the GroundwaterFlowBC.

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
pfset ComputationalGrid.Lower.X   0.0
pfset ComputationalGrid.Lower.Y   0.0
pfset ComputationalGrid.Lower.Z   0.0

pfset ComputationalGrid.DX       10.0
pfset ComputationalGrid.DY       10.0
pfset ComputationalGrid.DZ       0.05

pfset ComputationalGrid.NX          5
pfset ComputationalGrid.NY          5
pfset ComputationalGrid.NZ        200

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names "domain_input background_input"

#---------------------------------------------------------
# Domain Geometry Input
#---------------------------------------------------------
pfset GeomInput.domain_input.InputType    Box
pfset GeomInput.domain_input.GeomName     domain

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
pfset Geom.domain.Lower.X         0.0 
pfset Geom.domain.Lower.Y         0.0
pfset Geom.domain.Lower.Z         0.0

pfset Geom.domain.Upper.X        50.0
pfset Geom.domain.Upper.Y        50.0
pfset Geom.domain.Upper.Z        10.0

pfset Geom.domain.Patches "left right front back bottom top"

#---------------------------------------------------------
# Background Geometry Input
#---------------------------------------------------------
pfset GeomInput.background_input.InputType         Box
pfset GeomInput.background_input.GeomName          background

#---------------------------------------------------------
# Background Geometry
#---------------------------------------------------------
pfset Geom.background.Lower.X -99999999.0
pfset Geom.background.Lower.Y -99999999.0
pfset Geom.background.Lower.Z -99999999.0

pfset Geom.background.Upper.X  99999999.0
pfset Geom.background.Upper.Y  99999999.0
pfset Geom.background.Upper.Z  99999999.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
pfset TimingInfo.BaseUnit       0.25
pfset TimingInfo.StartCount     0.0
pfset TimingInfo.StartTime      0.0
pfset TimingInfo.StopTime       120.0
pfset TimingInfo.DumpInterval  -5
pfset TimeStep.Type             Constant
pfset TimeStep.Value            0.25

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names               "constant rainfall"

pfset Cycle.constant.Names           "alltime"
pfset Cycle.constant.alltime.Length   1
pfset Cycle.constant.Repeat          -1

pfset Cycle.rainfall.Names       "On Off"
pfset Cycle.rainfall.On.Length   40
pfset Cycle.rainfall.Off.Length  440
pfset Cycle.rainfall.Repeat     -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames    [pfget Geom.domain.Patches]

pfset Patch.left.BCPressure.Type             FluxConst
pfset Patch.left.BCPressure.Cycle            constant
pfset Patch.left.BCPressure.alltime.Value    0.0

pfset Patch.right.BCPressure.Type            FluxConst
pfset Patch.right.BCPressure.Cycle           constant
pfset Patch.right.BCPressure.alltime.Value   0.0

pfset Patch.front.BCPressure.Type            FluxConst
pfset Patch.front.BCPressure.Cycle           constant
pfset Patch.front.BCPressure.alltime.Value   0.0

pfset Patch.back.BCPressure.Type             FluxConst
pfset Patch.back.BCPressure.Cycle            constant
pfset Patch.back.BCPressure.alltime.Value    0.0

pfset Patch.bottom.BCPressure.Type   GroundwaterFlow
pfset Patch.bottom.BCPressure.Cycle  constant
pfset Patch.bottom.BCPressure.GroundwaterFlow.SpecificYield.Type   Constant
pfset Patch.bottom.BCPressure.GroundwaterFlow.SpecificYield.Value  0.1
pfset Patch.bottom.BCPressure.GroundwaterFlow.AquiferDepth.Type    Constant
pfset Patch.bottom.BCPressure.GroundwaterFlow.AquiferDepth.Value   90.0

pfset Patch.top.BCPressure.Type              FluxConst
pfset Patch.top.BCPressure.Cycle             rainfall
pfset Patch.top.BCPressure.On.Value         -0.005
pfset Patch.top.BCPressure.Off.Value         0.001

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type                        HydroStaticPatch
pfset ICPressure.GeomNames                   domain
pfset Geom.domain.ICPressure.Value           -1.25
pfset Geom.domain.ICPressure.RefGeom         domain
pfset Geom.domain.ICPressure.RefPatch        top

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity  1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------
pfset Geom.Porosity.GeomNames       "domain"
pfset Geom.domain.Porosity.Type     Constant
# Value for Silt soil
pfset Geom.domain.Porosity.Value    0.489
# Value for Clay soil
# pfset Geom.domain.Porosity.Value    0.459

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names               "domain background"
pfset Geom.domain.Perm.Type         Constant
# Value for Silt soil in m/hour
pfset Geom.domain.Perm.Value        0.01836
# Value for Clay soil in m/hour
# pfset Geom.domain.Perm.Value        0.000612

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain background"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

pfset Geom.background.Perm.Type     Constant
pfset Geom.background.Perm.Value    1.0

pfset Geom.background.Perm.TensorValX  1.0
pfset Geom.background.Perm.TensorValY  1.0
pfset Geom.background.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names "water"

pfset Phase.water.Density.Type      Constant
pfset Phase.water.Density.Value     1.0

pfset Phase.water.Viscosity.Type    Constant
pfset Phase.water.Viscosity.Value   1.0

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                Constant
pfset PhaseSources.water.GeomNames           domain
pfset PhaseSources.water.Geom.domain.Value   0.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------
pfset Phase.Saturation.Type            VanGenuchten
pfset Phase.Saturation.GeomNames       domain
pfset Geom.domain.Saturation.Alpha     0.657658
pfset Geom.domain.Saturation.N         2.678804
pfset Geom.domain.Saturation.SRes      0.102249
pfset Geom.domain.Saturation.SSat      1.0


#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain
pfset Geom.domain.RelPerm.Alpha        0.657658
pfset Geom.domain.RelPerm.N            2.678804

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------
pfset Mannings.Type               "Constant"
pfset Mannings.GeomNames          "domain"
pfset Mannings.Geom.domain.Value  5.5e-5

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type               Constant
pfset SpecificStorage.GeomNames          "domain"
pfset Geom.domain.SpecificStorage.Value  1.0e-4
 
#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
pfset TopoSlopesX.Type               "Constant"
pfset TopoSlopesX.GeomNames          "domain"
pfset TopoSlopesX.Geom.domain.Value  0.0
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
pfset TopoSlopesY.Type               "Constant"
pfset TopoSlopesY.GeomNames          "domain"
pfset TopoSlopesY.Geom.domain.Value  0.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names          ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------
pfset Geom.Retardation.GeomNames  ""

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                 ""

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
pfset Solver                                             Richards
pfset Solver.MaxIter                                     100000

pfset Solver.Nonlinear.MaxIter                           1000
pfset Solver.Nonlinear.ResidualTol                       1e-4
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          1e-4
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-5
pfset Solver.Nonlinear.StepTol                           1e-5

pfset Solver.Linear.KrylovDimension                      30

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10

pfset Solver.PrintPressure     False
pfset Solver.PrintSubsurfData  False
pfset Solver.PrintSaturation   True
pfset Solver.PrintMask         False

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------
# pfwritedb infiltration_gfb
pfrun infiltration_gfb
