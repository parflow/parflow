#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*

pfset FileVersion 4


pfset Process.Topology.P 1
pfset Process.Topology.Q 4
pfset Process.Topology.R 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------
set nn 600
set mm 150
pfset ComputationalGrid.Lower.X           0.0
pfset ComputationalGrid.Lower.Y           0.0
pfset ComputationalGrid.Lower.Z           0.0

pfset ComputationalGrid.NX            [expr [pfget Process.Topology.P] * $nn] 
pfset ComputationalGrid.NY            [expr [pfget Process.Topology.Q] * $mm] 
pfset ComputationalGrid.NZ            240

pfset ComputationalGrid.DX	         1.0
pfset ComputationalGrid.DY               1.0
pfset ComputationalGrid.DZ	         0.025

set nx [pfget ComputationalGrid.NX]
set dx [pfget ComputationalGrid.DX]
set ny [pfget ComputationalGrid.NY]
set dy [pfget ComputationalGrid.DY]
set nz [pfget ComputationalGrid.NZ]
set dz [pfget ComputationalGrid.DZ]

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
pfset GeomInput.Names                 "domain_input"

pfset GeomInput.domain_input.InputType  Box
pfset GeomInput.domain_input.GeomName  domain

pfset Geom.domain.Patches "left right front back bottom top"

pfset Geom.domain.Lower.X                         0.0
pfset Geom.domain.Lower.Y                         0.0
pfset Geom.domain.Lower.Z                         0.0

pfset Geom.domain.Upper.X                         [expr ($nx * $dx)]
pfset Geom.domain.Upper.Y                         [expr ($ny * $dy)]
pfset Geom.domain.Upper.Z                         [expr ($nz * $dz)]

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names                 domain

# Values in m/hour

pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           1.0e-3

#-----------------------------------------------------------------------------
# Perm Tensors
#-----------------------------------------------------------------------------

pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0d0
pfset Geom.domain.Perm.TensorValY  1.0d0
pfset Geom.domain.Perm.TensorValZ  1.0d0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value 1.0e-8

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	        Constant
pfset Phase.water.Density.Value	        1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

pfset Contaminants.Names			""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

pfset Geom.Retardation.GeomNames           ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
set time 1.
set fac 3.0
pfset TimingInfo.BaseUnit               1.0
pfset TimingInfo.StartCount             0
pfset TimingInfo.StartTime              0.0
pfset TimingInfo.StopTime               [expr ($time * $fac)]
pfset TimingInfo.DumpInterval           [expr ($time * $fac)]
pfset TimeStep.Type                     Constant
pfset TimeStep.Value                    $time

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames           domain

pfset Geom.domain.Porosity.Type          Constant
pfset Geom.domain.Porosity.Value         0.451

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------


pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          domain

pfset Geom.domain.RelPerm.Alpha         1.0
pfset Geom.domain.RelPerm.N             4. 

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten
pfset Phase.Saturation.GeomNames         domain

pfset Geom.domain.Saturation.Alpha        1.0
pfset Geom.domain.Saturation.N            4.
pfset Geom.domain.Saturation.SRes         0.15
pfset Geom.domain.Saturation.SSat         1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                           ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      1
pfset Cycle.constant.Repeat             -1
 
#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames        "left right front back bottom top"

pfset Patch.left.BCPressure.Type                      FluxConst
pfset Patch.left.BCPressure.Cycle                     "constant"
pfset Patch.left.BCPressure.alltime.Value             0.0

pfset Patch.right.BCPressure.Type                     FluxConst
pfset Patch.right.BCPressure.Cycle                    "constant"
pfset Patch.right.BCPressure.alltime.Value            0.0

pfset Patch.front.BCPressure.Type                     FluxConst
pfset Patch.front.BCPressure.Cycle                    "constant"
pfset Patch.front.BCPressure.alltime.Value            0.0

pfset Patch.back.BCPressure.Type                      FluxConst
pfset Patch.back.BCPressure.Cycle                     "constant"
pfset Patch.back.BCPressure.alltime.Value             0.0

#---- Bottom BC
pfset Patch.bottom.BCPressure.Type                    DirEquilRefPatch
pfset Patch.bottom.BCPressure.RefGeom                 domain
pfset Patch.bottom.BCPressure.RefPatch                bottom
pfset Patch.bottom.BCPressure.Cycle                   "constant"
pfset Patch.bottom.BCPressure.alltime.Value           0.0
#---- End Bottom BC

#---- Top BC
pfset Patch.top.BCPressure.Type                       FluxConst
pfset Patch.top.BCPressure.Cycle                      "constant"
pfset Patch.top.BCPressure.alltime.Value              -0.0008
#---- End Top BC

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "domain"
pfset TopoSlopesX.Geom.domain.Value 0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "domain"
pfset TopoSlopesY.Geom.domain.Value 0.0

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

pfset Mannings.Type "Constant"
pfset Mannings.GeomNames "domain"
pfset Mannings.Geom.domain.Value 5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                         Constant
pfset PhaseSources.water.GeomNames                    domain
pfset PhaseSources.water.Geom.domain.Value        0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                    NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                             Richards
pfset Solver.MaxIter                                     250000

pfset Solver.Nonlinear.MaxIter                           300
pfset Solver.Nonlinear.ResidualTol                       1e-5
pfset Solver.Nonlinear.EtaChoice                         Walker1 
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-16
pfset Solver.Nonlinear.StepTol				 1e-10
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      20
pfset Solver.Linear.MaxRestart                           2

pfset Solver.Linear.Preconditioner                       MGSemi
pfset Solver.Linear.Preconditioner.MGSemi.MaxIter        1
pfset Solver.Linear.Preconditioner.MGSemi.MaxLevels      10
pfset Solver.Drop                                       1E-20
pfset Solver.AbsTol                                     1E-12
 
pfset Solver.PrintSaturation                            False
pfset Solver.PrintSubsurf                               False
pfset Solver.PrintPressure                              False

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -3.0

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   bottom

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfwritedb clayL
#pfrun clayL


