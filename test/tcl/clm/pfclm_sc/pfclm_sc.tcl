## run 24 hour single column CLM test case
##
# Import the ParFlow package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
pfset FileVersion  4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

pfset Process.Topology.P 1
pfset Process.Topology.Q 1
pfset Process.Topology.R 1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X 0.0
pfset ComputationalGrid.Lower.Y 0.0
pfset ComputationalGrid.Lower.Z 0.0

pfset ComputationalGrid.DX      2.0
pfset ComputationalGrid.DY      2.0
pfset ComputationalGrid.DZ      0.1

pfset ComputationalGrid.NX      1
pfset ComputationalGrid.NY      1
pfset ComputationalGrid.NZ      20

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input"

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType "Box"
pfset GeomInput.domain_input.GeomName  "domain"

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X 0.0
pfset Geom.domain.Lower.Y 0.0
pfset Geom.domain.Lower.Z 0.0

pfset Geom.domain.Upper.X 2.0
pfset Geom.domain.Upper.Y 2.0
pfset Geom.domain.Upper.Z 2.0

pfset Geom.domain.Patches "x_lower x_upper y_lower y_upper z_lower z_upper"


#--------------------------------------------
# variable dz assignments
#------------------------------------------

pfset Solver.Nonlinear.VariableDz True
pfset dzScale.GeomNames           "domain"
pfset dzScale.Type                "nzList"
pfset dzScale.nzListNumber        20

# cells start at the bottom (0) and moves up to the top
# domain is 3.21 m thick, root zone is down to 19 cells 
# so the root zone is 2.21 m thick
#first cell is 10*0.1 1m thick
pfset Cell.0.dzScale.Value  10.0   
# next cell is 5*0.1 50 cm thick
pfset Cell.1.dzScale.Value  5.0    
pfset Cell.2.dzScale.Value  1.0   
pfset Cell.3.dzScale.Value  1.0
pfset Cell.4.dzScale.Value  1.0
pfset Cell.5.dzScale.Value  1.0
pfset Cell.6.dzScale.Value  1.0
pfset Cell.7.dzScale.Value  1.0
pfset Cell.8.dzScale.Value  1.0
pfset Cell.9.dzScale.Value  1.0
pfset Cell.10.dzScale.Value 1.0
pfset Cell.11.dzScale.Value 1.0
pfset Cell.12.dzScale.Value 1.0
pfset Cell.13.dzScale.Value 1.0
pfset Cell.14.dzScale.Value 1.0
pfset Cell.15.dzScale.Value 1.0
pfset Cell.16.dzScale.Value 1.0
pfset Cell.17.dzScale.Value 1.0
pfset Cell.18.dzScale.Value 1.0
#0.1* 0.1 = 0.01  1 cm top layer
pfset Cell.19.dzScale.Value 0.1   

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names              "domain"
pfset Geom.domain.Perm.Type        "Constant"
pfset Geom.domain.Perm.Value       0.001465
pfset Geom.domain.Perm.Value       0.1465

pfset Perm.TensorType              "TensorByGeom"
pfset Geom.Perm.TensorByGeom.Names "domain"
pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type              "Constant"
pfset SpecificStorage.GeomNames         "domain"
pfset Geom.domain.SpecificStorage.Value 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type     "Constant"
pfset Phase.water.Density.Value    1.0

pfset Phase.water.Viscosity.Type   "Constant"
pfset Phase.water.Viscosity.Value  1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names ""


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit     1.0
pfset TimingInfo.StartCount   0
pfset TimingInfo.StartTime    0.0
pfset TimingInfo.StopTime     24.0
pfset TimingInfo.DumpInterval 1.0
pfset TimeStep.Type           "Constant"
pfset TimeStep.Value          1.0


#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames    "domain"

pfset Geom.domain.Porosity.Type  "Constant"
pfset Geom.domain.Porosity.Value 0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName "domain"

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type  "Constant"
pfset Phase.water.Mobility.Value 1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type        "VanGenuchten"
pfset Phase.RelPerm.GeomNames   "domain"

pfset Geom.domain.RelPerm.Alpha 2.0
pfset Geom.domain.RelPerm.N     2.0

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type        "VanGenuchten"
pfset Phase.Saturation.GeomNames   "domain"

pfset Geom.domain.Saturation.Alpha 2.0
pfset Geom.domain.Saturation.N     3.0
pfset Geom.domain.Saturation.SRes  0.2
pfset Geom.domain.Saturation.SSat  1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names ""


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names "constant"
pfset Cycle.constant.Names "alltime"
pfset Cycle.constant.alltime.Length 1
pfset Cycle.constant.Repeat -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames "x_lower x_upper y_lower y_upper z_lower z_upper"

pfset Patch.x_lower.BCPressure.Type           "FluxConst"
pfset Patch.x_lower.BCPressure.Cycle          "constant"
pfset Patch.x_lower.BCPressure.alltime.Value 0.0

pfset Patch.y_lower.BCPressure.Type           "FluxConst"
pfset Patch.y_lower.BCPressure.Cycle          "constant"
pfset Patch.y_lower.BCPressure.alltime.Value 0.0

#PFCLM_SC.Patch.z_lower.BCPressure.Type  "FluxConst "
pfset Patch.z_lower.BCPressure.Type           "DirEquilRefPatch"
pfset Patch.z_lower.BCPressure.RefGeom        "domain"
pfset Patch.z_lower.BCPressure.RefPatch       "z_lower"
pfset Patch.z_lower.BCPressure.Cycle          "constant"
pfset Patch.z_lower.BCPressure.alltime.Value 0.0

pfset Patch.x_upper.BCPressure.Type           "FluxConst"
pfset Patch.x_upper.BCPressure.Cycle          "constant"
pfset Patch.x_upper.BCPressure.alltime.Value 0.0

pfset Patch.y_upper.BCPressure.Type           "FluxConst"
pfset Patch.y_upper.BCPressure.Cycle          "constant"
pfset Patch.y_upper.BCPressure.alltime.Value 0.0

pfset Patch.z_upper.BCPressure.Type           "OverlandFlow"
pfset Patch.z_upper.BCPressure.Cycle          "constant"
pfset Patch.z_upper.BCPressure.alltime.Value 0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

pfset TopoSlopesX.Type               "Constant"
pfset TopoSlopesX.GeomNames          "domain"
pfset TopoSlopesX.Geom.domain.Value 0.05

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

pfset TopoSlopesY.Type               "Constant"
pfset TopoSlopesY.GeomNames          "domain"
pfset TopoSlopesY.Geom.domain.Value 0.00

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

pfset Mannings.Type                "Constant"
pfset Mannings.GeomNames           "domain"
pfset Mannings.Geom.domain.Value  2.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type               "Constant"
pfset PhaseSources.water.GeomNames          "domain"
pfset PhaseSources.water.Geom.domain.Value 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution  "NoKnownSolution"

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver          "Richards"
pfset Solver.MaxIter 9000

pfset Solver.Nonlinear.MaxIter           100
pfset Solver.Nonlinear.ResidualTol       1e-5
pfset Solver.Nonlinear.EtaChoice         EtaConstant
pfset Solver.Nonlinear.EtaValue          1e-5
pfset Solver.Nonlinear.UseJacobian       False
pfset Solver.Nonlinear.DerivativeEpsilon 1e-12
pfset Solver.Nonlinear.StepTol           1e-30
pfset Solver.Nonlinear.Globalization      "LineSearch"
pfset Solver.Linear.KrylovDimension      100
pfset Solver.Linear.MaxRestarts          5
pfset Solver.Linear.Preconditioner        "PFMG"
pfset Solver.PrintSubsurf                False
pfset Solver.Drop                        1E-20
pfset Solver.AbsTol                      1E-9

#Writing output options for ParFlow
#  PFB  no SILO
pfset Solver.PrintSubsurfData         False
pfset Solver.PrintPressure            True
pfset Solver.PrintSaturation          True
pfset Solver.PrintCLM                 True
pfset Solver.PrintMask                True
pfset Solver.PrintSpecificStorage     True
pfset Solver.PrintEvapTrans           True

pfset Solver.WriteSiloMannings        False
pfset Solver.WriteSiloMask            False
pfset Solver.WriteSiloSlopes          False
pfset Solver.WriteSiloSaturation      False



#---------------------------------------------------
# LSM / CLM options
#---------------------------------------------------

# set LSM options to CLM
pfset Solver.LSM               "CLM"
# specify type of forcing, file name and location
pfset Solver.CLM.MetForcing    "1D"
pfset Solver.CLM.MetFileName   "forcing_singleColumn_3days_CONUS2.prn"
pfset Solver.CLM.MetFilePath   "./"

# Set CLM Plant Water Use Parameters
pfset Solver.CLM.EvapBeta        "Linear"
pfset Solver.CLM.VegWaterStress  "Saturation"
pfset Solver.CLM.ResSat         0.3
pfset Solver.CLM.WiltingPoint   0.3
pfset Solver.CLM.FieldCapacity  1.00
pfset Solver.CLM.IrrigationType  "none"
pfset Solver.CLM.RootZoneNZ      19
pfset Solver.CLM.SoiLayer        15

#Writing output options for CLM
#  no SILO, no native CLM logs
pfset Solver.PrintLSMSink        False
pfset Solver.CLM.CLMDumpInterval 1
pfset Solver.CLM.CLMFileDir      "output/"
pfset Solver.CLM.BinaryOutDir    False
pfset Solver.CLM.IstepStart      1
pfset Solver.WriteCLMBinary      False
pfset Solver.WriteSiloCLM        False
pfset Solver.CLM.WriteLogs       False
pfset Solver.CLM.WriteLastRST    True
pfset Solver.CLM.DailyRST        False
pfset Solver.CLM.SingleFile      True


#---------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------

pfset ICPressure.Type                 "HydroStaticPatch"
pfset ICPressure.GeomNames            "domain"
pfset Geom.domain.ICPressure.Value    2.0
pfset Geom.domain.ICPressure.RefGeom  "domain"
pfset Geom.domain.ICPressure.RefPatch "z_lower"

#-----------------------------------------------------------------------------
# Run ParFlow 
#-----------------------------------------------------------------------------

pfrun pfclm_sc
pfundist pfclm_sc

#
# Tests
#
source ../../pftest.tcl
set passed 1

set correct_output_dir "../../../correct_output/clm_output"

for {set i 0} { $i <= 24 } {incr i} {
    set i_string [format "%05d" $i]
    if ![pftestFile pfclm_sc.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits $correct_output_dir] {
    set passed 0
    }
    if ![pftestFile pfclm_sc.out.satur.$i_string.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
    set passed 0
    }
}

if $passed {
    puts "pfclm_sc : PASSED"
} {
    puts "pfclm_sc : FAILED"
}
