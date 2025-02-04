# this runs the CLM test case
# with RZ water stress distributed over the RZ
# as a function of moisture limitation as discussed in
# Ferguson, Jefferson, et al ESS 2016
#
# this also represents some CLM best practices from the experience
# of the Maxwell group -- limited output, no CLM logs, solver settings
# to maximize runtime especially on large, parallel runs
# @R Maxwell 24-Nov-27

#
# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin 
package require parflow
namespace import Parflow::*


#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
pfset FileVersion 4

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------

pfset Process.Topology.P        [lindex $argv 0]
pfset Process.Topology.Q        [lindex $argv 1]
pfset Process.Topology.R        [lindex $argv 2]

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------
pfset ComputationalGrid.Lower.X                0.0
pfset ComputationalGrid.Lower.Y                0.0
pfset ComputationalGrid.Lower.Z                 0.0

pfset ComputationalGrid.DX	               1000.
pfset ComputationalGrid.DY                     1000. 
pfset ComputationalGrid.DZ	                 0.5

pfset ComputationalGrid.NX                      5
pfset ComputationalGrid.NY                      5
pfset ComputationalGrid.NZ                     10 

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input"


#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType            Box
pfset GeomInput.domain_input.GeomName             domain

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------
pfset Geom.domain.Lower.X                        0.0 
pfset Geom.domain.Lower.Y                        0.0
pfset Geom.domain.Lower.Z                          0.0

pfset Geom.domain.Upper.X                        5000.
pfset Geom.domain.Upper.Y                        5000.
pfset Geom.domain.Upper.Z                       5. 

pfset Geom.domain.Patches  "x-lower x-upper y-lower y-upper z-lower z-upper"

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------
pfset Geom.Perm.Names "domain"

pfset Geom.domain.Perm.Type            Constant
pfset Geom.domain.Perm.Value           0.2


pfset Perm.TensorType               TensorByGeom

pfset Geom.Perm.TensorByGeom.Names  "domain"

pfset Geom.domain.Perm.TensorValX  1.0
pfset Geom.domain.Perm.TensorValY  1.0
pfset Geom.domain.Perm.TensorValZ  1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type            Constant
pfset SpecificStorage.GeomNames       "domain"
pfset Geom.domain.SpecificStorage.Value 1.0e-6

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names "water"

pfset Phase.water.Density.Type	Constant
pfset Phase.water.Density.Value	1.0

pfset Phase.water.Viscosity.Type	Constant
pfset Phase.water.Viscosity.Value	1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names			""


#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
 
pfset TimingInfo.BaseUnit        1.0
pfset TimingInfo.StartCount      0
pfset TimingInfo.StartTime       0.0
pfset TimingInfo.StopTime        5
pfset TimingInfo.DumpInterval    -1
pfset TimeStep.Type              Constant
pfset TimeStep.Value             1.0
 

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames          domain

pfset Geom.domain.Porosity.Type    Constant
pfset Geom.domain.Porosity.Value   0.390

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
pfset Domain.GeomName domain

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type        Constant
pfset Phase.water.Mobility.Value       1.0

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------
 
pfset Phase.RelPerm.Type               VanGenuchten
pfset Phase.RelPerm.GeomNames          "domain"
 
pfset Geom.domain.RelPerm.Alpha         3.5
pfset Geom.domain.RelPerm.N             2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type              VanGenuchten 
pfset Phase.Saturation.GeomNames         "domain"
 
pfset Geom.domain.Saturation.Alpha        3.5
pfset Geom.domain.Saturation.N            2.
pfset Geom.domain.Saturation.SRes         0.01
pfset Geom.domain.Saturation.SSat         1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names ""


#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
pfset Cycle.Names constant
pfset Cycle.constant.Names		"alltime"
pfset Cycle.constant.alltime.Length	 1
pfset Cycle.constant.Repeat		-1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]
 
pfset Patch.x-lower.BCPressure.Type                   FluxConst
pfset Patch.x-lower.BCPressure.Cycle                  "constant"
pfset Patch.x-lower.BCPressure.alltime.Value          0.0
 
pfset Patch.y-lower.BCPressure.Type                   FluxConst
pfset Patch.y-lower.BCPressure.Cycle                  "constant"
pfset Patch.y-lower.BCPressure.alltime.Value          0.0
 
pfset Patch.z-lower.BCPressure.Type                   FluxConst
pfset Patch.z-lower.BCPressure.Cycle                  "constant"
pfset Patch.z-lower.BCPressure.alltime.Value          0.0
 
pfset Patch.x-upper.BCPressure.Type                   FluxConst
pfset Patch.x-upper.BCPressure.Cycle                  "constant"
pfset Patch.x-upper.BCPressure.alltime.Value          0.0
 
pfset Patch.y-upper.BCPressure.Type                   FluxConst
pfset Patch.y-upper.BCPressure.Cycle                  "constant"
pfset Patch.y-upper.BCPressure.alltime.Value          0.0
 
pfset Patch.z-upper.BCPressure.Type                   OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle                  "constant"
pfset Patch.z-upper.BCPressure.alltime.Value          0.0

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
 
pfset TopoSlopesX.Type "Constant"
pfset TopoSlopesX.GeomNames "domain"
pfset TopoSlopesX.Geom.domain.Value -0.001
 
#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
 
pfset TopoSlopesY.Type "Constant"
pfset TopoSlopesY.GeomNames "domain"
pfset TopoSlopesY.Geom.domain.Value 0.001
 
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
 
pfset KnownSolution                                      NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------
 
pfset Solver                                             Richards
# Max iter limits total timesteps, this is important as PF-CLM will not run
# past this number of steps even if end time set longer
pfset Solver.MaxIter                                     500
 
pfset Solver.Nonlinear.MaxIter                           15
pfset Solver.Nonlinear.ResidualTol                       1e-9
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.01
pfset Solver.Nonlinear.UseJacobian                       True 
pfset Solver.Nonlinear.StepTol                           1e-20
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      15
pfset Solver.Linear.MaxRestart                           2
 
pfset Solver.Linear.Preconditioner                       PFMG 
pfset Solver.PrintSubsurf                                False
pfset Solver.Drop                                        1E-20
pfset Solver.AbsTol                                      1E-9
 
# This key turns on CLM LSM
pfset Solver.LSM                                         CLM

pfset Solver.CLM.MetForcing                              1D
pfset Solver.CLM.MetFileName                             narr_1hr.sc3.txt.0
pfset Solver.CLM.MetFilePath                             ./

#  We are NOT writing CLM files as SILO but setting this to True 
#  will write both SILO and PFB output for CLM (in a single file as
#  specified below)
pfset Solver.WriteSiloCLM                                False
pfset Solver.WriteSiloEvapTrans                          False 
pfset Solver.WriteSiloOverlandBCFlux                     False
#  We are writing CLM files as PFB
pfset Solver.PrintCLM                                    True

#Limit native CLM output and logs
pfset Solver.CLM.Print1dOut                           False
pfset Solver.BinaryOutDir                             False
pfset Solver.WriteCLMBinary                           False
pfset Solver.CLM.CLMDumpInterval                      1
pfset Solver.CLM.WriteLogs                          False 


# Set evaporation Beta (resistance) function to Linear 
pfset Solver.CLM.EvapBeta                             Linear
# Set plant water stress to be a function of Saturation
pfset Solver.CLM.VegWaterStress                       Saturation
# Set residual Sat for soil moisture resistance
pfset Solver.CLM.ResSat                               0.2
# Set wilting point limit and field capacity (values are for Saturation, not pressure) 
pfset Solver.CLM.WiltingPoint                         0.2
pfset Solver.CLM.FieldCapacity                        1.00
## this key sets the option described in Ferguson, Jefferson, et al ESS 2016
# a setting of 0 (default) will use standard water stress distribution
pfset Solver.CLM.RZWaterStress                           1
# No irrigation
pfset Solver.CLM.IrrigationType                       none


## writing only last daily restarts.  This will be at Midnight GMT and 
## starts at timestep 18, then intervals of 24 thereafter
pfset Solver.CLM.WriteLastRST                       True
pfset Solver.CLM.DailyRST                           True
# we write a single CLM file for all output at each timestep (one file / timestep
# for all 17 CLM output variables) as described in PF manual
pfset Solver.CLM.SingleFile                         True



# Initial conditions: water pressure
#---------------------------------------------------------
 
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -2.0
 
pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   z-upper



#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


pfrun clm_rz_water_stress 
pfundist clm_rz_water_stress

#
# Tests 
#
source ../pftest.tcl
set passed 1

set correct_output_dir "../../correct_output/clm_output"

# we compare pressure, saturation and CLM output

for {set i 0} { $i <= 5 } {incr i} {
    set i_string [format "%05d" $i]
    if ![pftestFile clm_rz_water_stress.out.press.$i_string.pfb "Max difference in Pressure for timestep $i_string" $sig_digits $correct_output_dir] {
	set passed 0
    }
    if ![pftestFile clm_rz_water_stress.out.satur.$i_string.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
	set passed 0
    }
    if {$i > 0} {
	if ![pftestFile clm_rz_water_stress.out.clm_output.$i_string.C.pfb "Max difference in Saturation for timestep $i_string" $sig_digits $correct_output_dir] {
	    set passed 0
	}
    }

}



if $passed {
    puts "clm_rz_stress : PASSED"
} {
    puts "clm_rz_stress : FAILED"
}

