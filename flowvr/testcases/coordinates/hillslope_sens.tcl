#Using parflow_BH


# Import the ParFlow TCL package
#
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

#-----------------------------------------------------------------------------
# File input version number
#-----------------------------------------------------------------------------
pfset FileVersion 4

#---------------------------------------------------------
# FlowVR parameters
#---------------------------------------------------------
pfset FlowVR.SteerLogMode "VerySimple"
#pfset FlowVR.SteerLogMode "Full"


pfset FlowVR.OnEnd    SendEmpty

pfset FlowVR.Outports.Names "pressure"

pfset FlowVR.Outports.pressure.Periodicity 1
pfset FlowVR.Outports.pressure.Variable  "pressure"
pfset FlowVR.Outports.pressure.Offset 0

#-----------------------------------------------------------------------------
# Process Topology
#-----------------------------------------------------------------------------
pfset Process.Topology.P        1
pfset Process.Topology.Q        1
pfset Process.Topology.R        1

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------

pfset ComputationalGrid.Lower.X                 0.0
pfset ComputationalGrid.Lower.Y                	0.0
pfset ComputationalGrid.Lower.Z                	0.0

pfset ComputationalGrid.DX	               	10.0
pfset ComputationalGrid.DY	                10.0
pfset ComputationalGrid.DZ	               	1.0

pfset ComputationalGrid.NX	     	       	1
pfset ComputationalGrid.NY	     		50
pfset ComputationalGrid.NZ                  	24


#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------
pfset GeomInput.Names "domain_input H1_input H2_input H3_input H4_input H5_input"

#-----------------------------------------------------------------------------
# Domain Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.domain_input.InputType           Box
pfset GeomInput.domain_input.GeomName            domain

pfset Geom.domain.Lower.X                        0
pfset Geom.domain.Lower.Y                        0
pfset Geom.domain.Lower.Z                        0

pfset Geom.domain.Upper.X                        10
pfset Geom.domain.Upper.Y                        500
pfset Geom.domain.Upper.Z                        24

pfset Geom.domain.Patches                        "x-lower x-upper y-lower y-upper z-lower z-upper"
#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------
#pfset Domain.GeomName                            "domain"
pfset Domain.GeomName                            domain

#-----------------------------------------------------------------------------
# H1 Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.H1_input.InputType               Box
pfset GeomInput.H1_input.GeomName                H1

pfset Geom.H1.Lower.X                            0
pfset Geom.H1.Lower.Y                            0
#pfset Geom.H1.Lower.Z                            23.96
pfset Geom.H1.Lower.Z                            20

pfset Geom.H1.Upper.X                            10
pfset Geom.H1.Upper.Y                            500
pfset Geom.H1.Upper.Z                            24

#-----------------------------------------------------------------------------
# H2 Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.H2_input.InputType               Box
pfset GeomInput.H2_input.GeomName                H2

pfset Geom.H2.Lower.X                            0
pfset Geom.H2.Lower.Y                            0
#pfset Geom.H2.Lower.Z                            23.5
pfset Geom.H2.Lower.Z                            15

pfset Geom.H2.Upper.X                            10
pfset Geom.H2.Upper.Y                            500
#pfset Geom.H2.Upper.Z                            23.96
pfset Geom.H2.Upper.Z                            20

#-----------------------------------------------------------------------------
# H3 Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.H3_input.InputType               Box
pfset GeomInput.H3_input.GeomName                H3

pfset Geom.H3.Lower.X                            0
pfset Geom.H3.Lower.Y                            0
#pfset Geom.H3.Lower.Z                            22.7
pfset Geom.H3.Lower.Z                            10

pfset Geom.H3.Upper.X                            10
pfset Geom.H3.Upper.Y                            500
#pfset Geom.H3.Upper.Z                            23.5
pfset Geom.H3.Upper.Z                            15

#-----------------------------------------------------------------------------
# H4 Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.H4_input.InputType               Box
pfset GeomInput.H4_input.GeomName                H4

pfset Geom.H4.Lower.X                            0
pfset Geom.H4.Lower.Y                            0
#pfset Geom.H4.Lower.Z                            21.5
pfset Geom.H4.Lower.Z                            5

pfset Geom.H4.Upper.X                            10
pfset Geom.H4.Upper.Y                            500
#pfset Geom.H4.Upper.Z                            22.7
pfset Geom.H4.Upper.Z                            10

#-----------------------------------------------------------------------------
# H5 Geometry Input
#-----------------------------------------------------------------------------
pfset GeomInput.H5_input.InputType               Box
pfset GeomInput.H5_input.GeomName                H5

pfset Geom.H5.Lower.X                            0
pfset Geom.H5.Lower.Y                            0
pfset Geom.H5.Lower.Z                            0

pfset Geom.H5.Upper.X                            10
pfset Geom.H5.Upper.Y                            500
#pfset Geom.H5.Upper.Z                            21.5
pfset Geom.H5.Upper.Z                            5

#-----------------------------------------------------------------------------
# Perm = Ksat (m/min)
#-----------------------------------------------------------------------------

#pfset Geom.Perm.Names                           "domain H1"
pfset Geom.Perm.Names                           "H1 H2 H3 H4 H5"
#pfset Geom.domain.Perm.Type                     Constant
#pfset Geom.domain.Perm.Value                    0.1

pfset Geom.H1.Perm.Type                         Constant
pfset Geom.H1.Perm.Value                        0.0000001667
pfset Geom.H1.Perm.Value                        0.01667

pfset Geom.H2.Perm.Type                         Constant

pfset Geom.H2.Perm.Value                        0.0042
pfset Geom.H3.Perm.Value                        0.10042
pfset Geom.H4.Perm.Value                        0.10042
pfset Geom.H5.Perm.Value                        0.10042

pfset Geom.H3.Perm.Type                         Constant
pfset Geom.H3.Perm.Value                        0.003

pfset Geom.H4.Perm.Type                         Constant
pfset Geom.H4.Perm.Value                        0.0042

pfset Geom.H5.Perm.Type                         Constant
pfset Geom.H5.Perm.Value                        0.0042


pfset Perm.TensorType                           TensorByGeom
pfset Geom.Perm.TensorByGeom.Names              "domain"

pfset Geom.domain.Perm.TensorValX               1.0
pfset Geom.domain.Perm.TensorValY               1.0
pfset Geom.domain.Perm.TensorValZ               1.0


#-----------------------------------------------------------------------------
# Specific Storage (/m)
#-----------------------------------------------------------------------------
pfset SpecificStorage.Type                      Constant
pfset SpecificStorage.GeomNames                 "domain"

pfset Geom.domain.SpecificStorage.Value         1.0e-3

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------
pfset Phase.Names                               "water"

pfset Phase.water.Density.Type	                Constant
pfset Phase.water.Density.Value	                1.0

pfset Phase.water.Viscosity.Type	        Constant
pfset Phase.water.Viscosity.Value	        1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------
pfset Contaminants.Names		        ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------
pfset Gravity				1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------
pfset TimingInfo.BaseUnit        15
pfset TimingInfo.StartCount      0.0
pfset TimingInfo.StartTime       0.0

pfset TimingInfo.DumpInterval    15
pfset TimeStep.Type              Constant
pfset TimeStep.Value             15.0

pfset TimingInfo.StopTime	 30

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

#pfset Geom.Porosity.GeomNames      "domain H1"
pfset Geom.Porosity.GeomNames      "H1 H2 H3 H4 H5"

#pfset Geom.domain.Porosity.Type    	Constant
#pfset Geom.domain.Porosity.Value   	0.1

pfset Geom.H1.Porosity.Type             Constant
#pfset Geom.H1.Porosity.Value   	        0.358
pfset Geom.H1.Porosity.Value   	        0.1
pfset Geom.H2.Porosity.Type             Constant
pfset Geom.H2.Porosity.Value   	        0.358

pfset Geom.H3.Porosity.Type             Constant
pfset Geom.H3.Porosity.Value   	        0.321

pfset Geom.H4.Porosity.Type             Constant
pfset Geom.H4.Porosity.Value   	        0.358

pfset Geom.H5.Porosity.Type             Constant
pfset Geom.H5.Porosity.Value   	        0.34

#-----------------------------------------------------------------------------
# Mobility
#-----------------------------------------------------------------------------
pfset Phase.water.Mobility.Type        Constant
pfset Phase.water.Mobility.Value       1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
pfset Wells.Names                        ""

#-----------------------------------------------------------------------------2005
# Time Cycles
#-----------------------------------------------------------------------------

pfset Cycle.Names                       "constant rainrec"
#pfset Cycle.Names                       "constant"

pfset Cycle.constant.Names              "alltime"
pfset Cycle.constant.alltime.Length      1
pfset Cycle.constant.Repeat              -1

#pfset Cycle.rainrec.Names              "rec rain"
#pfset Cycle.rainrec.rec.Length           2
#pfset Cycle.rainrec.rain.Length          3
#pfset Cycle.rainrec.Repeat               -1


pfset Cycle.rainrec.Names "0"
#0: raintime, 1: drytime
pfset Cycle.rainrec.0.Length              1
pfset Cycle.rainrec.Repeat               -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
pfset BCPressure.PatchNames                   [pfget Geom.domain.Patches]

pfset Patch.x-lower.BCPressure.Type		      FluxConst
pfset Patch.x-lower.BCPressure.Cycle		      "constant"
pfset Patch.x-lower.BCPressure.alltime.Value	      0.0

pfset Patch.y-lower.BCPressure.Type		      FluxConst
pfset Patch.y-lower.BCPressure.Cycle		      "constant"
pfset Patch.y-lower.BCPressure.alltime.Value	      0.0

pfset Patch.z-lower.BCPressure.Type                   FluxConst
pfset Patch.z-lower.BCPressure.Cycle                  "constant"
pfset Patch.z-lower.BCPressure.alltime.Value          0.0

pfset Patch.x-upper.BCPressure.Type		      FluxConst
pfset Patch.x-upper.BCPressure.Cycle		      "constant"
pfset Patch.x-upper.BCPressure.alltime.Value	      0.0

pfset Patch.y-upper.BCPressure.Type		      FluxConst
pfset Patch.y-upper.BCPressure.Cycle		      "constant"
pfset Patch.y-upper.BCPressure.alltime.Value	      0.0

pfset Patch.z-upper.BCPressure.Type                   OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle                  "constant"
pfset Patch.z-upper.BCPressure.alltime.Value          0.0

pfset Patch.z-upper.BCPressure.Cycle                  "rainrec"

## m / minutes during 15 minutes: 10mm / 15mn that's a 40mm/hr, pretty decent
pfset Patch.z-upper.BCPressure.0.Value                 -0.01

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------
pfset TopoSlopesX.Type 				"Constant"
pfset TopoSlopesX.GeomNames                     "domain"
pfset TopoSlopesX.Geom.domain.Value 		0.0

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------
pfset TopoSlopesY.Type 				"Constant"
pfset TopoSlopesY.GeomNames                     "domain"
pfset TopoSlopesY.Geom.domain.Value 		0.05

#---------------------------------------------------------
# Mannings coefficient (min^1/3/m)
#---------------------------------------------------------
pfset Mannings.Type 				"Constant"
pfset Mannings.GeomNames 			"domain"
#pfset Mannings.Geom.domain.Value 		0.0000056
#This Value will be changed in the steering....
#Search modus: linear. (Ks would be log) between 0.00000 0.00009
pfset Mannings.Geom.domain.Value 		0.0000222
#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type               VanGenuchten
#pfset Phase.RelPerm.GeomNames          "domain H1"
pfset Phase.RelPerm.GeomNames          "H1 H2 H3 H4 H5"

#pfset Geom.domain.RelPerm.Alpha            0.005
#pfset Geom.domain.RelPerm.N                2.0

#pfset Geom.H1.RelPerm.Alpha            1.176
#pfset Geom.H1.RelPerm.N                2.75

#pfset Geom.H2.RelPerm.Alpha            1.667
#pfset Geom.H2.RelPerm.N                3.0

#pfset Geom.H3.RelPerm.Alpha            2.5
#pfset Geom.H3.RelPerm.N                3.1

#pfset Geom.H4.RelPerm.Alpha            3.333
#pfset Geom.H4.RelPerm.N                3.0

#pfset Geom.H5.RelPerm.Alpha            5.0
#pfset Geom.H5.RelPerm.N                3.3

pfset Geom.H1.RelPerm.Alpha            1.176
pfset Geom.H1.RelPerm.N                1.75

pfset Geom.H2.RelPerm.Alpha            1.667
pfset Geom.H2.RelPerm.N                2.0

pfset Geom.H3.RelPerm.Alpha            2.5
pfset Geom.H3.RelPerm.N                2.1

pfset Geom.H4.RelPerm.Alpha            3.333
pfset Geom.H4.RelPerm.N                2.0

pfset Geom.H5.RelPerm.Alpha            5.0
pfset Geom.H5.RelPerm.N                2.3

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

pfset Phase.Saturation.Type            VanGenuchten
#pfset Phase.Saturation.GeomNames       "domain H1"
pfset Phase.Saturation.GeomNames       "H1 H2 H3 H4 H5"

#pfset Geom.domain.Saturation.Alpha     0.005
#pfset Geom.domain.Saturation.N         2.0
#pfset Geom.domain.Saturation.SRes      0.2
#pfset Geom.domain.Saturation.SSat      0.9

#pfset Geom.H1.Saturation.Alpha         1.176
#pfset Geom.H1.Saturation.N             2.75
pfset Geom.H1.Saturation.Alpha         1.176
pfset Geom.H1.Saturation.N             1.75
pfset Geom.H1.Saturation.SRes          0.02
pfset Geom.H1.Saturation.SSat          0.9

#pfset Geom.H2.Saturation.Alpha         1.667
#pfset Geom.H2.Saturation.N             3.0
pfset Geom.H2.Saturation.Alpha         1.667
pfset Geom.H2.Saturation.N             2.0
pfset Geom.H2.Saturation.SRes          0.02
pfset Geom.H2.Saturation.SSat          0.9

#pfset Geom.H3.Saturation.Alpha         2.5
#pfset Geom.H3.Saturation.N             3.1
pfset Geom.H3.Saturation.Alpha         2.5
pfset Geom.H3.Saturation.N             2.1
pfset Geom.H3.Saturation.SRes          0.02
pfset Geom.H3.Saturation.SSat          0.9

#pfset Geom.H4.Saturation.Alpha         3.333
#pfset Geom.H4.Saturation.N             3.0
pfset Geom.H4.Saturation.Alpha         3.333
pfset Geom.H4.Saturation.N             2.0
pfset Geom.H4.Saturation.SRes          0.02
pfset Geom.H4.Saturation.SSat          0.9

#pfset Geom.H5.Saturation.Alpha         5.0
#pfset Geom.H5.Saturation.N             3.3
pfset Geom.H5.Saturation.Alpha         5.0
pfset Geom.H5.Saturation.N             2.3
pfset Geom.H5.Saturation.SRes          0.02
pfset Geom.H5.Saturation.SSat          0.9


#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------
pfset PhaseSources.water.Type                         "Constant"
#pfset PhaseSources.water.GeomNames                    "domain"
pfset PhaseSources.water.GeomNames                    domain
pfset PhaseSources.water.Geom.domain.Value            0.0


#----------------------------------------------------------------
# CLM Settings:
# ---------------------------------------------------------------

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------
pfset ICPressure.Type                                 HydroStaticPatch
pfset ICPressure.GeomNames                            "domain"

#pfset Geom.domain.ICPressure.RefGeom                  "domain"
pfset Geom.domain.ICPressure.RefGeom                  domain
pfset Geom.domain.ICPressure.Value                    -15
pfset Geom.domain.ICPressure.RefPatch                 z-upper

#-----------------------------------------------------------------------------
# Set Outputs
#-----------------------------------------------------------------------------
pfset Solver.PrintDZMultiplier				                   False
pfset Solver.PrintOverlandSum				                     False
pfset Solver.PrintSlopes				                         False
pfset Solver.PrintEvapTrans				                       False
pfset Solver.PrintEvapTransSum				                   False
pfset Solver.PrintSubsurfData                            False
pfset Solver.PrintMannings                               False
pfset Solver.PrintPressure                               False
pfset Solver.PrintSaturation                             False
pfset Solver.PrintMask                                   False
pfset Solver.PrintSpecificStorage 		                   False


#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------
pfset KnownSolution                                      NoKnownSolution

#-----------------------------------------------------------------------------
# Variable DZ
#-----------------------------------------------------------------------------
pfset Solver.TerrainFollowingGrid                        True
#pfset Solver.Nonlinear.VariableDz                     False

pfset Solver.Nonlinear.VariableDz                        True

pfset dzScale.GeomNames				        domain

pfset dzScale.GeomNames                                  domain
pfset dzScale.Type                                       nzList

#0 is bottom layer
pfset dzScale.nzListNumber                               24

pfset Cell.0.dzScale.Value                               4.3
pfset Cell.1.dzScale.Value                               4.3
pfset Cell.2.dzScale.Value                               4.3
pfset Cell.3.dzScale.Value                               4.3
pfset Cell.4.dzScale.Value                               4.3
pfset Cell.5.dzScale.Value                               0.24
pfset Cell.6.dzScale.Value                               0.24
pfset Cell.7.dzScale.Value                               0.24
pfset Cell.8.dzScale.Value                               0.24
pfset Cell.9.dzScale.Value                               0.24
pfset Cell.10.dzScale.Value                              0.16
pfset Cell.11.dzScale.Value                              0.16
pfset Cell.12.dzScale.Value                              0.16
pfset Cell.13.dzScale.Value                              0.16
pfset Cell.14.dzScale.Value                              0.16
pfset Cell.15.dzScale.Value                              0.092
pfset Cell.16.dzScale.Value                              0.092
pfset Cell.17.dzScale.Value                              0.092
pfset Cell.18.dzScale.Value                              0.092
pfset Cell.19.dzScale.Value                              0.092
pfset Cell.20.dzScale.Value                              0.01
pfset Cell.21.dzScale.Value                              0.01
pfset Cell.22.dzScale.Value                              0.01
pfset Cell.23.dzScale.Value                              0.01


#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                             Richards

pfset Solver.MaxConvergenceFailures			 6
pfset Solver.MaxIter                                     100000
pfset Solver.AbsTol                                      1E-8
pfset Solver.Drop                                        1E-20
pfset Solver.Nonlinear.MaxIter                           200
pfset Solver.Nonlinear.ResidualTol                       1e-9
#pfset Solver.Nonlinear.ResidualTol                       1e-5
pfset Solver.Nonlinear.StepTol                           1e-30
#pfset Solver.Nonlinear.StepTol                           1e-7
pfset Solver.Nonlinear.EtaChoice                         EtaConstant
pfset Solver.Nonlinear.EtaValue                          0.001
pfset Solver.Nonlinear.UseJacobian                       True
pfset Solver.Nonlinear.DerivativeEpsilon                 1e-8
pfset Solver.Nonlinear.Globalization                     LineSearch
pfset Solver.Linear.KrylovDimension                      200
#pfset Solver.Linear.KrylovDimension                      15
pfset Solver.Linear.MaxRestart                           2
pfset Solver.Linear.Preconditioner                       PFMG

#dis line below new
pfset Solver.Linear.Preconditioner.PCMatrixType     FullJacobian


pfset OverlandFlowSpinUp				 0



#-----------------------------------------------------------------------------
# Set CLM parameters
#-----------------------------------------------------------------------------

#pfset Solver.LSM                                         CLM



#pfset Solver.CLM.MetForcing                              1D
#pfset Solver.CLM.MetFileName                             forcagePF.txt.0
#pfset Solver.CLM.MetFilePath                             ./

pfset Solver.CLM.CLMDumpInterval                         15

#pfset Solver.CLM.ForceVegetation			 True
pfset Solver.CLM.ForceVegetation			 False
#pfset Solver.CLM.RootZoneNZ				 22
#pfset Solver.CLM.BinaryOutDir				 False

#pfset Solver.CLM.SingleFile				 True

#pfset Solver.PrintCLM                                    True
#pfset Solver.WriteCLMBinary				 False

#pfset Solver.CLM.WriteLogs				 True
#pfset Solver.CLM.WriteLastRST				 True

#pfset Solver.CLM.EvapBeta				 "none"
#pfset Solver.CLM.ResSat					 0.1




#-----------------------------------------------------------------------------
# Run and Unload the Parflow output files
#-----------------------------------------------------------------------------
#pfset Solver.WriteSiloSubsurfData True
#pfset Solver.WriteSiloPressure True
pfset NetCDF.WritePressure True
pfset NetCDF.NumStepsPerFile 10

pfwritedb hillslope_sens
