# 2015-11-10_00_KGo
# 2016-09-27_00_LPo
# weak scaling problem
# periodic boundary condition
# checked and tested by SKo

set tcl_precision 17

# Import the ParFlow TCL package

lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

pfset FileVersion                                       4

set NP                                                  1
set NQ                                                  1
set NR                                                  1
set Proc                                                [ expr $NP*$NQ*$NR ]
set runname                                             nctest

pfset Process.Topology.P                                $NP
pfset Process.Topology.Q                                $NQ
pfset Process.Topology.R                                $NR

#-----------------------------------------------------------------------------
# Computational Grid
#-----------------------------------------------------------------------------

pfset ComputationalGrid.Lower.X                         0.0
pfset ComputationalGrid.Lower.Y                         0.0
pfset ComputationalGrid.Lower.Z                         0.0

set NX          424
set NY          436
set NZ          1

pfset ComputationalGrid.NX               [ expr $NX*$NP ]
pfset ComputationalGrid.NY               [ expr $NY*$NQ ]
pfset ComputationalGrid.NZ               [ expr $NZ*$NR ]

set DX                                                  1.0
set DY                                                  1.0
set DZ                                                  0.5

pfset ComputationalGrid.DX                              $DX
pfset ComputationalGrid.DY                              $DY
pfset ComputationalGrid.DZ                              $DZ

#-----------------------------------------------------------------------------
# The Names of the GeomInputs
#-----------------------------------------------------------------------------

pfset GeomInput.Names                                   "domaininput"

pfset GeomInput.domaininput.GeomName                    domain
pfset GeomInput.domaininput.InputType                   Box

#-----------------------------------------------------------------------------
# Domain Geometry
#-----------------------------------------------------------------------------

pfset Geom.domain.Lower.X                               0.0
pfset Geom.domain.Lower.Y                               0.0
pfset Geom.domain.Lower.Z                               0.0

pfset Geom.domain.Upper.X                               [ expr $NX*$DX*$NP ]
pfset Geom.domain.Upper.Y                               [ expr $NY*$DY*$NQ ]
pfset Geom.domain.Upper.Z                               [ expr $NZ*$DZ*$NR ]
pfset Geom.domain.Patches                               "x-lower x-upper y-lower y-upper z-lower z-upper"

#-----------------------------------------------------------------------------
# variable dz assignments
#-----------------------------------------------------------------------------

pfset Solver.Nonlinear.VariableDz                       True
pfset Solver.Nonlinear.VariableDz                       False

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

pfset Geom.Perm.Names                                   "domain"

# Values in m/hour
pfset Geom.domain.Perm.Type                             Constant
pfset Geom.domain.Perm.Value                            0.25

pfset Perm.TensorType                                   TensorByGeom

pfset Geom.Perm.TensorByGeom.Names                      "domain"

pfset Geom.domain.Perm.TensorValX                       1.0d0
pfset Geom.domain.Perm.TensorValY                       1.0d0
pfset Geom.domain.Perm.TensorValZ                       1.0d0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

pfset SpecificStorage.Type                              Constant
pfset SpecificStorage.GeomNames                         "domain"
pfset Geom.domain.SpecificStorage.Value                 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

pfset Phase.Names                                       "water"

pfset Phase.water.Density.Type                          Constant
pfset Phase.water.Density.Value                         1.0

pfset Phase.water.Viscosity.Type                        Constant
pfset Phase.water.Viscosity.Value                       1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

pfset Contaminants.Names                                ""

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

pfset Geom.Retardation.GeomNames                        ""

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

pfset Gravity                                           1.0

#-----------------------------------------------------------------------------
# Setup timing info [hr]
# dt=30min, simulation time=1d, output interval=1h, no restart
#-----------------------------------------------------------------------------

pfset TimingInfo.BaseUnit                               1.0
pfset TimingInfo.StartCount                             0
pfset TimingInfo.StartTime                              0.0
pfset TimingInfo.StopTime                               48.0
pfset TimingInfo.DumpInterval                           12.0
pfset TimeStep.Type                                     Constant
pfset TimeStep.Value                                    1.0

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

pfset Geom.Porosity.GeomNames                           "domain"

pfset Geom.domain.Porosity.Type                         Constant
pfset Geom.domain.Porosity.Value                        0.25

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

pfset Domain.GeomName                                   domain

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

pfset Phase.RelPerm.Type                                VanGenuchten
pfset Phase.RelPerm.GeomNames                           "domain"

pfset Geom.domain.RelPerm.Alpha                         1.0
pfset Geom.domain.RelPerm.N                             3.0

#-----------------------------------------------------------------------------
# Saturation
#-----------------------------------------------------------------------------

pfset Phase.Saturation.Type                             VanGenuchten
pfset Phase.Saturation.GeomNames                        "domain"

pfset Geom.domain.Saturation.Alpha                      1.0
pfset Geom.domain.Saturation.N                          3.0
pfset Geom.domain.Saturation.SRes                       0.1
pfset Geom.domain.Saturation.SSat                       1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

pfset Wells.Names                                       ""

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

pfset Cycle.Names                                       "constant rainrec"
pfset Cycle.constant.Names                              "alltime"
pfset Cycle.constant.alltime.Length                     1
pfset Cycle.constant.Repeat                             -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 3 hours
pfset Cycle.rainrec.Names                               "rain rec"
pfset Cycle.rainrec.rain.Length                         2
pfset Cycle.rainrec.rec.Length                          6
pfset Cycle.rainrec.Repeat                              -1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

pfset BCPressure.PatchNames                             [pfget Geom.domain.Patches]

pfset Patch.x-lower.BCPressure.Type                     FluxConst
pfset Patch.x-lower.BCPressure.Cycle                    "constant"
pfset Patch.x-lower.BCPressure.alltime.Value            0.0

pfset Patch.y-lower.BCPressure.Type                     FluxConst
pfset Patch.y-lower.BCPressure.Cycle                    "constant"
pfset Patch.y-lower.BCPressure.alltime.Value            0.0

pfset Patch.z-lower.BCPressure.Type                     FluxConst
pfset Patch.z-lower.BCPressure.Cycle                    "constant"
pfset Patch.z-lower.BCPressure.alltime.Value            0.0

pfset Patch.x-upper.BCPressure.Type                     FluxConst
pfset Patch.x-upper.BCPressure.Cycle                    "constant"
pfset Patch.x-upper.BCPressure.alltime.Value            0.0

pfset Patch.y-upper.BCPressure.Type                     FluxConst
pfset Patch.y-upper.BCPressure.Cycle                    "constant"
pfset Patch.y-upper.BCPressure.alltime.Value            0.0

# overland flow boundary condition with very heavy rainfall then slight ET
pfset Patch.z-upper.BCPressure.Type                     OverlandFlow
pfset Patch.z-upper.BCPressure.Cycle                    "constant"
pfset Patch.z-upper.BCPressure.rain.Value               -0.0000005
pfset Patch.z-upper.BCPressure.rec.Value                0.00000
pfset Patch.z-upper.BCPressure.alltime.Value            0.00000000

#-----------------------------------------------------------------------------
# Topo slopes in x-direction
#-----------------------------------------------------------------------------

pfset TopoSlopesX.Type                                  "PredefinedFunction"
pfset TopoSlopesX.GeomNames                             "domain"
pfset TopoSlopesX.PredefinedFunction                    "SineCosTopo"

#-----------------------------------------------------------------------------
# Topo slopes in y-direction
#-----------------------------------------------------------------------------

pfset TopoSlopesY.Type                                  "PredefinedFunction"
pfset TopoSlopesY.GeomNames                             "domain"
pfset TopoSlopesY.PredefinedFunction                    "SineCosTopo"

#-----------------------------------------------------------------------------
# Mannings coefficient
#-----------------------------------------------------------------------------

pfset Mannings.Type                                     "Constant"
pfset Mannings.GeomNames                                "domain"
pfset Mannings.Geom.domain.Value                        5.52e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

pfset PhaseSources.water.Type                           Constant
pfset PhaseSources.water.GeomNames                      domain
pfset PhaseSources.water.Geom.domain.Value              0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

pfset KnownSolution                                     NoKnownSolution

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

pfset Solver                                            Richards
pfset Solver.MaxIter                                    2500

pfset Solver.TerrainFollowingGrid                       True

pfset Solver.Nonlinear.MaxIter                          50
pfset Solver.Nonlinear.ResidualTol                      1e-6
pfset Solver.Nonlinear.StepTol                          1e-20
pfset Solver.Nonlinear.Globalization                    LineSearch
pfset Solver.Linear.KrylovDimension                     50
pfset  Solver.Drop                                      1E-20
pfset Solver.Nonlinear.EtaChoice                        EtaConstant
pfset Solver.Nonlinear.EtaValue                         0.001
pfset Solver.Nonlinear.UseJacobian                      True
pfset Solver.Linear.Preconditioner                      PFMG
pfset Solver.Linear.Preconditioner.PCMatrixType         FullJacobian

pfset Solver.WriteSiloSubsurfData                       False
pfset Solver.WriteSiloMask                              False
pfset Solver.WriteSiloPressure                          False
pfset Solver.WriteSiloSaturation                        False
pfset Solver.WriteSiloSlopes                            False

pfset Solver.PrintSubsurfData                           False
pfset Solver.PrintMask                                  False
pfset Solver.PrintSlopes                                False
pfset Solver.PrintPressure                              False
pfset Solver.PrintSaturation                            False

#-----------------------------------------------------------------------------
# New settings influencing everything in relation to the NetCDF I/O interface
#-----------------------------------------------------------------------------

# CLM with nc
pfset Solver.CLM.MetForcing                             1Dnc
pfset Solver.CLM.MetFilePath                            /Users/lpoorthuis/sandbox
pfset Solver.CLM.MetFileName                            out.nc

# writing variables with NetCDF
pfset NetCDF.WriteSubsurfData                           True
pfset NetCDF.WriteMask                                  True
pfset NetCDF.WritePressure                              True
pfset NetCDF.WriteSaturation                            True
pfset NetCDF.WriteSlopes                                True
pfset NetCDF.WriteMannings                              True
pfset NetCDF.WriteDZMult                                True
pfset NetCDF.WriteCLM                                   True
pfset NetCDF.WriteEvapTrans                             True
pfset NetCDF.WriteEvapTransSum                          True
pfset NetCDF.WriteOverlandSum                           True
pfset NetCDF.WriteOverlandBCFlux                        True

# potential performance tweaks via the NetCDF chunking feature
pfset NetCDF.ChunkingPressure                           False
pfset NetCDF.ChunkingPressure1                          0
pfset NetCDF.ChunkingPressure2                          0
pfset NetCDF.ChunkingPressure3                          0
pfset NetCDF.ChunkingPressure4                          0

pfset NetCDF.ChunkingSaturation                         False
pfset NetCDF.ChunkingSaturation1                        0
pfset NetCDF.ChunkingSaturation2                        0
pfset NetCDF.ChunkingSaturation3                        0
pfset NetCDF.ChunkingSaturation4                        0

# filesystem optimisation set via ROMIO hints from a file
# pfset NetCDF.ROMIOhints                                 romio.hints

# hard file name option
pfset NetCDF.HardFileName                               ""

# from now on vars are irrelevant to the solver routines
# NetCDF global metadata information
pfset NetCDF.Institution                                "institution"
pfset NetCDF.InstituteID                                "IN"
pfset NetCDF.ModelID                                    "PF"
pfset NetCDF.Experiment                                 "ParFlow NetCDF test"
pfset NetCDF.ExperimentID                               "test"
pfset NetCDF.Contact                                    "max.mustermann@muster.com (Max Mustermann)"
pfset NetCDF.Product                                    "output"
pfset NetCDF.VersionID                                  "v1"
pfset NetCDF.Domain                                     "sine"
pfset NetCDF.ProjectID                                  "PFL"
pfset NetCDF.References                                 "http://www.muster-url.com"
pfset NetCDF.Comment                                    "example"

# driving model settings
pfset NetCDF.DrivingModelID                             "sine"
pfset NetCDF.DrivingExperiment                          "sample"
pfset NetCDF.DrivingModelEnsemble                       "1"

# set lat lon vars if they are available
# pfset NetCDF.LatLonFile                                 "data.nc"
# pfset NetCDF.LatLonNames                                "lat lon"

# time vector tracking
# the starthour can be set via TimingInfo.StartTime
pfset NetCDF.StartDateYear                              2016
pfset NetCDF.StartDateMonth                             12
pfset NetCDF.StartDateDay                               30

pfset NetCDF.SplitInterval                              yearly

# specify a dir that contains nc data where the new model data can
# be appended
# pfset NetCDF.AppendDirectory                            "./data_dir"

#-----------------------------------------------------------------------------
# Initial conditions: water pressure
#-----------------------------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
pfset ICPressure.Type                                   HydroStaticPatch
pfset ICPressure.GeomNames                              domain
pfset Geom.domain.ICPressure.Value                      -10.0

pfset Geom.domain.ICPressure.RefGeom                    domain
pfset Geom.domain.ICPressure.RefPatch                   z-upper

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------

pfwritedb $runname
