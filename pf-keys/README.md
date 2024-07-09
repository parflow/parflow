# ParFlow Key Definitions

This directory contains the key definition files and the generator scripts to generate the Python library of ParFlow
keys and the readthedocs documentation.

### Missing keys

Our inventory of keys is based on the current version of the manual (as of August 2020) and checking the ParFlow code.
The following is a list of the keys that are currently missing from the manual but are in the ParFlow code:

 - {_}.BetaFluid
 - {_}.BetaFracture
 - {_}.BetaPerm
 - {_}.BetaPore
 - {_}.BoxSizePowerOf2
 - {_}.CLM.FstepStart
 - {_}.CLM.IrrigationThresholdType
 - {_}.CLM.MetFileSubdir
 - {_}.CoarseSolve
 - {_}.CompCompress
 - {_}.DiagScale
 - {_}.DiagSolver
 - {_}.DropTol
 - {_}.EvapTrans.FileLooping
 - {_}.Jacobian
 - {_}.MaxLevels
 - {_}.MaxMinNX
 - {_}.MaxMinNY
 - {_}.MaxMinNZ
 - {_}.NonlinearSolver
 - {_}.PolyDegree
 - {_}.PolyPC
 - {_}.Smoother
 - {_}.Spinup
 - {_}.Symmetric
 - {_}.TwoNorm
 - {_}.Weight
 - Cycle.Names
 - Geom.*geom_name*.HeatCapacity.Value
 - Geom.*geom_name*.FileName
 - Geom.*geom_name*.HeatCapacity.Value
 - Geom.*geom_name*.Perm.MaxSearchRad
 - Geom.*geom_name*.RelPerm.InterpolationMethod
 - Geom.*geom_name*.ThermalConductivity.KDry
 - Geom.*geom_name*.ThermalConductivity.KDry.Filename
 - Geom.*geom_name*.ThermalConductivity.KWet
 - Geom.*geom_name*.ThermalConductivity.KWet.Filename
 - Geom.*geom_name*.ThermalConductivity.Value
 - Geom.*geometry_name*.contaminant_name.Retardation.Rate
 - Geom.*geometry_name*.Porosity.FileName
 - GeomInput.geom_input_name.GeomName
 - OverlandFlowDiffusive
 - Phase.*phase_name*.Geom.*geom_name*.HeatCapacity.Value
 - Phase.*phase_name*.HeatCapacity.GeomNames
 - Phase.*phase_name*.HeatCapacity.Type
 - Phase.*phase_name*.InternalEnergy.Type
 - Phase.*phase_name*.InternalEnergy.Value
 - Phase.ThermalConductivity.Function1.File
 - PhaseSources.Geom.*geom_name*.Value
 - Solver
 - Solver.Linear.MaxRestart
 - Solver.Linear.Preconditioner.PCMatrixType
 - Solver.PrintDZMultiplier
 - Solver.PrintEvapTrans
 - Solver.PrintEvapTransSum
 - Solver.PrintMannings
 - Solver.PrintMask
 - Solver.PrintOverlandBCFlux
 - Solver.PrintOverlandSum
 - Solver.PrintSlopes
 - Solver.PrintSpecificStorage
 - Solver.PrintSubsurfData
 - Solver.PrintTop
 - Solver.RAPType
 - Solver.WritePfbMannings
 - Solver.WritePfbSlopes
 - Solver.WriteSiloCLM
 - Solver.WriteSiloDZMultiplier
 - Solver.WriteSiloOverlandBCFlux
 - Solver.WriteSiloPMPIOConcentration
 - Solver.WriteSiloPMPIODZMultiplier
 - Solver.WriteSiloPMPIOEvapTrans
 - Solver.WriteSiloPMPIOEvapTransSum
 - Solver.WriteSiloPMPIOMannings
 - Solver.WriteSiloPMPIOMask
 - Solver.WriteSiloPMPIOOverlandBCFlux
 - Solver.WriteSiloPMPIOOverlandSum
 - Solver.WriteSiloPMPIOPressure
 - Solver.WriteSiloPMPIOSaturation
 - Solver.WriteSiloPMPIOSlopes
 - Solver.WriteSiloPMPIOSpecificStorage
 - Solver.WriteSiloPMPIOSubsurfData
 - Solver.WriteSiloPMPIOTop
 - Solver.WriteSiloPMPIOVelocities
 - Solver.WriteSiloTop
 - TopoSlopes.Elevation.FileName
 - TopoSlopesY.Geom.*geometry_name*.Value
 - TopoSlopesY.Type
