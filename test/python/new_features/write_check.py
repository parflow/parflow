# -----------------------------------------------------------------------------
# Testing file writing
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import get_absolute_path

dsingle = Run("dsingle", __file__)

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
dsingle.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

dsingle.Process.Topology.P = 1
dsingle.Process.Topology.Q = 1
dsingle.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
dsingle.ComputationalGrid.Lower.X = -10.0
dsingle.ComputationalGrid.Lower.Y = 10.0
dsingle.ComputationalGrid.Lower.Z = 1.0

dsingle.ComputationalGrid.DX = 8.8888888888888893
dsingle.ComputationalGrid.DY = 10.666666666666666
dsingle.ComputationalGrid.DZ = 1.0

dsingle.ComputationalGrid.NX = 18
dsingle.ComputationalGrid.NY = 15
dsingle.ComputationalGrid.NZ = 8

# -----------------------------------------------------------------------------
# The Names of the GeomInputs
# -----------------------------------------------------------------------------
dsingle.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)


# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------
dsingle.GeomInput.domain_input.InputType = "Box"
dsingle.GeomInput.domain_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------
dsingle.Geom.domain.Lower.X = -10.0
dsingle.Geom.domain.Lower.Y = 10.0
dsingle.Geom.domain.Lower.Z = 1.0

dsingle.Geom.domain.Upper.X = 150.0
dsingle.Geom.domain.Upper.Y = 170.0
dsingle.Geom.domain.Upper.Z = 9.0

dsingle.Geom.domain.Patches = "left right front back bottom top"

# -----------------------------------------------------------------------------
# Background Geometry Input
# -----------------------------------------------------------------------------
dsingle.GeomInput.background_input.InputType = "Box"
dsingle.GeomInput.background_input.GeomName = "background"

# -----------------------------------------------------------------------------
# Background Geometry
# -----------------------------------------------------------------------------
dsingle.Geom.background.Lower.X = -99999999.0
dsingle.Geom.background.Lower.Y = -99999999.0
dsingle.Geom.background.Lower.Z = -99999999.0

dsingle.Geom.background.Upper.X = 99999999.0
dsingle.Geom.background.Upper.Y = 99999999.0
dsingle.Geom.background.Upper.Z = 99999999.0


# -----------------------------------------------------------------------------
# Source_Region Geometry Input
# -----------------------------------------------------------------------------
dsingle.GeomInput.source_region_input.InputType = "Box"
dsingle.GeomInput.source_region_input.GeomName = "source_region"

# -----------------------------------------------------------------------------
# Source_Region Geometry
# -----------------------------------------------------------------------------
dsingle.Geom.source_region.Lower.X = 65.56
dsingle.Geom.source_region.Lower.Y = 79.34
dsingle.Geom.source_region.Lower.Z = 4.5

dsingle.Geom.source_region.Upper.X = 74.44
dsingle.Geom.source_region.Upper.Y = 89.99
dsingle.Geom.source_region.Upper.Z = 5.5


# -----------------------------------------------------------------------------
# Concen_Region Geometry Input
# -----------------------------------------------------------------------------
dsingle.GeomInput.concen_region_input.InputType = "Box"
dsingle.GeomInput.concen_region_input.GeomName = "concen_region"

# -----------------------------------------------------------------------------
# Concen_Region Geometry
# -----------------------------------------------------------------------------
dsingle.Geom.concen_region.Lower.X = 60.0
dsingle.Geom.concen_region.Lower.Y = 80.0
dsingle.Geom.concen_region.Lower.Z = 4.0

dsingle.Geom.concen_region.Upper.X = 80.0
dsingle.Geom.concen_region.Upper.Y = 100.0
dsingle.Geom.concen_region.Upper.Z = 6.0

# -----------------------------------------------------------------------------
# Perm
# -----------------------------------------------------------------------------
dsingle.Geom.Perm.Names = "background"

dsingle.Geom.background.Perm.Type = "Constant"
dsingle.Geom.background.Perm.Value = 4.0

dsingle.Perm.TensorType = "TensorByGeom"

dsingle.Geom.Perm.TensorByGeom.Names = "background"

dsingle.Geom.background.Perm.TensorValX = 1.0
dsingle.Geom.background.Perm.TensorValY = 1.0
dsingle.Geom.background.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------
# specific storage does not figure into the impes (fully sat) case but we still
# need a key for it

dsingle.SpecificStorage.Type = "Constant"
dsingle.SpecificStorage.GeomNames = ""
dsingle.Geom.domain.SpecificStorage.Value = 1.0e-4

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

dsingle.Phase.Names = "water"

dsingle.Phase.water.Density.Type = "Constant"
dsingle.Phase.water.Density.Value = 1.0

dsingle.Phase.water.Viscosity.Type = "Constant"
dsingle.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------
dsingle.Contaminants.Names = "tce"
dsingle.Contaminants.tce.Degradation.Value = 0.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

dsingle.Gravity = 1.0

# -----------------------------------------------------------------------------
# Setup timing info
# -----------------------------------------------------------------------------

dsingle.TimingInfo.BaseUnit = 1.0
dsingle.TimingInfo.StartCount = 0
dsingle.TimingInfo.StartTime = 0.0
dsingle.TimingInfo.StopTime = 1000.0
dsingle.TimingInfo.DumpInterval = -1

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

dsingle.Geom.Porosity.GeomNames = "background"

dsingle.Geom.background.Porosity.Type = "Constant"
dsingle.Geom.background.Porosity.Value = 1.0

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
dsingle.Domain.GeomName = "domain"

# -----------------------------------------------------------------------------
# Mobility
# -----------------------------------------------------------------------------
dsingle.Phase.water.Mobility.Type = "Constant"
dsingle.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Retardation
# -----------------------------------------------------------------------------
dsingle.Geom.Retardation.GeomNames = "background"
dsingle.Geom.background.tce.Retardation.Type = "Linear"
dsingle.Geom.background.tce.Retardation.Rate = 0.0

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------
dsingle.Cycle.Names = "constant"
dsingle.Cycle.constant.Names = "alltime"
dsingle.Cycle.constant.alltime.Length = 1
dsingle.Cycle.constant.Repeat = -1

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------
dsingle.Wells.Names = "snoopy"

dsingle.Wells.snoopy.InputType = "Recirc"

dsingle.Wells.snoopy.Cycle = "constant"

dsingle.Wells.snoopy.ExtractionType = "Flux"
dsingle.Wells.snoopy.InjectionType = "Flux"

dsingle.Wells.snoopy.X = 71.0
dsingle.Wells.snoopy.Y = 90.0
dsingle.Wells.snoopy.ExtractionZLower = 5.0
dsingle.Wells.snoopy.ExtractionZUpper = 5.0
dsingle.Wells.snoopy.InjectionZLower = 2.0
dsingle.Wells.snoopy.InjectionZUpper = 2.0

dsingle.Wells.snoopy.ExtractionMethod = "Standard"
dsingle.Wells.snoopy.InjectionMethod = "Standard"

dsingle.Wells.snoopy.alltime.Extraction.Flux.water.Value = 5.0
dsingle.Wells.snoopy.alltime.Injection.Flux.water.Value = 7.5
dsingle.Wells.snoopy.alltime.Injection.Concentration.water.tce.Fraction = 0.1

# -----------------------------------------------------------------------------
# Assigning well with newly assigned interval name
# -----------------------------------------------------------------------------

dsingle.Wells.snoopy.alltime.Extraction.Flux.water.Value = 5.0
dsingle.Wells.snoopy.alltime.Injection.Flux.water.Value = 7.5
dsingle.Wells.snoopy.alltime.Injection.Concentration.water.tce.Fraction = 0.1

# -----------------------------------------------------------------------------
# Boundary Conditions: Pressure
# -----------------------------------------------------------------------------
dsingle.BCPressure.PatchNames = "left right front back bottom top"

dsingle.Patch.left.BCPressure.Type = "DirEquilRefPatch"
dsingle.Patch.left.BCPressure.Cycle = "constant"
dsingle.Patch.left.BCPressure.RefGeom = "domain"
dsingle.Patch.left.BCPressure.RefPatch = "bottom"
dsingle.Patch.left.BCPressure.alltime.Value = 14.0

dsingle.Patch.right.BCPressure.Type = "DirEquilRefPatch"
dsingle.Patch.right.BCPressure.Cycle = "constant"
dsingle.Patch.right.BCPressure.RefGeom = "domain"
dsingle.Patch.right.BCPressure.RefPatch = "bottom"
dsingle.Patch.right.BCPressure.alltime.Value = 9.0

dsingle.Patch.front.BCPressure.Type = "FluxConst"
dsingle.Patch.front.BCPressure.Cycle = "constant"
dsingle.Patch.front.BCPressure.alltime.Value = 0.0

dsingle.Patch.back.BCPressure.Type = "FluxConst"
dsingle.Patch.back.BCPressure.Cycle = "constant"
dsingle.Patch.back.BCPressure.alltime.Value = 0.0

dsingle.Patch.bottom.BCPressure.Type = "FluxConst"
dsingle.Patch.bottom.BCPressure.Cycle = "constant"
dsingle.Patch.bottom.BCPressure.alltime.Value = 0.0

dsingle.Patch.top.BCPressure.Type = "FluxConst"
dsingle.Patch.top.BCPressure.Cycle = "constant"
dsingle.Patch.top.BCPressure.alltime.Value = 0.0


# ---------------------------------------------------------
# Topo slopes in x-direction
# ---------------------------------------------------------
# topo slopes do not figure into the impes (fully sat) case but we still
# need keys for them

dsingle.TopoSlopesX.Type = "Constant"
dsingle.TopoSlopesX.GeomNames = "domain"

dsingle.TopoSlopesX.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Topo slopes in y-direction
# ---------------------------------------------------------

dsingle.TopoSlopesY.Type = "Constant"
dsingle.TopoSlopesY.GeomNames = "domain"

dsingle.TopoSlopesY.Geom.domain.Value = 0.0

# ---------------------------------------------------------
# Mannings coefficient
# ---------------------------------------------------------
# mannings roughnesses do not figure into the impes (fully sat) case but we still
# need a key for them

dsingle.Mannings.Type = "Constant"
dsingle.Mannings.GeomNames = "domain"
dsingle.Mannings.Geom.domain.Value = 0.0


# ---------------------------------------------------------
# dzScale values
# ---------------------------------------------------------

dsingle.Solver.Nonlinear.VariableDz = True
dsingle.dzScale.Type = "nzList"
dsingle.dzScale.nzListNumber = 8
dsingle.Cell._0.dzScale.Value = 0.5
dsingle.Cell._1.dzScale.Value = 0.5
dsingle.Cell._2.dzScale.Value = 0.5
dsingle.Cell._3.dzScale.Value = 0.5
dsingle.Cell._4.dzScale.Value = 0.5
dsingle.Cell._5.dzScale.Value = 0.5
dsingle.Cell._6.dzScale.Value = 0.5
dsingle.Cell._7.dzScale.Value = 0.5


# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

dsingle.PhaseSources.water.Type = "Constant"
dsingle.PhaseSources.water.GeomNames = "background"
dsingle.PhaseSources.water.Geom.background.Value = 0.0

dsingle.PhaseConcen.water.tce.Type = "Constant"
dsingle.PhaseConcen.water.tce.GeomNames = "concen_region"
dsingle.PhaseConcen.water.tce.Geom.concen_region.Value = 0.8


dsingle.Solver.WriteSiloSubsurfData = True
dsingle.Solver.WriteSiloPressure = True
dsingle.Solver.WriteSiloSaturation = True
dsingle.Solver.WriteSiloConcentration = True


# -----------------------------------------------------------------------------
# The Solver Impes MaxIter default value changed so to get previous
# results we need to set it back to what it was
# -----------------------------------------------------------------------------
dsingle.Solver.MaxIter = 5


# -----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
# -----------------------------------------------------------------------------

generatedFile, runArg = dsingle.write()

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/dsingle.pfidb.ref")
) as ref:
    if new.read() == ref.read():
        print("Success we have the same file")
    else:
        print("Files are different")
        sys.exit(1)
