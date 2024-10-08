# -----------------------------------------------------------------------------
#  Testing pfset python function for setting keys that aren't in the library
# -----------------------------------------------------------------------------

import sys
from parflow import Run
from parflow.tools.fs import get_absolute_path

pfset_test = Run("pfset_test", __file__)

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------
pfset_test.Domain.GeomName = "domain"

# ---------------------------------------------------------
# The Names of the GeomInputs
# ---------------------------------------------------------
pfset_test.GeomInput.Names = (
    "domain_input background_input source_region_input concen_region_input"
)

# ---------------------------------------------------------
# Domain Geometry Input
# ---------------------------------------------------------
pfset_test.GeomInput.domain_input.InputType = "Box"
pfset_test.GeomInput.domain_input.GeomName = "domain"

# ---------------------------------------------------------
# Background Geometry Input
# ---------------------------------------------------------
pfset_test.GeomInput.background_input.InputType = "Box"
pfset_test.GeomInput.background_input.GeomName = "background"

# ---------------------------------------------------------
# Source_Region Geometry Input
# ---------------------------------------------------------
pfset_test.GeomInput.source_region_input.InputType = "Box"
pfset_test.GeomInput.source_region_input.GeomName = "source_region"

# ---------------------------------------------------------
# Concen_Region Geometry Input
# ---------------------------------------------------------
pfset_test.GeomInput.concen_region_input.InputType = "Box"
pfset_test.GeomInput.concen_region_input.GeomName = "concen_region"

# ---------------------------------------------------------
# Phase names
# ---------------------------------------------------------
pfset_test.Phase.Names = "water"

# ---------------------------------------------------------
# pfset: Key/Value
# ---------------------------------------------------------

# Test key that does not exist
pfset_test.pfset(key="A.New.Key.Test", value="SomeSuperContent")
# Test key that does not exist with partial valid path
pfset_test.pfset(key="Process.Topology.Random.Path", value=5)
# Test key that does not exist from a child element
pfset_test.Process.Topology.pfset(key="Random.PathFromTopology", value=6)
# Test setting a valid value from a full path
pfset_test.pfset(key="Process.Topology.P", value=2)
# Test setting a valid value from a relative path
pfset_test.Process.pfset(key="Topology.Q", value=3)
# Test setting a valid field
pfset_test.Process.Topology.pfset(key="R", value=4)
# Test setting an invalid field
pfset_test.Process.Topology.pfset(key="Seb", value=5)

# -----------------------------------------------------------------------------
# pfset: hierarchical_map
# -----------------------------------------------------------------------------

pfset_test.pfset(
    hierarchical_map={
        "SpecificStorage": {
            "Type": "Constant",
            "GeomNames": "domain",
        }
    }
)

constOne = {"Type": "Constant", "Value": 1.0}
pfset_test.Phase.water.Density.pfset(hierarchical_map=constOne)
pfset_test.Phase.water.Viscosity.pfset(flat_map=constOne)

# ---------------------------------------------------------
# pfset: yaml_file
# ---------------------------------------------------------

pfset_test.pfset(yaml_file="$PF_SRC/test/input/BasicSettings.yaml")
pfset_test.pfset(yaml_file="$PF_SRC/test/input/ComputationalGrid.yaml")
pfset_test.Geom.pfset(yaml_file="$PF_SRC/test/input/GeomChildren.yaml")

# ---------------------------------------------------------
# pfset: yaml_content
# ---------------------------------------------------------

pfset_test.Geom.source_region.pfset(
    yaml_content="""
Lower:
  X: 65.56
  Y: 79.34
  Z: 4.5
Upper:
  X: 74.44
  Y: 89.99
  Z: 5.5
"""
)


pfset_test.Geom.concen_region.pfset(
    yaml_content="""
Lower:
  X: 60.0
  Y: 80.0
  Z: 4.0
Upper:
  X: 80.0
  Y: 100.0
  Z: 6.0
"""
)

# -----------------------------------------------------------------------------
# pfset: flat_map
# -----------------------------------------------------------------------------

pfset_test.pfset(
    flat_map={
        "Phase.Saturation.Type": "VanGenuchten",
        "Phase.Saturation.GeomNames": "domain",
    }
)

pfset_test.Phase.pfset(
    flat_map={
        "RelPerm.Type": "VanGenuchten",
        "RelPerm.GeomNames": "domain",
    }
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

pfset_test.validate()
generatedFile, runFile = pfset_test.write(file_format="yaml")

# Prevent regression
with open(generatedFile) as new, open(
    get_absolute_path("$PF_SRC/test/correct_output/pfset_test.yaml.ref")
) as ref:
    if new.read() == ref.read():
        print("Success we have the same file")
    else:
        print("Files are different")
        sys.exit(1)
