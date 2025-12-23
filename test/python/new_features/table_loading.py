from parflow import Run
from parflow.tools.builders import SubsurfacePropertiesBuilder

run = Run("table_test", __file__)

run.GeomInput.Names = "s1 s2 s3 s4"

soil_properties = """
# -----------------------------------------------------------------------------
# Sample header
# -----------------------------------------------------------------------------
key  Perm  Porosity  SpecificStorage  RelPerm  RelPermA  RelPermN A

s1   0.3   0.9       0.567            1.1      1.2       1.4      5
s2   0.4   0.87      0.76             2.2      2.3       1.7      6
s3   0.5   0.345     0.2234           3.3      2.4       2.3      7

# -----------------------------------------------------------------------------
# Deep ground
# -----------------------------------------------------------------------------

s4   0.6   0.567     0.554            4.4      3.5       3.4      8
s5   0.65  0.675     0.455            5.5      4.5       5.6      9

s6   -     0.7       -                6.6      -         -        -
"""

soil_builder = SubsurfacePropertiesBuilder(run).load_txt_content(soil_properties)

print("-" * 80)
print("basic mapping")
print("-" * 80)
soil_builder.print()
print("-" * 80)
print("remap s5 to s4")
print("-" * 80)
soil_builder.assign("s5", "s4").print()

print("-" * 80)
print("remap s6 to s4 (with skip)")
print("-" * 80)
soil_builder.assign("s6", "s4").print()

print("+" * 80)
soil_properties_transpose = """
# -----------------------------------------------------------------------------
# Sample header transposed
# -----------------------------------------------------------------------------
key           s1      s2     s3       s4      s5      s6
Perm          0.3     0.4    0.5      0.6     0.65    -
Porosity      0.9     0.87   0.345    0.567   0.675   0.7
SpecStorage   0.567   0.76   0.2234   0.554   0.455   -
RelPerm       1.1     2.2    3.3      4.4     5.5     6.6
RelPermA      1.2     2.3    2.4      3.5     4.5     -
RelPermN      1.4     1.7    2.3      3.4     5.6     -
A             5       6      7        8       9       -
# -----------------------------------------------------------------------------
"""
soil_builder = SubsurfacePropertiesBuilder(run).load_txt_content(
    soil_properties_transpose
)

print("-" * 80)
print("basic mapping")
print("-" * 80)
soil_builder.print()
print("-" * 80)
print("remap s5 to s4")
print("-" * 80)
soil_builder.assign("s5", "s4").print()
print("-" * 80)
print("remap s6 to s4 (with skip)")
print("-" * 80)
soil_builder.assign("s6", "s4").print()

soil_builder.apply()

run.write()
