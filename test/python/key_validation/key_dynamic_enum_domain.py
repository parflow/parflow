# -----------------------------------------------------------------------------
# SCRIPT TO RUN LITTLE WASHITA DOMAIN WITH TERRAIN-FOLLOWING GRID
# DETAILS:
# Arguments are 1) runname 2) year
# -----------------------------------------------------------------------------

from parflow import Run

LW_Test = Run("LW_Test", __file__)


# -----------------------------------------------------------------------------

LW_Test.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------
LW_Test.Process.Topology.P = 1
LW_Test.Process.Topology.Q = 1
LW_Test.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

LW_Test.ComputationalGrid.Lower.X = 0.0
LW_Test.ComputationalGrid.Lower.Y = 0.0
LW_Test.ComputationalGrid.Lower.Z = 0.0

LW_Test.ComputationalGrid.DX = 1000.0
LW_Test.ComputationalGrid.DY = 1000.0
LW_Test.ComputationalGrid.DZ = 2.0

LW_Test.ComputationalGrid.NX = 41
LW_Test.ComputationalGrid.NY = 41
LW_Test.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

LW_Test.GeomInput.Names = "box_input indi_input"

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

LW_Test.GeomInput.box_input.InputType = "Box"
LW_Test.GeomInput.box_input.GeomName = "domain"

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

LW_Test.Geom.domain.Lower.X = 0.0
LW_Test.Geom.domain.Lower.Y = 0.0
LW_Test.Geom.domain.Lower.Z = 0.0
#
LW_Test.Geom.domain.Upper.X = 41000.0
LW_Test.Geom.domain.Upper.Y = 41000.0
LW_Test.Geom.domain.Upper.Z = 100.0
LW_Test.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

LW_Test.GeomInput.indi_input.InputType = "IndicatorField"
LW_Test.GeomInput.indi_input.GeomNames = (
    "s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8"
)
LW_Test.Geom.indi_input.FileName = "IndicatorFile_Gleeson.50z.pfb"

LW_Test.GeomInput.s1.Value = 1
LW_Test.GeomInput.s2.Value = 2
LW_Test.GeomInput.s3.Value = 3
LW_Test.GeomInput.s4.Value = 4
LW_Test.GeomInput.s5.Value = 5
LW_Test.GeomInput.s6.Value = 6
LW_Test.GeomInput.s7.Value = 7
LW_Test.GeomInput.s8.Value = 8
LW_Test.GeomInput.s9.Value = 9
LW_Test.GeomInput.s10.Value = 10
LW_Test.GeomInput.s11.Value = 11
LW_Test.GeomInput.s12.Value = 12
LW_Test.GeomInput.s13.Value = 13
LW_Test.GeomInput.g1.Value = 21
LW_Test.GeomInput.g2.Value = 22
LW_Test.GeomInput.g3.Value = 23
LW_Test.GeomInput.g4.Value = 24
LW_Test.GeomInput.g5.Value = 25
LW_Test.GeomInput.g6.Value = 26
LW_Test.GeomInput.g7.Value = 27
LW_Test.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

LW_Test.Geom.Perm.Names = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8"

LW_Test.Geom.domain.Perm.Type = "Constant"
LW_Test.Geom.domain.Perm.Value = 0.2

LW_Test.Geom.s1.Perm.Type = "Constant"
LW_Test.Geom.s1.Perm.Value = 0.269022595

LW_Test.Geom.s2.Perm.Type = "Constant"
LW_Test.Geom.s2.Perm.Value = 0.043630356

LW_Test.Geom.s3.Perm.Type = "Constant"
LW_Test.Geom.s3.Perm.Value = 0.015841225

LW_Test.Geom.s4.Perm.Type = "Constant"
LW_Test.Geom.s4.Perm.Value = 0.007582087

LW_Test.Geom.s5.Perm.Type = "Constant"
LW_Test.Geom.s5.Perm.Value = 0.01818816

LW_Test.Geom.s6.Perm.Type = "Constant"
LW_Test.Geom.s6.Perm.Value = 0.005009435

LW_Test.Geom.s7.Perm.Type = "Constant"
LW_Test.Geom.s7.Perm.Value = 0.005492736

LW_Test.Geom.s8.Perm.Type = "Constant"
LW_Test.Geom.s8.Perm.Value = 0.004675077

LW_Test.Geom.s9.Perm.Type = "Constant"
LW_Test.Geom.s9.Perm.Value = 0.003386794

LW_Test.Geom.g2.Perm.Type = "Constant"
LW_Test.Geom.g2.Perm.Value = 0.025

LW_Test.Geom.g3.Perm.Type = "Constant"
LW_Test.Geom.g3.Perm.Value = 0.059

LW_Test.Geom.g6.Perm.Type = "Constant"
LW_Test.Geom.g6.Perm.Value = 0.2

LW_Test.Geom.g8.Perm.Type = "Constant"
LW_Test.Geom.g8.Perm.Value = 0.68

LW_Test.Perm.TensorType = "TensorByGeom"
LW_Test.Geom.Perm.TensorByGeom.Names = "domain"
LW_Test.Geom.domain.Perm.TensorValX = 1.0
LW_Test.Geom.domain.Perm.TensorValY = 1.0
LW_Test.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

LW_Test.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

LW_Test.TimingInfo.BaseUnit = 1.0
LW_Test.TimingInfo.StartCount = 0.0
LW_Test.TimingInfo.StartTime = 0.0
LW_Test.TimingInfo.StopTime = 12.0
LW_Test.TimingInfo.DumpInterval = 24.0
LW_Test.TimeStep.Type = "Constant"
LW_Test.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

LW_Test.Geom.Porosity.GeomNames = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9"

LW_Test.Geom.domain.Porosity.Type = "Constant"
LW_Test.Geom.domain.Porosity.Value = 0.4

LW_Test.Geom.s1.Porosity.Type = "Constant"
LW_Test.Geom.s1.Porosity.Value = 0.375

LW_Test.Geom.s2.Porosity.Type = "Constant"
LW_Test.Geom.s2.Porosity.Value = 0.39

LW_Test.Geom.s3.Porosity.Type = "Constant"
LW_Test.Geom.s3.Porosity.Value = 0.387

LW_Test.Geom.s4.Porosity.Type = "Constant"
LW_Test.Geom.s4.Porosity.Value = 0.439

LW_Test.Geom.s5.Porosity.Type = "Constant"
LW_Test.Geom.s5.Porosity.Value = 0.489

LW_Test.Geom.s6.Porosity.Type = "Constant"
LW_Test.Geom.s6.Porosity.Value = 0.399

LW_Test.Geom.s7.Porosity.Type = "Constant"
LW_Test.Geom.s7.Porosity.Value = 0.384

LW_Test.Geom.s8.Porosity.Type = "Constant"
LW_Test.Geom.s8.Porosity.Value = 0.482

LW_Test.Geom.s9.Porosity.Type = "Constant"
LW_Test.Geom.s9.Porosity.Value = 0.442

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

LW_Test.Domain.GeomName = "domain"

import unittest


class TestBaseCase(unittest.TestCase):
    """
    Values inside enum locations should fail
    """

    def test(self):
        LW_Test.Geom.Perm.Names = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8"
        result = LW_Test.Geom.Perm.validate()
        self.assertEqual(result, 0)


class TestExtraValueCase(unittest.TestCase):
    """
    Value outside enum locations should fail
    """

    def test(self):
        LW_Test.Geom.Perm.Names = "domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8 test"
        result = LW_Test.Geom.Perm.validate()
        self.assertEqual(result, 1)


unittest.main()
