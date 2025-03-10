import pystencils as ps
import sympy as sp

from pystencilssfg import SourceFileGenerator, SfgConfig
from pystencilssfg.lang.cpp import *

cfg = SfgConfig()

with SourceFileGenerator(cfg) as sfg:
    dtype = "float64"

    # z_i = x_i * y_i (PFVProd)
    x, y, z = ps.fields("x, y, z: float64[3D]", layout="fzyx")

    update_rule_PFVProd = ps.Assignment(z.center(), x.center() * y.center())
    update_kernel_PFVProd = sfg.kernels.create(update_rule_PFVProd, "VProd_gen")

    sfg.function("PyCodegen_VProd").externC()(sfg.call(update_kernel_PFVProd))

    # DotProd = x dot y (PFVDotProd)
    r = ps.TypedSymbol("r", dtype)
    update_rule_PFVDotProd = ps.AddReductionAssignment(r, x.center() * y.center())
    update_kernel_PFVDotProd = sfg.kernels.create(update_rule_PFVDotProd, "PFVDotProd_gen")

    sfg.function("PyCodegen_VDotProd").externC()(sfg.call(update_kernel_PFVDotProd))

    # L1Norm = sum_i |x_i| (PFVL1Norm)
    r = ps.TypedSymbol("r", dtype)
    update_rule_PFVL1Norm = ps.AddReductionAssignment(r, sp.Abs(x.center()))
    update_kernel_PFVL1Norm = sfg.kernels.create(update_rule_PFVL1Norm, "PFVL1Norm_gen")

    sfg.function("PyCodegen_VL1Norm").externC()(sfg.call(update_kernel_PFVL1Norm))

    # z = a * x + b * y (PFVLinearSum)
    a, b = sp.symbols("a, b")

    update_rule_PFVLinearSum = ps.Assignment(z.center(), a * x.center() + b * y.center())
    update_kernel_PFVLinearSum = sfg.kernels.create(update_rule_PFVLinearSum, "VLinearSum_gen")

    sfg.function("PyCodegen_VLinearSum").externC()(sfg.call(update_kernel_PFVLinearSum))