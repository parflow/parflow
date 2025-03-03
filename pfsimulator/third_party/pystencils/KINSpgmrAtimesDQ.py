import pystencils as ps
import sympy as sp

from pystencilssfg import SourceFileGenerator
from pystencilssfg.lang.cpp import *

with SourceFileGenerator() as sfg:
    dtype = "float64"

    sfg.include("stdint.h")
    sfg.include("<cstdint>", True)

    # Kernel 1:
    # z_i = x_i * y_i (PFVProd)
    x, y, z = ps.fields("x, y, z: float64[3D]", layout="fzyx")

    update_rule_PFVProd = ps.Assignment(z.center(), x.center() * y.center())
    update_kernel_PFVProd = sfg.kernels.create(update_rule_PFVProd, "VProd_gen")

    sfg.function("PyCodegen_VProd")(sfg.call(update_kernel_PFVProd))

    # Kernel 2:
    # DotProd = x dot y (PFVDotProd)
    r = ps.TypedSymbol("r", dtype)
    update_rule_PFVDotProd = ps.AddReductionAssignment(r, x.center() * y.center())
    update_kernel_PFVDotProd = sfg.kernels.create(update_rule_PFVDotProd, "PFVDotProd_gen")

    sfg.function("PyCodegen_PFVDotProd")(sfg.call(update_kernel_PFVDotProd))

    # Kernel 3:
    # z = a * x + b * y (PFVLinearSum)
    a, b = sp.symbols("a, b")

    update_rule_PFVLinearSum = ps.Assignment(z.center(), a * x.center() + b * y.center())
    update_kernel_PFVLinearSum = sfg.kernels.create(update_rule_PFVLinearSum, "VLinearSum_gen")

    sfg.function("PyCodegen_VLinearSum")(sfg.call(update_kernel_PFVLinearSum))