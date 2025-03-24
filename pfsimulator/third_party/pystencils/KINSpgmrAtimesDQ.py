import pystencils as ps
import sympy as sp

from pystencilssfg import SourceFileGenerator, SfgConfig
from pystencilssfg.lang.cpp import *

with SourceFileGenerator() as sfg:
    dtype = "float64"

    def invoke(kernel):
        # TODO: invoke handling for CUDA
        # if sfg.context.project_info['use_cuda']:
        # ...
        #else:
        sfg.call(kernel)


    # z_i = x_i * y_i (PFVProd)
    x, y, z = ps.fields(f"x, y, z: {dtype}[3D]", layout="fzyx")

    update_rule_PFVProd = ps.Assignment(z.center(), x.center() * y.center())
    update_kernel_PFVProd = sfg.kernels.create(update_rule_PFVProd, "VProd_gen")

    sfg.function("PyCodegen_VProd")(invoke(update_kernel_PFVProd))

    # DotProd = x dot y (PFVDotProd)
    r = ps.TypedSymbol("r", dtype)
    update_rule_PFVDotProd = ps.AddReductionAssignment(r, x.center() * y.center())
    update_kernel_PFVDotProd = sfg.kernels.create(update_rule_PFVDotProd, "PFVDotProd_gen")

    sfg.function("PyCodegen_VDotProd")(invoke(update_kernel_PFVDotProd))

    # L1Norm = sum_i |x_i| (PFVL1Norm)
    r = ps.TypedSymbol("r", dtype)
    update_rule_PFVL1Norm = ps.AddReductionAssignment(r, sp.Abs(x.center()))
    update_kernel_PFVL1Norm = sfg.kernels.create(update_rule_PFVL1Norm, "PFVL1Norm_gen")

    sfg.function("PyCodegen_VL1Norm")(invoke(update_kernel_PFVL1Norm))

    # z = a * x + b * y (PFVLinearSum)
    a, b = sp.symbols("a, b")

    update_rule_PFVLinearSum = ps.Assignment(z.center(), a * x.center() + b * y.center())
    update_kernel_PFVLinearSum = sfg.kernels.create(update_rule_PFVLinearSum, "VLinearSum_gen")

    sfg.function("PyCodegen_VLinearSum")(invoke(update_kernel_PFVLinearSum))