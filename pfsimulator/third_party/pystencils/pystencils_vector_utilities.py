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
        return sfg.call(kernel)

    def create_kernel_func(assign, func_name):
        k = sfg.kernels.create(assign, f"{func_name}_gen")
        sfg.function(f"PyCodegen_{func_name}")(invoke(k))

    # symbols

    a, b, c = sp.symbols("a, b, c")

    r = ps.TypedSymbol("r", dtype)

    # fields

    x, y, z = ps.fields(f"x, y, z: {dtype}[3D]", layout="fzyx")

    # kernels

    ## z_i = x_i * y_i (PFVProd)
    create_kernel_func(ps.Assignment(z.center(), x.center() * y.center()), "VProd")

    ## z = c * x (PFVScale)
    create_kernel_func(ps.Assignment(z.center(), c * x.center()), "VScale")

    ## DotProd = x dot y (PFVDotProd)
    create_kernel_func(ps.AddReductionAssignment(r, x.center() * y.center()), "VDotProd")

    ## L1Norm = sum_i |x_i| (PFVL1Norm)
    create_kernel_func(ps.AddReductionAssignment(r, sp.Abs(x.center())), "VL1Norm")

    ## z = a * x + b * y (PFVLinearSum)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() + b * y.center()), "VLinearSum")