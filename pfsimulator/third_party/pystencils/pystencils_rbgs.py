from pystencilssfg import SourceFileGenerator

from pystencils_codegen import *

with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info['default_dtype']

    # fields

    ## subentries of A

    a0, a1, a2, a3, a4, a5, a6 = ps.fields(f"a0, a1, a2, a3, a4, a5, a6: {default_dtype}[3D]", layout="fzyx")
    A = [a0, a1, a2, a3, a4, a5, a6]

    ## entries of solution field x offset by stencil
    x0, x1, x2, x3, x4, x5, x6 = ps.fields(f"x0, x1, x2, x3, x4, x5, x6: {default_dtype}[3D]", layout="fzyx")
    x = [x0, x1, x2, x3, x4, x5, x6]

    ## right-hand side
    b = ps.fields(f"b: {default_dtype}[3D]", layout="fzyx")

    # kernels

    ## zero-optimization kernel
    create_kernel_func(sfg, ps.Assignment(x0.center(), b.center() / a0.center()), "RBGS_ZeroOptimizationKernel", allow_vect=False)

    ## regular 7pt kernel
    stencil_convolution = sum([xi.center() * ai.center() for xi, ai in zip(x, A)])
    create_kernel_func(sfg, ps.Assignment(x0.center(), b.center() - stencil_convolution / a0.center()), "RBGS_7PtKernel", allow_vect=False)