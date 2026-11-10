import sympy as sp

from pystencils import TypedSymbol
from pystencils import DynamicType

from field_factory import FieldFactory
from pystencils_codegen import *

with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info["default_dtype"]

    # iteration space

    nx = TypedSymbol("_size_0", DynamicType.INDEX_TYPE)
    ny = TypedSymbol("_size_1", DynamicType.INDEX_TYPE)
    nz = TypedSymbol("_size_2", DynamicType.INDEX_TYPE)

    def strides(name):
        return tuple(TypedSymbol(f"_stride_{name}_{d}", DynamicType.INDEX_TYPE) for d in range(3))

    # fused kernel: residual r = b - A x of a 7pt stencil, fusing Copy(b, r) and Matvec(-1.0, A, x, 1.0, r)

    ## matrix A, solution x offset by stencil and residual r (shared vector strides), right-hand side b
    m_ff = FieldFactory((nx, ny, nz), strides("m"))
    v_ff = FieldFactory((nx, ny, nz), strides("v"))
    b_ff = FieldFactory((nx, ny, nz), strides("b"))

    A = [m_ff.create_new(f"a{i}") for i in range(7)]
    x = [v_ff.create_new(f"x{i}") for i in range(7)]
    r = v_ff.create_new("r")
    b = b_ff.create_new("b")

    ## the stencil sum is its own value, so r = b - (A x) is not expanded into a sum with b
    stencil_convolution = sp.Symbol("ax")
    create_kernel_func(
        sfg,
        [
            ps.Assignment(stencil_convolution, sum([ai.center() * xi.center() for ai, xi in zip(A, x)])),
            ps.Assignment(r.center(), b.center() - stencil_convolution),
        ],
        "MGSemi_FusedCopyAndMatvec7Pt",
    )

    # fused kernels: prolongation of the coarse error e_c to the fine error e_f, fused with the update
    # x = x + e_f (Axpy).
    # The fine fields are strided by the coarsening (SubregionSX/SY/SZ), so these kernels are not vectorized.

    ## fine error e_f, solution x and coarse error e_c
    f_ff = FieldFactory((nx, ny, nz), strides("f"))
    x_ff = FieldFactory((nx, ny, nz), strides("x"))
    c_ff = FieldFactory((nx, ny, nz), strides("c"))

    e_f = f_ff.create_new("e_f")
    x_fine = x_ff.create_new("x")
    e_c = c_ff.create_new("e_c")

    ## coarse points: injection
    create_kernel_func(
        sfg,
        [
            ps.Assignment(e_f.center(), e_c.center()),
            ps.Assignment(x_fine.center(), x_fine.center() + e_c.center()),
        ],
        "MGSemi_FusedProlongInjectionAndAxpy",
        allow_vect=False,
    )

    ## fine points: interpolation from the neighboring coarse points of e_f, weighted by the
    ## transfer operator P (p1, p2)
    p_ff = FieldFactory((nx, ny, nz), strides("p"))
    p1 = p_ff.create_new("p1")
    p2 = p_ff.create_new("p2")

    e_f_lower = f_ff.create_new("e_f_lower")
    e_f_upper = f_ff.create_new("e_f_upper")

    e = sp.Symbol("e")
    create_kernel_func(
        sfg,
        [
            ps.Assignment(e, p1.center() * e_f_lower.center() + p2.center() * e_f_upper.center()),
            ps.Assignment(e_f.center(), e),
            ps.Assignment(x_fine.center(), x_fine.center() + e),
        ],
        "MGSemi_FusedProlongInterpolationAndAxpy",
        allow_vect=False,
    )
