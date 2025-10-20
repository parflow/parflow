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

    # field strides

    m_sx = TypedSymbol("_stride_m_0", DynamicType.INDEX_TYPE)
    m_sy = TypedSymbol("_stride_m_1", DynamicType.INDEX_TYPE)
    m_sz = TypedSymbol("_stride_m_2", DynamicType.INDEX_TYPE)

    v_sx = TypedSymbol("_stride_v_0", DynamicType.INDEX_TYPE)
    v_sy = TypedSymbol("_stride_v_1", DynamicType.INDEX_TYPE)
    v_sz = TypedSymbol("_stride_v_2", DynamicType.INDEX_TYPE)

    # field declarations

    m_ff = FieldFactory((nx, ny, nz), (m_sx, m_sy, m_sz))
    v_ff = FieldFactory((nx, ny, nz), (v_sx, v_sy, v_sz))

    ## subentries of matrix A
    A = [m_ff.create_new(f"a{i}") for i in range(7)]

    ## entries of solution field x offset by stencil
    x = [v_ff.create_new(f"x{i}") for i in range(7)]

    ## right-hand side b
    b = v_ff.create_new("b")

    # kernels

    ## zero-optimization kernel
    create_kernel_func(
        sfg,
        ps.Assignment(x[0].center(), b.center() / A[0].center()),
        "RBGS_ZeroOptimizationKernel",
        allow_vect=False,
    )

    ## regular 7pt kernel
    stencil_convolution = sum([xi.center() * ai.center() for xi, ai in zip(x[1:], A[1:])])
    create_kernel_func(
        sfg,
        ps.Assignment(x[0].center(), (b.center() - stencil_convolution) / A[0].center()),
        "RBGS_7PtKernel",
        allow_vect=False,
    )
