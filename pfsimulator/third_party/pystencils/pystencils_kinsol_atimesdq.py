import sympy as sp

from pystencils import Kernel
from pystencils.types.quick import Fp, SInt
from pystencils.types import PsPointerType

from pystencils_codegen import *


def create_multi_reduction_kernel_wrapper(
    sfg: SourceFileGenerator,
    kernel: Kernel,
    output_names: dict,
    optimize: bool,
    allow_vect: bool,
    has_init_val: bool,
    timing_index: bool = False
):
    kernel_params = [
        pw
        for pw in kernel.parameters
        if pw.wrapped.is_field_parameter or not isinstance(pw.dtype, PsPointerType)
    ]

    # the reduction accumulators are exactly the pointer parameters that are not tied to
    # a field; order them to match output_names via their (unique) parameter names
    reduction_params = [
        pw
        for pw in kernel.parameters
        if not pw.wrapped.is_field_parameter and isinstance(pw.dtype, PsPointerType)
    ]

    assert len(reduction_params) == len(output_names), (
        f"Expected {len(output_names)} reduction accumulators, found {len(reduction_params)}"
    )

    output_names = [output_names[pw.name] for pw in reduction_params]

    params = []
    args = []

    # TODO: code duplication (see create_reduction_kernel_wrapper)
    target = sfg.context.project_info["target"]
    use_cuda = sfg.context.project_info.get("use_cuda")

    if target.is_vector_cpu() and optimize and allow_vect:
        for param in kernel_params:
            pattern = re.compile("_stride_(.*)_1")
            match = pattern.findall(param.name)

            if match:
                stridename = f"_stride_{match[0]}_0"
                stride = sfg.var(stridename, SInt(64, const=True))
                params += [stride]
                args += [stridename]
            params += [param]
            args += [param.wrapped.name]
    else:
        for param in kernel_params:
            params += [param]
            args += [param.wrapped.name]

    sfg.include("parflow.h")
    if use_cuda:
        sfg.include("pf_cudamalloc.h")

    rptr_names = [f"reduction_writeback_ptr_{name}" for name in output_names]

    alloc_lines = []
    prefetch_h2d_lines = []
    prefetch_d2h_lines = []
    init_lines = []
    dealloc_lines = []
    writeback_lines = []

    for rptr_name, out_name in zip(rptr_names, output_names):
        alloc = f"double* {rptr_name} = "
        if use_cuda:
            alloc += "(double*)_talloc_device(sizeof(double))"
        else:
            alloc += "talloc(double, 1)"
        alloc_lines += [f"{alloc};"]

        if use_cuda:
            prefetch_d2h_lines += [f"MemPrefetchDeviceToHost_cuda({rptr_name}, sizeof(double), 0);"]

        if has_init_val:
            initval_name = f"initval_{out_name}"
            params += [sfg.var(initval_name, Fp(64, const=True))]
            init_lines += [f"*{rptr_name} = {initval_name};"]
        else:
            init_lines += [f"*{rptr_name} = 0.0;"]

        if use_cuda:
            prefetch_h2d_lines += [f"MemPrefetchHostToDevice_cuda({rptr_name}, sizeof(double), 0);"]

        writeback_lines += [f"*{out_name} = *{rptr_name};"]

        if use_cuda:
            dealloc_lines += [f"_tfree_device({rptr_name});"]
        else:
            dealloc_lines += [f"tfree({rptr_name});"]

    # one output pointer per reduction result -- the wrapper returns void
    for out_name in output_names:
        params += [sfg.var(out_name, PsPointerType(Fp(64)))]

    timing_begin = ""
    timing_end = ""
    if timing_index:
        params += [sfg.var("timing_index", SInt(32))]
        timing_begin = "BeginTiming(timing_index);"
        timing_end = "EndTiming(timing_index);"

    code = f"""
    {"".join(alloc_lines)}

    {"".join(prefetch_d2h_lines)}
    {"".join(init_lines)}
    {"".join(prefetch_h2d_lines)}

    {timing_begin}
    {kernel.name[:-4]}(
        {", ".join(args)}, {", ".join(rptr_names)}
    );
    {timing_end}

    {"cudaStreamSynchronize(0);" if use_cuda else ""}

    {"".join(prefetch_d2h_lines)}

    {"".join(writeback_lines)}
    {"".join(dealloc_lines)}
"""

    sfg.function(f"{kernel.name[:-4]}_wrapper").params(*params)(
        code,
    )


def create_kernel_func_and_multi_reduction_wrapper(
    sfg: SourceFileGenerator,
    assignments,
    func_name: str,
    output_names: dict,
    optimize: bool = True,
    allow_vect: bool = True,
    has_init_val: bool = False,
    timing_index: bool = False,
):
    # create kernel func
    kernel = create_kernel_func(sfg, assignments, func_name, optimize, allow_vect, reduction=True)

    # create multi-reduction wrapper func
    create_multi_reduction_kernel_wrapper(
        sfg, kernel, output_names, optimize, allow_vect, has_init_val, timing_index
    )


with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info["default_dtype"]

    # symbols

    r_sutsv = ps.TypedSymbol("r_sutsv", default_dtype)
    r_vtv = ps.TypedSymbol("r_vtv", default_dtype)
    r_sq1norm = ps.TypedSymbol("r_sq1norm", default_dtype)

    # fields

    uu, vv, uscale = ps.fields(f"uu, vv, uscale: {default_dtype}[3D]", layout="fzyx")

    # scaled quantities shared by all three reductions below
    su = uu.center() * uscale.center()
    sv = vv.center() * uscale.center()

    # Fused reduction kernel for KINSpgmrAtimesDQ (kinsol/kinspgmr.c). Computes in a single pass over the grid:
    #
    #   sutsv   = (Du * uu) . (Du * v)      dot product
    #   vtv     = (Du * v)  . (Du * v)      dot product (norm^2)
    #   sq1norm = || Du * v ||_1            L1 norm
    #
    # where Du = uscale.
    create_kernel_func_and_multi_reduction_wrapper(
        sfg,
        [
            ps.AddReductionAssignment(r_sutsv, su * sv),
            ps.AddReductionAssignment(r_vtv, sv * sv),
            ps.AddReductionAssignment(r_sq1norm, sp.Abs(sv)),
        ],
        "KINAtimesDQFusedReduce",
        {"r_sutsv": "sutsv", "r_vtv": "vtv", "r_sq1norm": "sq1norm"},
        timing_index=True,
    )
