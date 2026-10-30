import pystencils as ps
import re
import textwrap

from pystencilssfg import SourceFileGenerator, AugExpr
from pystencilssfg.lang.gpu import cuda

from pystencils.codegen.properties import FieldBasePtr, FieldStride, FieldShape

from pystencils.types.quick import UInt, SInt
from pystencils.types import deconstify, PsPointerType, PsCustomType

# set up kernel config
def get_kernel_cfg(
    sfg: SourceFileGenerator,
    optimize: bool,
    allow_vect: bool,
    reduction: bool,
):
    if target := sfg.context.project_info["target"]:
        # gpus often lack hardware support for int64
        index_dtype = SInt(32) if sfg.context.project_info.get("use_cuda") else SInt(64)

        kernel_cfg = ps.CreateKernelConfig(
            target=target,
            index_dtype=index_dtype
        )

        if optimize:
            # cpu optimizations
            if sfg.context.project_info.get("use_cpu"):

                # vectorization
                if target.is_vector_cpu() and allow_vect:
                    kernel_cfg.cpu.vectorize.enable = True
                    kernel_cfg.cpu.vectorize.assume_inner_stride_one = True

                # OpenMP
                if sfg.context.project_info.get("use_openmp"):
                    kernel_cfg.cpu.openmp.enable = True

            # gpu optimizations
            if sfg.context.project_info.get("use_cuda"):
                if not reduction:
                    # use same default as in pf_cudaloops.h
                    kernel_cfg.gpu.default_block_size = (64, 4, 4)
                else:
                    # sets default block size for kernel invocation
                    if default_bs := sfg.context.project_info.get("default_block_size"):
                        kernel_cfg.gpu.default_block_size = default_bs

                    # get corresponding reduction configuration. defaults to atomics if all configurations are disabled
                    use_warp_reductions = sfg.context.project_info.get("use_warp_reductions")
                    use_shared_mem_reductions = sfg.context.project_info.get("use_shared_mem_reductions")
                    use_cub_reductions = sfg.context.project_info.get("use_cub_reductions")

                    # ensures that block sizes are divisible by warp size for faster reductions
                    if use_warp_reductions or use_shared_mem_reductions or use_cub_reductions:
                        kernel_cfg.gpu.assume_warp_aligned_block_size = True
                        kernel_cfg.gpu.warp_size = 32

                    # extend warp-level reductions with an additional shared memory reduction for further speedup
                    if use_shared_mem_reductions:
                        kernel_cfg.gpu.use_shared_mem_reductions = True

                    # use CUB back-end for fast reductions
                    if use_cub_reductions:
                        kernel_cfg.gpu.use_cub_reductions = True

                    # sets GPU indexing scheme
                    if indexing_scheme := sfg.context.project_info.get("gpu_indexing_scheme"):
                        if indexing_scheme in ("linear1d", "linear3d", "blockwise4d", "gridstrided_linear1d", "gridstrided_linear3d"):
                            kernel_cfg.gpu.indexing_scheme = indexing_scheme
                        else:
                            raise ValueError(f"Unsupported indexing scheme: {indexing_scheme}")

        return kernel_cfg
    else:
        raise ValueError("Target not specified in platform file.")


# vectorized cpu kernels assume a unit inner stride and drop the parameter _stride_X_0
def dropped_inner_strides(kernel):
    """Maps _stride_X_1 to _stride_X_0 for the inner strides missing from the kernel parameters.
    create_kernel_func adds them back to the generated function, so its signature does not depend on vectorization."""
    names = {p.name for p in kernel.parameters}
    dropped = {}
    for p in kernel.parameters:
        if (m := re.fullmatch("_stride_(.*)_1", p.name)) and f"_stride_{m[1]}_0" not in names:
            dropped[p.name] = f"_stride_{m[1]}_0"
    return dropped


def create_kernel_func(
        sfg: SourceFileGenerator,
        assign,
        func_name: str,
        optimize: bool = True,
        allow_vect: bool = True,
        reduction: bool = False,
        timing_index: bool = False,
        sync: bool = False
):
    """Generates the kernel and the function PyCodegen_<func_name> launching it."""
    target = sfg.context.project_info["target"]
    func_name = f"PyCodegen_{func_name}"
    kernel_name = f"{func_name}_gen"
    kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(sfg, optimize, allow_vect, reduction))

    timing_var = None
    if timing_index:
        sfg.include("parflow.h")
        timing_var = sfg.var("timing_index", SInt(32))

    def bracket_timing(*body):
        if timing_var is None:
            return list(body)
        return ["BeginTiming(timing_index);", *body, "EndTiming(timing_index);"]

    if target.is_cpu():
        if optimize and target.is_vector_cpu() and allow_vect:
            # extend parameter list with missing _stride_XYZ_0 parameters
            dropped = dropped_inner_strides(kernel)
            params = []
            for param in kernel.parameters:
                if param.name in dropped:
                    params += [sfg.var(dropped[param.name], SInt(64, const=True))]
                params += [param]

            if timing_var is not None:
                params += [timing_var]

            sfg.function(func_name).params(*params)(
                # TODO: mark _stride_XYZ_0 params as unused via void cast
                *bracket_timing(sfg.call(kernel))
            )
        else:
            # no extra handling needed -> just call the kernel
            if timing_var is not None:
                sfg.function(func_name).params(*kernel.parameters, timing_var)(
                    *bracket_timing(sfg.call(kernel))
                )
            else:
                sfg.function(func_name)(sfg.call(kernel))
    elif target.is_gpu() and sfg.context.project_info.get("use_cuda"):
        # invocation for GPUs with (potentially manual) specification of CUDA grid/block size
        kernel_call = [sfg.gpu_invoke(kernel)]

        # automatically extend kernel call with error handling
        sfg.include("<stdio.h>")

        error_check = [
            "cudaError_t err = cudaPeekAtLastError();",
            sfg.branch("err != cudaSuccess")(
                'printf("\\n\\n%s in %s at line %d\\n", cudaGetErrorString(err), __FILE__, __LINE__);\n'
                "exit(1);"
            ),
        ]
        synchronize = ["cudaStreamSynchronize(0);"] if sync else []

        if timing_var is not None:
            sfg.function(func_name).params(*kernel.parameters, timing_var)(
                *(bracket_timing(*kernel_call) + error_check + synchronize)
            )
        else:
            sfg.function(func_name)(
                *(kernel_call + error_check + synchronize)
            )
    else:
        ValueError(f"Invalid target {target}. Only (vector) CPU and CUDA targets are "
                   f"available for pystencils code generation.")

    return kernel


# Wrappers replacing the GrGeom loops of ParFlow.
#
# A wrapper takes a Subvector for every field of the kernel, named after the field pointer with a
# "_sub" suffix, and the remaining scalar parameters of the kernel. It fetches the field pointers at
# the start of the iteration space and derives the strides of the fields from their subvectors.

SUBVECTOR_PTR = PsPointerType(PsCustomType("Subvector"))


def kernel_func_name(kernel):
    """Name of the function that create_kernel_func generated for the kernel"""
    return kernel.name.removesuffix("_gen")


def field_ptr_params(kernel):
    return [p for p in kernel.parameters if p.wrapped.get_properties(FieldBasePtr)]


def scalar_params(kernel):
    """Kernel parameters that are neither field pointers nor field shapes or strides"""
    return [
        p for p in kernel.parameters
        if not (
            p.wrapped.get_properties(FieldBasePtr)
            or p.wrapped.get_properties(FieldStride)
            or p.wrapped.get_properties(FieldShape)
        )
    ]


def grgeom_wrapper_params(sfg: SourceFileGenerator, kernel, extra=(), exclude=(), timing_index: bool = False):
    """Parameters of a wrapper: the GrGeom loop parameters (gr_domain, r, ix, iy, iz, nx, ny, nz), the extra
    parameters, a Subvector per field, the scalar kernel parameters not in exclude and the timing index"""
    params = [sfg.var("gr_domain", PsPointerType(PsCustomType("GrGeomSolid"))), sfg.var("r", SInt(32))]
    params += [sfg.var(f"i{d}", SInt(32)) for d in "xyz"]
    params += [sfg.var(f"n{d}", SInt(32)) for d in "xyz"]
    params += list(extra)
    params += [sfg.var(f"{p.name}_sub", SUBVECTOR_PTR) for p in field_ptr_params(kernel)]
    params += [p for p in scalar_params(kernel) if p.name not in exclude]
    if timing_index:
        params += [sfg.var("timing_index", SInt(32))]
    return params


def fetch_field_ptrs(kernel, start, starts=None):
    """C code declaring the field pointers of the kernel at the element start = (x, y, z) of their subvectors.
    starts maps field pointer names to a different start of that field."""
    starts = starts or {}
    return "\n".join(
        f"double* {p.name} = SubvectorElt({p.name}_sub, {', '.join(starts.get(p.name, start))});"
        for p in field_ptr_params(kernel)
    )


def kernel_call(kernel, sizes, strides=None, timing_index: bool = False):
    """C call of the kernel on an iteration space of the given sizes = (x, y, z).

    Fields of the same FieldFactory share their stride symbols, these are computed from the subvector of the
    first field using them. strides maps stride symbol names to a different C expression."""
    strides = dict(strides or {})
    for p in field_ptr_params(kernel):
        sub = f"{p.name}_sub"
        for base_ptr in p.wrapped.get_properties(FieldBasePtr):
            for d, stride in enumerate(base_ptr.field.strides):
                strides.setdefault(stride.name, ("1", f"SubvectorNX({sub})", f"SubvectorNX({sub}) * SubvectorNY({sub})")[d])

    # arguments in the parameter order of the function generated by create_kernel_func
    dropped = dropped_inner_strides(kernel)
    args = []
    for p in kernel.parameters:
        if p.name in dropped:
            args += [strides[dropped[p.name]]]
        if shapes := p.wrapped.get_properties(FieldShape):
            args += [sizes[next(iter(shapes)).coordinate]]
        elif p.wrapped.get_properties(FieldStride):
            args += [strides[p.name]]
        else:
            args += [p.name]

    call = f"{kernel_func_name(kernel)}({', '.join(args)});"
    if timing_index:
        call = f"BeginTiming(timing_index);\n{call}\nEndTiming(timing_index);"
    return call


def indent(code: str, spaces: int):
    """Indents all but the first line of code, which is placed inside an indented f-string"""
    return textwrap.indent(code, " " * spaces).lstrip()


def grgeom_boxes_branch(sfg: SourceFileGenerator, code: str):
    """Runs code if the loops of gr_domain are given by its interior boxes, aborts for mesh refinement"""
    return sfg.branch("r == 0 && GrGeomSolidInteriorBoxes(gr_domain)")(code)(
        """
    printf(\"\\n\\nPystencils support unavailable for mesh refinement at file %s and line %d\\n\", __FILE__, __LINE__);
    exit(1);"""
    )


# wrapper running a kernel on the interior boxes of a GrGeomSolid, i.e. a replacement for GrGeomInLoop
def create_grgeom_in_loop_wrapper(sfg: SourceFileGenerator, kernel, timing_index: bool = False):
    sfg.include("parflow.h")

    sizes = [f"PV_i{d}u - PV_i{d}l + 1" for d in "xyz"]

    code = grgeom_boxes_branch(
        sfg,
        f"""
int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;
int *PV_visiting = NULL;
PF_UNUSED(PV_visiting);
BoxArray *boxes = GrGeomSolidInteriorBoxes(gr_domain);
for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) {{
    Box box = BoxArrayGetBox(boxes, PV_box);
    /* find octree and region intersection */
    PV_ixl = pfmax(ix, box.lo[0]);
    PV_iyl = pfmax(iy, box.lo[1]);
    PV_izl = pfmax(iz, box.lo[2]);
    PV_ixu = pfmin((ix + nx - 1), box.up[0]);
    PV_iyu = pfmin((iy + ny - 1), box.up[1]);
    PV_izu = pfmin((iz + nz - 1), box.up[2]);

    {indent(fetch_field_ptrs(kernel, ("PV_ixl", "PV_iyl", "PV_izl")), 4)}

    if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu) {{
        {indent(kernel_call(kernel, sizes, timing_index=timing_index), 8)}
    }}
}}
    """,
    )

    sfg.function(f"{kernel_func_name(kernel)}_wrapper").params(*grgeom_wrapper_params(sfg, kernel, timing_index=timing_index))(
        code,
    )
