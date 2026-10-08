import pystencils as ps
import re

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


def create_kernel_func(
        sfg: SourceFileGenerator,
        assign,
        func_name: str,
        optimize: bool = True,
        allow_vect: bool = True,
        reduction: bool = False,
        timing_index: bool = False
):
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
            params = []
            missing_strides = []
            for i, param in enumerate(kernel.parameters):
                pattern = re.compile("_stride_(.*)_1")
                match = pattern.findall(param.name)

                if match:
                    stride = sfg.var(f"_stride_{match[0]}_0", SInt(64, const=True))
                    params += [stride]
                    missing_strides += [stride]
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

        if timing_var is not None:
            sfg.function(func_name).params(*kernel.parameters, timing_var)(
                *(bracket_timing(*kernel_call) + [
                    "cudaError_t err = cudaPeekAtLastError();",
                    sfg.branch("err != cudaSuccess")(
                        'printf("\\n\\n%s in %s at line %d\\n", cudaGetErrorString(err), __FILE__, __LINE__);\n'
                        "exit(1);"
                    ),
                ])
            )
        else:
            sfg.function(func_name)(
                *(kernel_call + [
                    "cudaError_t err = cudaPeekAtLastError();",
                    sfg.branch("err != cudaSuccess")(
                        'printf("\\n\\n%s in %s at line %d\\n", cudaGetErrorString(err), __FILE__, __LINE__);\n'
                        "exit(1);"
                    ),
                ])
            )
    else:
        ValueError(f"Invalid target {target}. Only (vector) CPU and CUDA targets are "
                   f"available for pystencils code generation.")

    return kernel


# wrapper running a kernel on the interior boxes of a GrGeomSolid, i.e. a replacement for GrGeomInLoop
def create_grgeom_in_loop_wrapper(sfg: SourceFileGenerator, kernel, timing_index: bool = False):
    params = []

    params += [sfg.var("gr_domain", PsPointerType(PsCustomType("GrGeomSolid")))]
    params += [sfg.var("r", SInt(32))]
    params += [sfg.var(f"i{d}", SInt(32)) for d in ["x", "y", "z"]]
    params += [sfg.var(f"n{d}", SInt(32)) for d in ["x", "y", "z"]]

    fetch_subvectors = []

    # fields of the same FieldFactory share their stride symbols -> fetch them from the first such field
    stride_subvectors = {}

    for param in kernel.parameters:
        if base_ptrs := param.wrapped.get_properties(FieldBasePtr):
            fieldname = param.name
            fieldname_sub = f"{fieldname}_sub"

            params += [sfg.var(fieldname_sub, PsPointerType(PsCustomType("Subvector")))]

            fetch_subvectors += [
                f"double* {fieldname} = SubvectorElt({fieldname_sub}, PV_ixl, PV_iyl, PV_izl);\n"
            ]

            for base_ptr in base_ptrs:
                for stride in base_ptr.field.strides:
                    stride_subvectors.setdefault(stride.name, fieldname_sub)
        elif not (
            param.wrapped.get_properties(FieldStride)
            or param.wrapped.get_properties(FieldShape)
        ):
            params += [param]

    # kernel arguments in parameter order: field pointers, sizes and strides of the box, and the remaining symbols
    args = []
    for param in kernel.parameters:
        if param.wrapped.get_properties(FieldBasePtr):
            args += [param.name]
        elif shapes := param.wrapped.get_properties(FieldShape):
            d = "xyz"[next(iter(shapes)).coordinate]
            args += [f"PV_i{d}u - PV_i{d}l + 1"]
        elif strides := param.wrapped.get_properties(FieldStride):
            sub = stride_subvectors[param.name]
            args += [("1", f"SubvectorNX({sub})", f"SubvectorNX({sub}) * SubvectorNY({sub})")[next(iter(strides)).coordinate]]
        else:
            args += [param.name]

    sfg.include("parflow.h")

    timing_begin = ""
    timing_end = ""
    if timing_index:
        params += [sfg.var("timing_index", SInt(32))]
        timing_begin = "BeginTiming(timing_index);"
        timing_end = "EndTiming(timing_index);"

    code = sfg.branch("r == 0 && GrGeomSolidInteriorBoxes(gr_domain)")(
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

    {"    ".join(fetch_subvectors)}

    if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu) {{
        {timing_begin}
        {kernel.name[:-4]}(
            {", ".join(args)}
        );
        {timing_end}
    }}
}}
    """
    )(
        """
    printf(\"\\n\\nPystencils support unavailable for mesh refinement at file %s and line %d\\n\", __FILE__, __LINE__);
    exit(1);"""
    )

    sfg.function(f"{kernel.name[:-4]}_wrapper").params(*params)(
        code,
    )
