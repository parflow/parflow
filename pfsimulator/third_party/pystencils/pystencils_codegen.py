import pystencils as ps
import re

from pystencilssfg import SourceFileGenerator, AugExpr
from pystencilssfg.lang.gpu import cuda

from pystencils.codegen.properties import FieldShape

from pystencils.types.quick import UInt, SInt
from pystencils.types import deconstify

# set up kernel config
def get_kernel_cfg(
    sfg: SourceFileGenerator,
    optimize: bool,
    allow_vect: bool,
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

            # gpu optimization: warp level reductions
            if sfg.context.project_info.get("use_cuda"):
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
    sfg: SourceFileGenerator, assign, func_name: str, optimize: bool = False, allow_vect: bool = True
):
    target = sfg.context.project_info["target"]
    func_name = f"PyCodegen_{func_name}"
    kernel_name = f"{func_name}_gen"
    kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(sfg, optimize, allow_vect))

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

            sfg.function(func_name).params(*params)(
                # TODO: mark _stride_XYZ_0 params as unused via void cast
                sfg.call(kernel)
            )
        else:
            # no extra handling needed -> just call the kernel
            sfg.function(func_name)(sfg.call(kernel))
    elif target.is_gpu() and sfg.context.project_info.get("use_cuda"):
        # invocation for GPUs with (potentially manual) specification of CUDA grid/block size
        kernel_call = [sfg.gpu_invoke(kernel)]

        # automatically extend kernel call with error handling
        sfg.include("<stdio.h>")
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
