import operator
from functools import reduce

import pystencils as ps
import re

from pystencilssfg import SourceFileGenerator
from pystencilssfg.lang.gpu import cuda

from pystencils.types.quick import SInt

# set up kernel config
def get_kernel_cfg(
    sfg: SourceFileGenerator,
    allow_vect: bool,
):
    if target := sfg.context.project_info["target"]:
        kernel_cfg = ps.CreateKernelConfig(
            target=target,
        )

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
                if indexing_scheme in ("linear1d", "linear3d", "gridstrided_linear3d"):
                    kernel_cfg.gpu.indexing_scheme = indexing_scheme
                else:
                    raise ValueError(f"Unsupported indexing scheme: {indexing_scheme}")

            # makes user responsible for setting GPU block/grid size.
            # this can be well combined with the "gridstrided_linear3d" indexing scheme
            if sfg.context.project_info.get("use_manual_exec_cfg_cuda"):
                kernel_cfg.gpu.manual_launch_grid = True

        return kernel_cfg
    else:
        raise ValueError("Target not specified in platform file.")


def create_kernel_func(
    sfg: SourceFileGenerator, assign, func_name: str, allow_vect: bool = True
):
    target = sfg.context.project_info["target"]
    func_name = f"PyCodegen_{func_name}"
    kernel_name = f"{func_name}_gen"
    kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(sfg, allow_vect))

    if target.is_cpu():
        if target.is_vector_cpu() and allow_vect:
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
        sfg.include("<stdio.h>")

        if sfg.context.project_info.get("use_manual_exec_cfg_cuda"):
            block_size = cuda.dim3(const=True).var("block_size")
            grid_size = cuda.dim3(const=True).var("grid_size")

            # get manual block and grid sizes from user configuration
            default_bs = sfg.context.project_info.get("default_block_size")
            manual_gs = sfg.context.project_info.get("manual_grid_size")

            kernel_call = [
                sfg.init(block_size)(*[str(bs) for bs in default_bs]),
                sfg.init(grid_size)(*[str(bs) for bs in manual_gs]),
                sfg.gpu_invoke(kernel, block_size=block_size, grid_size=grid_size)
            ]
        else:
            kernel_call = [sfg.gpu_invoke(kernel)]

        # automatically extend kernel call with error handling
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
