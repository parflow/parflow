import pystencils as ps
import re

from pystencilssfg import SourceFileGenerator
from pystencilssfg.lang.gpu import cuda

from pystencils.types.quick import SInt

DEFAULT_BLOCK_SIZE: tuple[int, int, int] = (256, 1, 1)


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
            kernel_cfg.gpu.assume_warp_aligned_block_size = True
            kernel_cfg.gpu.warp_size = 32
            kernel_cfg.gpu.default_block_size = DEFAULT_BLOCK_SIZE
            kernel_cfg.gpu.use_cub_reductions = True
            #kernel_cfg.gpu.use_shared_mem_reductions = True

            if sfg.context.project_info.get("use_manual_exec_cfg_cuda"):
                kernel_cfg.gpu.manual_launch_grid = True
                kernel_cfg.gpu.indexing_scheme = "gridstrided_linear3d"
            else:
                kernel_cfg.gpu.indexing_scheme = "linear3d"

        return kernel_cfg
    else:
        raise ValueError("Target not specified in platform file.")


def invoke(sfg: SourceFileGenerator, k):
    if sfg.context.project_info.get("use_cuda"):
        sfg.include("<stdio.h>")

        if sfg.context.project_info.get("use_manual_exec_cfg_cuda"):
            block_size = cuda.dim3(const=True).var("block_size")
            grid_size = cuda.dim3(const=True).var("grid_size")

            kernel_call = [
                sfg.init(block_size)(*[str(bs) for bs in DEFAULT_BLOCK_SIZE]),
                sfg.init(grid_size)("1", "24", "18"),
                sfg.gpu_invoke(k, block_size=block_size, grid_size=grid_size)
            ]
        else:
            kernel_call = [sfg.gpu_invoke(k)]

        return kernel_call + [
            "cudaError_t err = cudaPeekAtLastError();",
            sfg.branch("err != cudaSuccess")(
                'printf("\\n\\n%s in %s at line %d\\n", cudaGetErrorString(err), __FILE__, __LINE__);\n'
                "exit(1);"
            ),
        ]
    else:
        return [sfg.call(k)]


def create_kernel_func(
    sfg: SourceFileGenerator, assign, func_name: str, allow_vect: bool = True
):
    target = sfg.context.project_info["target"]
    func_name = f"PyCodegen_{func_name}"
    kernel_name = f"{func_name}_gen"
    kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(sfg, allow_vect))
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
            *invoke(sfg, kernel)
        )
    else:
        # no extra handling needed -> just invoke the kernel
        sfg.function(func_name)(*invoke(sfg, kernel))

    return kernel
