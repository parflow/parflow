import sympy as sp

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
    if target:= sfg.context.project_info['target']:
        kernel_cfg = ps.CreateKernelConfig(
            target=target,
        )

        # cpu optimizations
        if sfg.context.project_info['use_cpu']:

            # vectorization
            if target.is_vector_cpu() and allow_vect:
                kernel_cfg.cpu.vectorize.enable = True
                kernel_cfg.cpu.vectorize.assume_inner_stride_one = True

            # OpenMP
            if sfg.context.project_info['use_openmp']:
                kernel_cfg.cpu.openmp.enable = True

        # gpu optimization: warp level reductions
        if sfg.context.project_info['use_cuda']:
            kernel_cfg.gpu.assume_warp_aligned_block_size = True
            kernel_cfg.gpu.warp_size = 32

        return kernel_cfg
    else:
        raise ValueError("Target not specified in platform file.")


def invoke(sfg: SourceFileGenerator, k):
    if sfg.context.project_info['use_cuda']:
        return sfg.gpu_invoke(k)
    else:
        return sfg.call(k)


def create_kernel_func(
        sfg: SourceFileGenerator,
        assign,
        func_name: str,
        allow_vect: bool = True
):
    target = sfg.context.project_info['target']
    func_name = f"PyCodegen_{func_name}"
    kernel_name = f"{func_name}_gen"
    kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(sfg, allow_vect))
    if target.is_vector_cpu() and allow_vect:
        # extend parameter list with missing _stride_XYZ_0 parameters
        params = []
        missing_strides = []
        for i, param in enumerate(kernel.parameters):
            pattern = re.compile('_stride_(.*)_1')
            match = pattern.findall(param.name)

            if match:
                stride = sfg.var(f"_stride_{match[0]}_0", SInt(64, const=True))
                params += [stride]
                missing_strides += [stride]
            params += [param]

        sfg.function(func_name).params(*params)(
            # TODO: mark _stride_XYZ_0 params as unused via void cast
            invoke(sfg, kernel)
        )
    else:
        # no extra handling needed -> just invoke the kernel
        sfg.function(func_name)(invoke(sfg, kernel))