import sympy as sp

import pystencils as ps
import re

from pystencilssfg import SourceFileGenerator
from pystencilssfg.lang.gpu import cuda

with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info['default_dtype']
    target = sfg.context.project_info['target']

    block_size = cuda.dim3().var("blockSize")

    # set up kernel config
    def get_kernel_cfg(
            allow_vect: bool,
    ):
        if target:
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

            return kernel_cfg


    def invoke(kernel):
        if sfg.context.project_info['use_cuda']:
            return sfg.gpu_invoke(kernel)
        else:
            return sfg.call(kernel)


    def create_kernel_func(assign, func_name: str, allow_vect: bool = True):
        func_name = f"PyCodegen_{func_name}"
        kernel_name = f"{func_name}_gen"
        kernel = sfg.kernels.create(assign, kernel_name, get_kernel_cfg(allow_vect))
        if target.is_vector_cpu() and allow_vect:
            # extend parameter list with missing _stride_XYZ_0 parameters
            params = []
            missing_strides = []
            for i, param in enumerate(kernel.parameters):
                pattern = re.compile('_stride_(.*)_1')
                match = pattern.findall(param.name)

                if match:
                    stride = sfg.var(f"_stride_{match[0]}_0", "int64")
                    params += [stride]
                    missing_strides += [stride]
                params += [param]

            sfg.function(func_name).params(*params)(
                # TODO: mark _stride_XYZ_0 params as unused via void cast
                invoke(kernel)
            )
        else:
            # no extra handling needed -> just invoke the kernel
            sfg.function(func_name)(invoke(kernel))


    # symbols

    a, b, c = sp.symbols("a, b, c")

    r = ps.TypedSymbol("r", default_dtype)

    # fields

    x, y, w = ps.fields(f"x, y, w: {default_dtype}[3D]", layout="fzyx")
    z = ps.fields(f"z: {default_dtype}[3D]", layout="fzyx")

    # kernels

    # z = a * x + b * y (PFVLinearSum)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() + b * y.center()), "VLinearSum")

    # z = c (PFVConstInit)
    create_kernel_func(ps.Assignment(z.center(), c), "VConstInit")

    # z_i = x_i * y_i (PFVProd)
    create_kernel_func(ps.Assignment(z.center(), x.center() * y.center()), "VProd")

    # z_i = x_i / y_i (PFVDiv)
    create_kernel_func(ps.Assignment(z.center(), x.center() / y.center()), "VDiv")

    # z = c * x (PFVScale)
    create_kernel_func(ps.Assignment(z.center(), c * x.center()), "VScale")

    # z_i = |x_i| (PFVAbs)
    create_kernel_func(ps.Assignment(z.center(), sp.Abs(x.center())), "VAbs")

    # z_i = 1 / x_i (PFVInv)
    create_kernel_func(ps.Assignment(z.center(), 1.0 / x.center()), "VInv")

    # z_i = x_i + b (PFVAddConst)
    create_kernel_func(ps.Assignment(z.center(), x.center() + b), "VAddConst")

    # Returns x dot y (PFVDotProd)
    create_kernel_func(ps.AddReductionAssignment(r, x.center() * y.center()), "VDotProd")

    # Returns ||x||_{max} (PFVMaxNorm)
    create_kernel_func(ps.MaxReductionAssignment(r, sp.Max(x.center())), "VMaxNorm")

    # Returns sum_i (x_i * w_i)^2 (PFVWrmsNormHelper)
    create_kernel_func(ps.AddReductionAssignment(r, x.center() ** 2 * w.center() ** 2), "VWrmsNormHelper")

    # Returns sum_i |x_i| (PFVL1Norm)
    create_kernel_func(ps.AddReductionAssignment(r, sp.Abs(x.center())), "VL1Norm")

    # Returns min_i x_i (PFVMin)
    create_kernel_func(ps.MinReductionAssignment(r, x.center()), "VMin")

    # Returns max_i x_i (PFVMax)
    create_kernel_func(ps.MaxReductionAssignment(r, x.center()), "VMax")

    # TODO: Implement? Not ideal target code for pystencils
    #  PFVConstrProdPos(c, x)            Returns FALSE if some c_i = 0 &
    #                                        c_i*x_i <= 0.0

    # z_i = (x_i > c)(PFVCompare)
    create_kernel_func(
        ps.Assignment(z.center(),
                      sp.Piecewise((1.0, sp.Abs(x.center()) >= c), (0.0, True))),
        "VCompare", allow_vect=False)

    # TODO: Implement? Not ideal target code for pystencils
    #  PFVInvTest(x, z)                  Returns (x_i != 0 forall i), z_i = 1 / x_i

    # y = x (PFVCopy)
    # create_kernel_func(ps.Assignment(y.center(), x.center()), "VCopy")

    # z = x + y (PFVSum)
    create_kernel_func(ps.Assignment(z.center(), x.center() + y.center()), "VSum")

    # z = x - y (PFVDiff)
    create_kernel_func(ps.Assignment(z.center(), x.center() - y.center()), "VDiff")

    # z = - x (PFVNeg)
    create_kernel_func(ps.Assignment(z.center(), - x.center()), "VNeg")

    # z = c * (x + y) (PFVScaleSum)
    create_kernel_func(ps.Assignment(z.center(), c * (x.center() + y.center())), "VScaleSum")

    # z = c * (x - y) (PFVScaleDiff)
    create_kernel_func(ps.Assignment(z.center(), c * (x.center() - y.center())), "VScaleDiff")

    # z = a * x + y (PFVLin1)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() + y.center()), "VLin1")

    # z = a * x - y (PFVLin2)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() - y.center()), "VLin2")

    # z = y + a * x (PFVAxpy)
    create_kernel_func(ps.Assignment(z.center(), y.center() + a * x.center()), "VAxpy")

    # z = x * a (PFVScaleBy)
    create_kernel_func(ps.Assignment(z.center(), x.center() * a), "VScaleBy")

    # TODO: Implement?
    #  PFVLayerCopy (a, b, x, y)        NBE: Extracts layer b from vector y, inserts into layer a of vector x