import pystencils as ps
import sympy as sp

from pystencilssfg import SourceFileGenerator, SfgConfig
from pystencilssfg.lang.cpp import *

with SourceFileGenerator() as sfg:
    dtype = sfg.context.project_info['float_precision']

    # set up kernel config
    def get_kernel_cfg(
            enable_vect: bool = True,
            enable_omp: bool = True
    ):
        if target := sfg.context.project_info['target']:
            kernel_cfg = ps.CreateKernelConfig(
                target=target,
            )

            # cpu optimizations
            if sfg.context.project_info['use_cpu']:

                # vectorization
                if target.is_vector_cpu() and enable_vect:
                    kernel_cfg.cpu.vectorize.enable = True
                    kernel_cfg.cpu.vectorize.assume_inner_stride_one = True

                # OpenMP
                if sfg.context.project_info['use_openmp'] and enable_omp:
                    kernel_cfg.cpu.openmp.enable = True

            return kernel_cfg

    def invoke(kernel):
        # TODO: invoke handling for CUDA
        # if sfg.context.project_info['use_cuda']:
        #     ...
        # else:
        #     ...
        return sfg.call(kernel)

    def create_kernel_func(assign, func_name):
        k = sfg.kernels.create(assign, f"{func_name}_gen", get_kernel_cfg())
        sfg.function(f"PyCodegen_{func_name}")(invoke(k))

    def create_kernel_func_no_vect(assign, func_name):
        k = sfg.kernels.create(assign, f"{func_name}_gen", get_kernel_cfg(enable_vect=False))
        sfg.function(f"PyCodegen_{func_name}")(invoke(k))

    # symbols

    a, b, c = sp.symbols("a, b, c")

    r = ps.TypedSymbol("r", dtype)

    # fields

    x, y, z, w = ps.fields(f"x, y, z, w: {dtype}[3D]", layout="fzyx")

    # kernels

    ## z = a * x + b * y (PFVLinearSum)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() + b * y.center()), "VLinearSum")

    ## z = c (PFVConstInit)
    create_kernel_func(ps.Assignment(z.center(), c), "VConstInit")

    ## z_i = x_i * y_i (PFVProd)
    create_kernel_func(ps.Assignment(z.center(), x.center() * y.center()), "VProd")

    ## z_i = x_i / y_i (PFVDiv)
    create_kernel_func(ps.Assignment(z.center(), x.center() / y.center()), "VDiv")

    ## z = c * x (PFVScale)
    create_kernel_func(ps.Assignment(z.center(), c * x.center()), "VScale")

    ## z_i = |x_i| (PFVAbs)
    create_kernel_func(ps.Assignment(z.center(), sp.Abs(x.center())), "VAbs")

    ## z_i = 1 / x_i (PFVInv)
    create_kernel_func(ps.Assignment(z.center(), 1.0 / x.center()), "VInv")

    ## z_i = x_i + b (PFVAddConst)
    create_kernel_func(ps.Assignment(z.center(), x.center() + b), "VAddConst")

    ## Returns x dot y (PFVDotProd)
    create_kernel_func(ps.AddReductionAssignment(r, x.center() * y.center()), "VDotProd")

    ## Returns ||x||_{max} (PFVMaxNorm)
    create_kernel_func(ps.MaxReductionAssignment(r, sp.Max(x.center())), "VMaxNorm")

    ## Returns sum_i (x_i * w_i)^2 (PFVWrmsNormHelper)
    create_kernel_func(ps.AddReductionAssignment(r, x.center()**2 * w.center()**2), "VWrmsNormHelper")

    ## Returns sum_i |x_i| (PFVL1Norm)
    create_kernel_func(ps.AddReductionAssignment(r, sp.Abs(x.center())), "VL1Norm")

    ## Returns min_i x_i (PFVMin)
    create_kernel_func(ps.MinReductionAssignment(r, x.center()), "VMin")

    ## Returns max_i x_i (PFVMax)
    create_kernel_func(ps.MaxReductionAssignment(r, x.center()), "VMax")

    # TODO: Implement? Not ideal target code for pystencils
    #  PFVConstrProdPos(c, x)            Returns FALSE if some c_i = 0 &
    #                                        c_i*x_i <= 0.0

    ## z_i = (x_i > c)(PFVCompare)
    create_kernel_func_no_vect(
        ps.Assignment(z.center(),
                      sp.Piecewise((1.0, sp.Abs(x.center()) >= c), (0.0, True))),
        "VCompare")

    # TODO: Implement? Not ideal target code for pystencils
    #  PFVInvTest(x, z)                  Returns (x_i != 0 forall i), z_i = 1 / x_i

    ## y = x (PFVCopy)
    #create_kernel_func(ps.Assignment(y.center(), x.center()), "VCopy")

    ## z = x + y (PFVSum)
    create_kernel_func(ps.Assignment(z.center(), x.center() + y.center()), "VSum")

    ## z = x - y (PFVDiff)
    create_kernel_func(ps.Assignment(z.center(), x.center() - y.center()), "VDiff")

    ## z = - x (PFVNeg)
    create_kernel_func(ps.Assignment(z.center(), - x.center()), "VNeg")

    ## z = c * (x + y) (PFVScaleSum)
    create_kernel_func(ps.Assignment(z.center(), c * (x.center() + y.center())), "VScaleSum")

    ## z = c * (x - y) (PFVScaleDiff)
    create_kernel_func(ps.Assignment(z.center(), c * (x.center() - y.center())), "VScaleDiff")

    ## z = a * x + y (PFVLin1)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() + y.center()), "VLin1")

    ## z = a * x - y (PFVLin2)
    create_kernel_func(ps.Assignment(z.center(), a * x.center() - y.center()), "VLin2")

    ## y = y + a * x (PFVAxpy)
    create_kernel_func(ps.Assignment(y.center(), y.center() + a * x.center()), "VAxpy")

    ## x = x * a (PFVScaleBy)
    create_kernel_func(ps.Assignment(x.center(), x.center() * a), "VScaleBy")

    # TODO: Implement?
    #  PFVLayerCopy (a, b, x, y)        NBE: Extracts layer b from vector y, inserts into layer a of vector x