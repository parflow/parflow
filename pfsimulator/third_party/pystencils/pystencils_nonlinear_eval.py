import sympy

from pystencils import TypedSymbol
from pystencils import DynamicType

from field_factory import FieldFactory
from pystencils_codegen import *

# sympy is not imported as sp here, since the saturation field below is called sp


def stride_symbols(name: str):
    """Stride symbols shared by the fields of one FieldFactory"""
    return tuple(TypedSymbol(f"_stride_{name}_{d}", DynamicType.INDEX_TYPE) for d in range(3))


def create_kernel_func_and_wrapper(
    sfg: SourceFileGenerator, assign, func_name: str, optimize: bool = True, allow_vect: bool = True,
    timing_index: bool = False
):
    # create kernel func
    kernel = create_kernel_func(sfg, assign, func_name, optimize, allow_vect)

    # create wrapper func
    create_grgeom_in_loop_wrapper(sfg, kernel, timing_index)


with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info["default_dtype"]

    # symbols

    vol, dt = sympy.symbols("vol, dt")

    # constants

    del_x_slope = 1.0
    del_y_slope = 1.0

    # iteration space

    nx = TypedSymbol("_size_0", DynamicType.INDEX_TYPE)
    ny = TypedSymbol("_size_1", DynamicType.INDEX_TYPE)
    nz = TypedSymbol("_size_2", DynamicType.INDEX_TYPE)

    # field strides

    f_ff = FieldFactory((nx, ny, nz), stride_symbols("f"))
    po_ff = FieldFactory((nx, ny, nz), stride_symbols("po"))

    # field declarations

    z_mult_dat, dp, odp, sp, pp, opp, osp, fp, ss, et, src = [f_ff.create_new(name) for name in
                                                              ["z_mult_dat", "dp", "odp", "sp", "pp", "opp", "osp",
                                                               "fp", "ss", "et", "src"]]

    pop = po_ff.create_new("pop")

    # kernels

    # terms of Richards' equation (docs/user_manual/models.rst), integrated over the cell volume and time step,
    # shared by the fused and the unfused kernels

    # accumulation, d(S rho phi)/dt:
    # (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol * del_x_slope * del_y_slope * z_mult_dat[ip]
    accumulation = (
        (sp.center() * dp.center() - osp.center() * odp.center())
        * pop.center()
        * vol
        * del_x_slope
        * del_y_slope
        * z_mult_dat.center()
    )

    # compressible storage, S S_s dp/dt:
    # ss[ip] * vol * del_x_slope * del_y_slope * z_mult_dat[ip] * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip])
    compressible_storage = (
        ss.center()
        * vol
        * del_x_slope
        * del_y_slope
        * z_mult_dat.center()
        * (pp.center() * sp.center() * dp.center() - opp.center() * osp.center() * odp.center())
    )

    # source terms, Q (phase sources and evapotranspiration):
    # vol * del_x_slope * del_y_slope * z_mult_dat[ip] * dt * (phase_source[ip] + et[ip])
    def source_terms(phase_source):
        return vol * del_x_slope * del_y_slope * z_mult_dat.center() * dt * (phase_source.center() + et.center())

    # fp[ip] = accumulation

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), accumulation),
        "NlFunctionEval_Accumulation",
        timing_index=True,
    )

    # fp[ip] += compressible storage

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), fp.center() + compressible_storage),
        "NlFunctionEval_AddCompressibleStorage",
        timing_index=True,
    )

    # fused: fp[ip] = accumulation + compressible storage - source terms
    # src is the phase source vector, sp the saturation (both are sp in the unfused kernels)

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), accumulation + compressible_storage - source_terms(src)),
        "NlFunctionEval_FusedAccumulationAndSourceTerms",
        timing_index=True,
    )

    # fp[ip] -= source terms (sp is the phase source here)

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), fp.center() - source_terms(sp)),
        "NlFunctionEval_AddSourceTerms",
        timing_index=True,
    )
