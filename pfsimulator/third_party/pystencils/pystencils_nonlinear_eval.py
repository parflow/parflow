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

    # flux: base
    # fp[ip] = (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol * del_x_slope * del_y_slope * z_mult_dat[ip]

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(
            fp.center(),
            (sp.center() * dp.center() - osp.center() * odp.center())
            * pop.center()
            * vol
            * del_x_slope
            * del_y_slope
            * z_mult_dat.center(),
        ),
        "Flux_Base",
        timing_index=True,
    )

    # flux: add compressible storage
    # fp[ip] += ss[ip] * vol * del_x_slope * del_y_slope * z_mult_dat[ip] * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip])

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(
            fp.center(),
            fp.center()
            + (
                ss.center()
                * vol
                * del_x_slope
                * del_y_slope
                * z_mult_dat.center()
                * (
                    pp.center() * sp.center() * dp.center()
                    - opp.center() * osp.center() * odp.center()
                )
            ),
        ),
        "Flux_AddCompressibleStorage",
        timing_index=True,
    )

    # flux: fused base + compressible storage + source terms
    # src is the phase source vector, sp the saturation (both are sp in the unfused kernels)

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(
            fp.center(),
            (sp.center() * dp.center() - osp.center() * odp.center())
            * pop.center()
            * vol
            * del_x_slope
            * del_y_slope
            * z_mult_dat.center()
            + (
                ss.center()
                * vol
                * del_x_slope
                * del_y_slope
                * z_mult_dat.center()
                * (
                    pp.center() * sp.center() * dp.center()
                    - opp.center() * osp.center() * odp.center()
                )
            )
            - (
                vol
                * del_x_slope
                * del_y_slope
                * z_mult_dat.center()
                * dt
                * (src.center() + et.center())
            ),
        ),
        "Flux_FusedAccumulationAndSourceTerms",
        timing_index=True,
    )

    # flux: add source terms
    # fp[ip] -= vol * del_x_slope * del_y_slope * z_mult_dat[ip] * dt * (sp[ip] + et[ip])

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(
            fp.center(),
            fp.center()
            - (
                vol
                * del_x_slope
                * del_y_slope
                * z_mult_dat.center()
                * dt
                * (sp.center() + et.center())
            ),
        ),
        "Flux_AddSourceTerms",
        timing_index=True,
    )
