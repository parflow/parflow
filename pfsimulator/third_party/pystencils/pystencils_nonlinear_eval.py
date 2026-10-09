import sympy as sp

from pystencils import TypedSymbol
from pystencils import DynamicType

from field_factory import FieldFactory
from pystencils_codegen import *


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

    vol, dt = sp.symbols("vol, dt")

    # constants

    del_x_slope = 1.0
    del_y_slope = 1.0

    # iteration space

    nx = TypedSymbol("_size_0", DynamicType.INDEX_TYPE)
    ny = TypedSymbol("_size_1", DynamicType.INDEX_TYPE)
    nz = TypedSymbol("_size_2", DynamicType.INDEX_TYPE)

    # field strides

    f_sx = TypedSymbol("_stride_f_0", DynamicType.INDEX_TYPE)
    f_sy = TypedSymbol("_stride_f_1", DynamicType.INDEX_TYPE)
    f_sz = TypedSymbol("_stride_f_2", DynamicType.INDEX_TYPE)

    po_sx = TypedSymbol("_stride_po_0", DynamicType.INDEX_TYPE)
    po_sy = TypedSymbol("_stride_po_1", DynamicType.INDEX_TYPE)
    po_sz = TypedSymbol("_stride_po_2", DynamicType.INDEX_TYPE)

    f_ff = FieldFactory((nx, ny, nz), (f_sx, f_sy, f_sz))
    po_ff = FieldFactory((nx, ny, nz), (po_sx, po_sy, po_sz))

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
