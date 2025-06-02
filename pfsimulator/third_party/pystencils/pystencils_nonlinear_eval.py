import sympy as sp

from pystencilssfg import SourceFileGenerator

from pystencils_codegen import *

with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info['default_dtype']

    # symbols

    vol = sp.symbols("vol")

    # constants

    del_x_slope = 1.0
    del_y_slope = 1.0

    # fields

    z_mult_dat, dp, odp, sp, pp, opp, osp, pop, fp = ps.fields(
        f"z_mult_dat, dp, odp, sp, pp, opp, osp, pop, fp: {default_dtype}[3D]",
        layout="fzyx"
    )

    # kernels

    # flux base
    # fp[ip] = (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol * del_x_slope * del_y_slope * z_mult_dat[ip]

    create_kernel_func(sfg,
                       ps.Assignment(
                           fp.center(),
                           (sp.center() * dp.center() - osp.center() * odp.center()) *
                                pop.center() * vol * del_x_slope * del_y_slope * z_mult_dat.center()
                       ),
                       "Flux_Base")
