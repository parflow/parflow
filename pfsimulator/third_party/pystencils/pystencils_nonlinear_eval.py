import sympy as sp

from pystencilssfg import SourceFileGenerator

from pystencils_codegen import *

with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info['default_dtype']

    # symbols

    vol, dt = sp.symbols("vol, dt")

    # constants

    del_x_slope = 1.0
    del_y_slope = 1.0

    # fields

    z_mult_dat, dp, odp, sp, pp, opp, osp, pop, fp, ss, et = ps.fields(
        f"z_mult_dat, dp, odp, sp, pp, opp, osp, pop, fp, ss, et: {default_dtype}[3D]",
        layout="fzyx"
    )

    # kernels

    # flux: base
    # fp[ip] = (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol * del_x_slope * del_y_slope * z_mult_dat[ip]

    create_kernel_func(sfg,
                       ps.Assignment(
                           fp.center(),
                           (sp.center() * dp.center() - osp.center() * odp.center()) *
                                pop.center() * vol * del_x_slope * del_y_slope * z_mult_dat.center()
                       ),
                       "Flux_Base")

    # flux: add compressible storage
    # fp[ip] += ss[ip] * vol * del_x_slope * del_y_slope * z_mult_dat[ip] * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip])

    create_kernel_func(sfg,
                       ps.Assignment(
                           fp.center(),
                           fp.center() + (
                                   ss.center() * vol * del_x_slope * del_y_slope * z_mult_dat.center() *
                                       (pp.center() * sp.center() * dp.center() - opp.center() * osp.center() * odp.center())
                           )
                       ),
                       "Flux_AddCompressibleStorage")

    # flux: add source terms
    # fp[ip] -= vol * del_x_slope * del_y_slope * z_mult_dat[ip] * dt * (sp[ip] + et[ip])

    create_kernel_func(sfg,
                       ps.Assignment(
                           fp.center(),
                           fp.center() - (
                                   vol * del_x_slope * del_y_slope * z_mult_dat.center() * dt * (sp.center() + et.center())
                           )
                       ),
                       "Flux_AddSourceTerms")

