import sympy as sp

from pystencilssfg import SourceFileGenerator

from pystencils import Kernel
from pystencils.codegen.properties import FieldBasePtr
from pystencils.types import PsPointerType, PsCustomType
from pystencils.types.quick import Fp, SInt

from pystencils_codegen import *

def create_kernel_wrapper(
        kernel: Kernel
):
    params = []

    params += [sfg.var("gr_domain", PsPointerType(PsCustomType("GrGeomSolid")))]
    params += [sfg.var("r", SInt(32))]
    params += [sfg.var(f"i{d}", SInt(32)) for d in ["x", "y", "z"]]
    params += [sfg.var(f"n{d}", SInt(32)) for d in ["x", "y", "z"]]

    fetch_subvectors = []
    fetch_strides = []
    for param in kernel.parameters:
        if param.wrapped.get_properties(FieldBasePtr):
            fieldname = param.name
            fieldname_sub = f"{fieldname}_sub"

            params += [sfg.var(fieldname_sub, PsPointerType(PsCustomType("Subvector")))]

            fetch_subvectors += [f"""
        double* {fieldname} = SubvectorElt({fieldname_sub}, PV_ixl, PV_iyl, PV_izl);"""]

            fetch_strides += [f"""
        const int nx_{fieldname} = SubvectorNX({fieldname_sub});
        const int ny_{fieldname} = SubvectorNY({fieldname_sub});
        const int nz_{fieldname} = SubvectorNZ({fieldname_sub});"""]

    sfg.include("parflow.h")

    code = f"""
if (r == 0 && GrGeomSolidInteriorBoxes(gr_domain)) {{
    int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;
    int *PV_visiting = NULL;
    PF_UNUSED(PV_visiting);
    BoxArray *boxes = GrGeomSolidInteriorBoxes(gr_domain);
    for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) {{
        Box box = BoxArrayGetBox(boxes, PV_box);
        /* find octree and region intersection */
#ifdef PARFLOW_HAVE_CUDA
        PV_ixl = pfmax(ix, BoxArrayMinCell(boxes, 0));
        PV_iyl = pfmax(iy, BoxArrayMinCell(boxes, 1));
        PV_izl = pfmax(iz, BoxArrayMinCell(boxes, 2));
        PV_ixu = pfmin((ix + nx - 1), BoxArrayMaxCell(boxes, 0));
        PV_iyu = pfmin((iy + ny - 1), BoxArrayMaxCell(boxes, 1));
        PV_izu = pfmin((iz + nz - 1), BoxArrayMaxCell(boxes, 2));
#else
        PV_ixl = pfmax(ix, box.lo[0]);
        PV_iyl = pfmax(iy, box.lo[1]);
        PV_izl = pfmax(iz, box.lo[2]);
        PV_ixu = pfmin((ix + nx - 1), box.up[0]);
        PV_iyu = pfmin((iy + ny - 1), box.up[1]);
        PV_izu = pfmin((iz + nz - 1), box.up[2]);
#endif

{"".join(fetch_subvectors)}

{"".join(fetch_strides)}
    }}
}}
"""


    sfg.function(f"{kernel.name}_wrapper").params(*params)(
        code,
        # TODO: invoke kernel with corresponding args
        #sfg.branch("PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu")(
        #    *invoke(sfg, kernel)
        #)
    )


def create_kernel_func_and_wrapper(
        sfg: SourceFileGenerator,
        assign,
        func_name: str,
        allow_vect: bool = True
):
    # create kernel func
    kernel = create_kernel_func(sfg, assign, func_name, allow_vect)

    # create wrapper func
    create_kernel_wrapper(kernel)

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

    create_kernel_func_and_wrapper(sfg,
                       ps.Assignment(
                           fp.center(),
                           (sp.center() * dp.center() - osp.center() * odp.center()) *
                                pop.center() * vol * del_x_slope * del_y_slope * z_mult_dat.center()
                       ),
                       "Flux_Base")

    # flux: add compressible storage
    # fp[ip] += ss[ip] * vol * del_x_slope * del_y_slope * z_mult_dat[ip] * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip])

    create_kernel_func_and_wrapper(sfg,
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

    create_kernel_func_and_wrapper(sfg,
                       ps.Assignment(
                           fp.center(),
                           fp.center() - (
                                   vol * del_x_slope * del_y_slope * z_mult_dat.center() * dt * (sp.center() + et.center())
                           )
                       ),
                       "Flux_AddSourceTerms")

