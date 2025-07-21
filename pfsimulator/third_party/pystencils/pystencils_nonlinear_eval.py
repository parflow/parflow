import sympy as sp

from pystencilssfg import SourceFileGenerator

from pystencils import Kernel
from pystencils.codegen.properties import FieldBasePtr, FieldStride, FieldShape
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

    fieldnames = []
    fetch_subvectors = []
    fetch_sizes = []

    kernel_symbols = []

    fieldstrides = []
    for param in kernel.parameters:
        if param.wrapped.get_properties(FieldBasePtr):
            fieldname = param.name
            fieldname_sub = f"{fieldname}_sub"

            fieldnames += [fieldname]

            params += [sfg.var(fieldname_sub, PsPointerType(PsCustomType("Subvector")))]

            fetch_subvectors += [f"double* {fieldname} = SubvectorElt({fieldname_sub}, PV_ixl, PV_iyl, PV_izl);\n"]

            nx, ny, nz = f"nx_{fieldname}", f"ny_{fieldname}", f"nz_{fieldname}"
            fetch_sizes += [f"const int {nx} = SubvectorNX({fieldname_sub});\n"]
            fetch_sizes += [f"const int {ny} = SubvectorNY({fieldname_sub});\n"]
            fetch_sizes += [f"const int {nz} = SubvectorNZ({fieldname_sub});\n"]

            fieldstrides += ["1", f"{nx}", f"{nx} * {ny}"]
        elif not (param.wrapped.get_properties(FieldStride) or param.wrapped.get_properties(FieldShape)):
            params += [param]
            kernel_symbols += [param.name]

    sfg.include("parflow.h")

    code = sfg.branch("r == 0 && GrGeomSolidInteriorBoxes(gr_domain)")(f"""
int PV_ixl, PV_iyl, PV_izl, PV_ixu, PV_iyu, PV_izu;
int *PV_visiting = NULL;
PF_UNUSED(PV_visiting);
BoxArray *boxes = GrGeomSolidInteriorBoxes(gr_domain);
for (int PV_box = 0; PV_box < BoxArraySize(boxes); PV_box++) {{
    Box box = BoxArrayGetBox(boxes, PV_box);
    /* find octree and region intersection */
    PV_ixl = pfmax(ix, box.lo[0]);
    PV_iyl = pfmax(iy, box.lo[1]);
    PV_izl = pfmax(iz, box.lo[2]);
    PV_ixu = pfmin((ix + nx - 1), box.up[0]);
    PV_iyu = pfmin((iy + ny - 1), box.up[1]);
    PV_izu = pfmin((iz + nz - 1), box.up[2]);

    {"    ".join(fetch_subvectors)}

    {"    ".join(fetch_sizes)}

    if (PV_ixl <= PV_ixu && PV_iyl <= PV_iyu && PV_izl <= PV_izu) {{
        {kernel.name[:-4]}(
            {", ".join(fieldnames)},
            PV_ixu - PV_ixl + 1, PV_iyu - PV_iyl + 1, PV_izu - PV_izl + 1,
            {", ".join(fieldstrides)},
            {", ".join(kernel_symbols)}
        );
    }}
}}
    """)


    sfg.function(f"{kernel.name}_wrapper").params(*params)(
        code,
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

