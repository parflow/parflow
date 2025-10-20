import sympy as sp

from pystencils import Kernel
from pystencils import TypedSymbol
from pystencils import DynamicType
from pystencils.codegen.properties import FieldBasePtr, FieldStride, FieldShape
from pystencils.types import PsPointerType, PsCustomType
from pystencils.types.quick import Fp, SInt

from field_factory import FieldFactory
from pystencils_codegen import *


def create_kernel_wrapper(kernel: Kernel):
    params = []

    params += [sfg.var("gr_domain", PsPointerType(PsCustomType("GrGeomSolid")))]
    params += [sfg.var("r", SInt(32))]
    params += [sfg.var(f"i{d}", SInt(32)) for d in ["x", "y", "z"]]
    params += [sfg.var(f"n{d}", SInt(32)) for d in ["x", "y", "z"]]

    fieldnames = []
    fetch_subvectors = []

    kernel_symbols = []

    for param in kernel.parameters:
        if param.wrapped.get_properties(FieldBasePtr):
            fieldname = param.name
            fieldname_sub = f"{fieldname}_sub"

            fieldnames += [fieldname]

            params += [sfg.var(fieldname_sub, PsPointerType(PsCustomType("Subvector")))]

            fetch_subvectors += [
                f"double* {fieldname} = SubvectorElt({fieldname_sub}, PV_ixl, PV_iyl, PV_izl);\n"
            ]
        elif not (
            param.wrapped.get_properties(FieldStride)
            or param.wrapped.get_properties(FieldShape)
        ):
            params += [param]
            kernel_symbols += [param.name]

        # fetch common field sizes and assemble field strides
        fetch_sizes = []
        fieldstrides = []

        # all fields except 'pop' use same strides as 'f' field in replaced kernels -> omit duplicates
        for fieldname in ("_data_fp",) + (("_data_pop",) if any(p.name == "_data_pop" for p in kernel.parameters) else ()):
            fieldname_sub = f"{fieldname}_sub"
            nx, ny, nz = f"nx_{fieldname}", f"ny_{fieldname}", f"nz_{fieldname}"

            fetch_sizes += [f"const int {nx} = SubvectorNX({fieldname_sub});\n"]
            fetch_sizes += [f"const int {ny} = SubvectorNY({fieldname_sub});\n"]

            fieldstrides += ["1", f"{nx}", f"{nx} * {ny}"]

    sfg.include("parflow.h")

    code = sfg.branch("r == 0 && GrGeomSolidInteriorBoxes(gr_domain)")(
        f"""
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
    """
    )(
        """
    printf(\"\\n\\nPystencils support unavailable for mesh refinement at file %s and line %d\\n\", __FILE__, __LINE__);
    exit(1);"""
    )

    sfg.function(f"{kernel.name[:-4]}_wrapper").params(*params)(
        code,
    )


def create_kernel_func_and_wrapper(
    sfg: SourceFileGenerator, assign, func_name: str, allow_vect: bool = True
):
    # create kernel func
    kernel = create_kernel_func(sfg, assign, func_name, allow_vect)

    # create wrapper func
    create_kernel_wrapper(kernel)


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

    z_mult_dat, dp, odp, sp, pp, opp, osp, fp, ss, et = [f_ff.create_new(name) for name in
                                                         ["z_mult_dat", "dp", "odp", "sp", "pp", "opp", "osp", "fp",
                                                          "ss", "et"]]

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
    )
