import sympy as sp

from pystencils import TypedSymbol
from pystencils import DynamicType

from field_factory import FieldFactory
from pystencils_codegen import *


def create_kernel_func_and_wrapper(sfg: SourceFileGenerator, assignments, func_name: str):
    kernel = create_kernel_func(sfg, assignments, func_name)
    create_grgeom_in_loop_wrapper(sfg, kernel)


with SourceFileGenerator() as sfg:
    default_dtype = sfg.context.project_info["default_dtype"]

    # iteration space

    nx = TypedSymbol("_size_0", DynamicType.INDEX_TYPE)
    ny = TypedSymbol("_size_1", DynamicType.INDEX_TYPE)
    nz = TypedSymbol("_size_2", DynamicType.INDEX_TYPE)

    # field strides

    v_sx = TypedSymbol("_stride_v_0", DynamicType.INDEX_TYPE)
    v_sy = TypedSymbol("_stride_v_1", DynamicType.INDEX_TYPE)
    v_sz = TypedSymbol("_stride_v_2", DynamicType.INDEX_TYPE)

    v_ff = FieldFactory((nx, ny, nz), (v_sx, v_sy, v_sz))

    # field declarations: pressure, density, relative permeability and saturation

    pp, pd, kr, sat = [v_ff.create_new(name) for name in ["pp", "pd", "kr", "sat"]]

    # van Genuchten parameters of the region

    alpha, n, gravity, s_res, s_dif = sp.symbols("vg_alpha, vg_n, gravity, s_res, s_dif")

    # common terms of the van Genuchten curves (problem_phase_rel_perm.c and problem_saturation.c).

    m, head, ah, ahn, opahn, opahnm, ahnm1 = sp.symbols("m, head, ah, ahn, opahn, opahnm, ahnm1")

    vg_terms = [
        ps.Assignment(m, 1.0 - 1.0 / n),
        ps.Assignment(head, sp.Abs(pp.center()) / (pd.center() * gravity)),
        ps.Assignment(ah, alpha * head),
        ps.Assignment(ahn, ah ** n),
        ps.Assignment(opahn, 1.0 + ahn),
        ps.Assignment(opahnm, opahn ** m),
        ps.Assignment(ahnm1, ahn / ah),
    ]

    saturated = sp.Ge(pp.center(), 0.0)

    # kernels

    # relative permeability (PhaseRelPerm, CALCFCN)
    # kr = (1 - ahnm1 / opahn^m)^2 / opahn^(m / 2)

    coeff = sp.Symbol("coeff")

    create_kernel_func_and_wrapper(
        sfg,
        vg_terms + [
            ps.Assignment(coeff, 1.0 - ahnm1 / opahnm),
            ps.Assignment(kr.center(), sp.Piecewise((1.0, saturated), (coeff * coeff / sp.sqrt(opahnm), True))),
        ],
        "VanGRelPerm",
    )

    # derivative of the relative permeability (PhaseRelPerm, CALCDER)
    # 2 * (coeff / opahn^(m / 2)) * ((n - 1) * ah^(n - 2) * alpha * opahn^(-m)
    #                                - ahnm1 * m * opahn^(-(m + 1)) * n * alpha * ahnm1)
    #   + coeff^2 * (m / 2) * opahn^(-(m + 2) / 2) * n * alpha * ahnm1

    sqrt_opahnm, inv_opahnm = sp.symbols("sqrt_opahnm, inv_opahnm")

    create_kernel_func_and_wrapper(
        sfg,
        vg_terms + [
            ps.Assignment(sqrt_opahnm, sp.sqrt(opahnm)),
            ps.Assignment(inv_opahnm, 1.0 / opahnm),
            ps.Assignment(coeff, 1.0 - ahnm1 * inv_opahnm),
            ps.Assignment(
                kr.center(),
                sp.Piecewise(
                    (0.0, saturated),
                    (
                        2.0 * (coeff / sqrt_opahnm)
                        * ((n - 1.0) * (ahnm1 / ah) * alpha * inv_opahnm
                           - ahnm1 * m * (inv_opahnm / opahn) * n * alpha * ahnm1)
                        + coeff * coeff * (m / 2.0) / (sqrt_opahnm * opahn) * n * alpha * ahnm1,
                        True,
                    ),
                ),
            ),
        ],
        "VanGRelPermDer",
    )

    # saturation (Saturation, CALCFCN)
    # s = s_dif / opahn^m + s_res

    create_kernel_func_and_wrapper(
        sfg,
        vg_terms + [
            ps.Assignment(sat.center(), sp.Piecewise((s_dif + s_res, saturated), (s_dif / opahnm + s_res, True))),
        ],
        "VanGSaturation",
    )

    # derivative of the saturation (Saturation, CALCDER)
    # (m * n * alpha * ahnm1) * s_dif / opahn^(m + 1)

    create_kernel_func_and_wrapper(
        sfg,
        vg_terms + [
            ps.Assignment(
                sat.center(),
                sp.Piecewise((0.0, saturated), ((m * n * alpha * ahnm1) * s_dif / (opahnm * opahn), True)),
            ),
        ],
        "VanGSaturationDer",
    )
