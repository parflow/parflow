import sympy

from pystencils import TypedSymbol
from pystencils import DynamicType

from field_factory import FieldFactory
from pystencils_codegen import *

# sympy is not imported as sp here, since the saturation field below is called sp


def stride_symbols(name: str):
    """Stride symbols shared by the fields of one FieldFactory"""
    return tuple(TypedSymbol(f"_stride_{name}_{d}", DynamicType.INDEX_TYPE) for d in range(3))


# averages of nl_function_eval.c

def mean(a, b):
    """Mean"""
    return 0.5 * (a + b)


def harmonic_mean(a, b):
    """HarmonicMean, PMean(a, b, c, d) is HarmonicMean(c, d)"""
    return sympy.Piecewise((2.0 * a * b / (a + b), sympy.Ne(a + b, 0)), (0.0, True))


def harmonic_mean_dz(a, b, c, d):
    """HarmonicMeanDZ = PMeanDZ"""
    return sympy.Piecewise((((c + d) * a * b) / ((b * c) + (a * d)), sympy.Ne((c * b) + (a * d), 0)), (0.0, True))


def upstream_mean(a, b, c, d):
    """UpstreamMean = RPMean"""
    return sympy.Piecewise((c, a - b >= 0), (d, True))


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

    # terms of the accumulation and source kernels, shared by the fused and the unfused kernels

    # (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol * del_x_slope * del_y_slope * z_mult_dat[ip]
    base = (
        (sp.center() * dp.center() - osp.center() * odp.center())
        * pop.center()
        * vol
        * del_x_slope
        * del_y_slope
        * z_mult_dat.center()
    )

    # ss[ip] * vol * del_x_slope * del_y_slope * z_mult_dat[ip] * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip])
    storage = (
        ss.center()
        * vol
        * del_x_slope
        * del_y_slope
        * z_mult_dat.center()
        * (pp.center() * sp.center() * dp.center() - opp.center() * osp.center() * odp.center())
    )

    # vol * del_x_slope * del_y_slope * z_mult_dat[ip] * dt * (phase_source[ip] + et[ip])
    def source(phase_source):
        return vol * del_x_slope * del_y_slope * z_mult_dat.center() * dt * (phase_source.center() + et.center())

    # flux: base
    # fp[ip] = base

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), base),
        "Flux_Base",
        timing_index=True,
    )

    # flux: add compressible storage
    # fp[ip] += storage

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), fp.center() + storage),
        "Flux_AddCompressibleStorage",
        timing_index=True,
    )

    # flux: fused base + compressible storage + source terms
    # src is the phase source vector, sp the saturation (both are sp in the unfused kernels)

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), base + storage - source(src)),
        "Flux_FusedAccumulationAndSourceTerms",
        timing_index=True,
    )

    # flux: add source terms (sp is the phase source here)
    # fp[ip] -= source

    create_kernel_func_and_wrapper(
        sfg,
        ps.Assignment(fp.center(), fp.center() - source(sp)),
        "Flux_AddSourceTerms",
        timing_index=True,
    )

    # flux: face fluxes and their divergence (SlopeUpwindFormulation Original, tfgupwind == 0)
    #
    # The flux loop of NlFunctionEval computes the fluxes through the right, front and upper face of every cell
    # and scatters them with atomics: + into the cell itself and - into the neighbor. Here the first kernel
    # computes the same face fluxes and velocities on the same loop bounds (a replacement for the GrGeomInLoop)
    # and stores the fluxes in face flux vectors. The second kernel adds the divergence of the face fluxes to fp,
    # writing every cell once. The face flux vectors are zero where no flux is computed, so the divergence kernel
    # runs on all cells of the subgrid, also those next to the domain that receive fluxes through their lower faces.
    # The terrain-following terms sin(atan(s)) and cos(atan(s)) are simplified to s / sqrt(1 + s^2) and
    # 1 / sqrt(1 + s^2) by sympy.

    rpp, kx, ky, kz, fbx, fby, fbz = [f_ff.create_new(name) for name in ["rpp", "kx", "ky", "kz", "fbx", "fby", "fbz"]]

    ## 2D slopes, the wrapper passes a z stride of 0
    s_ff = FieldFactory((nx, ny, nz), stride_symbols("s"))
    xssl, yssl = [s_ff.create_new(name) for name in ["xssl", "yssl"]]

    ## face velocities, vx.center() is the velocity through the right face of the cell etc.
    vx, vy, vz = [FieldFactory((nx, ny, nz), stride_symbols(name)).create_new(name) for name in ["vx", "vy", "vz"]]

    ## face fluxes, flux_x.center() is the flux through the right face of the cell etc.
    flux_ff = FieldFactory((nx, ny, nz), stride_symbols("flux"))
    flux_x, flux_y, flux_z = [flux_ff.create_new(name) for name in ["flux_x", "flux_y", "flux_z"]]

    dx, dy, dz, gravity, viscosity = sympy.symbols("dx, dy, dz, gravity, viscosity")

    ffx, ffy, ffz = dy * dz, dx * dz, dx * dy

    c = (0, 0, 0)

    def face_flux_xy(axis):
        """Flux through the right (axis 0) or front (axis 1) face of the cell"""
        n = (1, 0, 0) if axis == 0 else (0, 1, 0)
        ssl, perm, fb, h, ff = (xssl, kx, fbx, dx, ffx) if axis == 0 else (yssl, ky, fby, dy, ffy)
        tag = "xy"[axis]
        dir_g, dir_g_c, diff, updir, u = sympy.symbols(f"dir_g_{tag}, dir_g_c_{tag}, diff_{tag}, updir_{tag}, u_{tag}")

        return [
            ps.Assignment(dir_g, mean(gravity * sympy.sin(sympy.atan(ssl[c])), gravity * sympy.sin(sympy.atan(ssl[n])))),
            ps.Assignment(dir_g_c, mean(gravity * sympy.cos(sympy.atan(ssl[c])), gravity * sympy.cos(sympy.atan(ssl[n])))),
            ps.Assignment(diff, pp[c] - pp[n]),
            ps.Assignment(updir, (diff / h) * dir_g_c - dir_g),
            # the original adds the pressure (diff / h * dir_g_c) and gravity (-dir_g) terms
            # separately, their sum is updir
            ps.Assignment(
                u,
                z_mult_dat[c] * ff * harmonic_mean(perm[c], perm[n])
                * updir
                * upstream_mean(updir, 0.0, rpp[c] * dp[c], rpp[n] * dp[n])
                / viscosity
                * fb[c],
            ),
        ], u

    def face_flux_z():
        """Flux through the upper face of the cell"""
        n = (0, 0, 1)
        sep, lower, upper, u = sympy.symbols("sep_z, lower_z, upper_z, u_z")

        return [
            ps.Assignment(sep, dz * mean(z_mult_dat[c], z_mult_dat[n])),
            ps.Assignment(
                lower, pp[c] / sep - (z_mult_dat[c] / (z_mult_dat[c] + z_mult_dat[n])) * dp[c] * gravity
            ),
            ps.Assignment(
                upper, pp[n] / sep + (z_mult_dat[n] / (z_mult_dat[c] + z_mult_dat[n])) * dp[n] * gravity
            ),
            ps.Assignment(
                u,
                ffz * harmonic_mean_dz(kz[c], kz[n], z_mult_dat[c], z_mult_dat[n])
                * (lower - upper)
                * upstream_mean(lower, upper, rpp[c] * dp[c], rpp[n] * dp[n])
                / viscosity
                * fbz[c],
            ),
        ], u

    (x_assignments, u_x), (y_assignments, u_y), (z_assignments, u_z) = face_flux_xy(0), face_flux_xy(1), face_flux_z()

    # no vectorization on cpus: the conditionals of the means cannot be vectorized by pystencils
    face_flux_kernel = create_kernel_func(
        sfg,
        x_assignments + y_assignments + z_assignments + [
            ps.Assignment(flux_x.center(), u_x),
            ps.Assignment(flux_y.center(), u_y),
            ps.Assignment(flux_z.center(), u_z),
            ps.Assignment(vx.center(), u_x / (ffx * z_mult_dat.center())),
            ps.Assignment(vy.center(), u_y / (ffy * z_mult_dat.center())),
            ps.Assignment(vz.center(), u_z / ffz),
        ],
        "Flux_FaceFluxes",
        allow_vect=False,
        ghost_layers=0,
    )

    ## velocities are stored on the upper faces of the cells, slopes are 2D
    create_grgeom_in_loop_wrapper(
        sfg,
        face_flux_kernel,
        starts={
            "_data_vx": ("PV_ixl + 1", "PV_iyl", "PV_izl"),
            "_data_vy": ("PV_ixl", "PV_iyl + 1", "PV_izl"),
            "_data_vz": ("PV_ixl", "PV_iyl", "PV_izl + 1"),
            "_data_xssl": ("PV_ixl", "PV_iyl", "grid2d_iz"),
            "_data_yssl": ("PV_ixl", "PV_iyl", "grid2d_iz"),
        },
        strides={"_stride_s_2": "0"},
        extra=[sfg.var("grid2d_iz", SInt(32))],
    )

    ## fp += dt * (fluxes through the right, front and upper face - fluxes through the left, back and lower face)
    divergence_kernel = create_kernel_func(
        sfg,
        ps.Assignment(
            fp.center(),
            fp.center()
            + dt * (flux_x.center() + flux_y.center() + flux_z.center())
            - dt * flux_x[-1, 0, 0]
            - dt * flux_y[0, -1, 0]
            - dt * flux_z[0, 0, -1],
        ),
        "Flux_FaceFluxDivergence",
        ghost_layers=0,
    )
    create_box_loop_wrapper(sfg, divergence_kernel)
