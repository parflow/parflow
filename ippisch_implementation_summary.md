# Implementation Summary: Ippisch Air-Entry Modification & Saturation Lookup Table

## Branch: `feature/ippisch-air-entry`

8 commits, 31 files changed, +2882/-111 lines

## What was implemented

### Feature 1: Ippisch air-entry modification (Ippisch et al., 2006)

Introduces a finite air-entry pressure head h_s that eliminates the Kr derivative singularity at saturation for any n > 1. The standard Van Genuchten-Mualem model has an unbounded dKr/dS at saturation when n < 2 (m < 0.5), which causes Newton solver convergence difficulties for fine-textured soils with measured n values below 2.

The modification rescales effective saturation and relative permeability:

- S_e(h) = S_c(h) / S_c(h_s) for |h| > h_s, S_e = 1 for |h| <= h_s
- Kr_ippisch = Kr_standard / (sqrt(S_cs) * F_denom^2)

Applied to both relative permeability (`problem_phase_rel_perm.c`) and saturation (`problem_saturation.c`).

Three modes via `Phase.Saturation.VanGenuchten.AirEntryMode`:
- `None` (default) — standard VG, fully backward compatible
- `Constant` — single h_s for all regions
- `InverseAlpha` — h_s = 1/alpha computed per region
- `PerRegion` — h_s specified per geometry

Scope: by-region indicator path (`data_from_file == 0`) only. PFB path is out of scope.

### Feature 2: Saturation lookup table

Mirrors the existing Kr lookup table pattern using Fritsch-Carlson monotonic Hermite spline interpolation. Provides faster S(h) evaluation for Richards' equation solves.

New keys per region:
- `Geom.{geom_name}.Saturation.NumSamplePoints` (default 0 = direct eval)
- `Geom.{geom_name}.Saturation.MinPressureHead`
- `Geom.{geom_name}.Saturation.InterpolationMethod` (Spline or Linear)

Works with or without Ippisch modification.

## Files modified

| File | Changes |
|------|---------|
| `pfsimulator/parflow_lib/problem_phase_rel_perm.c` | VanGTable struct (h_s field), VanGComputeTable (Ippisch scaling, table domain from h_s), VanGLookupSpline (air-entry check), all direct-eval fallbacks (surface AND interior loops), linear interp paths, key reading, cleanup |
| `pfsimulator/parflow_lib/problem_saturation.c` | SatTable struct, SatComputeTable(), SatLookupSpline(), Type1 struct (h_s_values, lookup_tables), direct-eval with Ippisch (CALCFCN/CALCDER), key reading, cleanup |
| `pf-keys/definitions/phase.yaml` | AirEntryMode, AirEntryHead keys under Phase.Saturation.VanGenuchten |
| `pf-keys/definitions/geom.yaml` | Saturation.AirEntryHead, NumSamplePoints, MinPressureHead, InterpolationMethod |
| `docs/user_manual/models.rst` | New "Ippisch Air-Entry Modification" subsection with equations |
| `docs/user_manual/keys.rst` | Documentation for all new keys with TCL/Python syntax examples |
| `docs/user_manual/refs.bib` | Ippisch et al. (2006) reference |

## New files

| File | Purpose |
|------|---------|
| `test/python/crater2D_ippisch_spline.py` | Test: AirEntryMode=InverseAlpha, n=1.5, both Kr+Sat tables |
| `test/python/crater2D_ippisch_constant.py` | Test: AirEntryMode=Constant, h_s=0.02, n=1.5, both tables |
| `test/python/compare_ippisch_performance.py` | Diagnostic: 6-config solver performance comparison |
| `test/python/debug_ippisch_table.py` | Diagnostic: table accuracy verification |
| `test/correct_output/crater2D_ippisch_*.pfb` | 20 reference PFB files for regression testing |

## Testing

- **Backward compatibility**: crater2D, crater2D_vangtable_spline, crater2D_vangtable_linear — all pass with identical results
- **New Ippisch tests**: Both pass against correct_output with 5 significant digits
- **Table vs direct-eval**: Verified identical solver iterations and field differences < 1e-9 for both standard VG and Ippisch (n=1.5 and n=2.0)
- **Formatting**: uncrustify (C files), black (Python files)

## Bug found and fixed during development

The interior cell loop (`GrGeomInLoop` in the "Compute rel. perms. on interior" section of `problem_phase_rel_perm.c`) was initially not modified with the Ippisch air-entry check and scaling. Only the surface loop (`GrGeomSurfLoop`) was modified. This caused the direct-eval path to compute standard VG Kr for interior cells while the table path correctly computed Ippisch Kr via `VanGLookupSpline`, leading to divergent solver behavior (different Newton iterations and ~131 pressure head difference at the wetting front).

The bug was isolated through systematic testing:
1. `ipp_sat_only` (Sat table only) matched direct-eval perfectly
2. `ipp_kr_only` (Kr table only) diverged — pointing to the Kr table path
3. Spline bypass test (direct-eval formulas in the table code path) still diverged — proving the computed values were correct but the code path structure differed
4. Discovery of the second `GrGeomInLoop` section that was missing Ippisch modifications

## Reference

Ippisch, O., Vogel, H.-J., & Bastian, P. (2006). Validity limits for the van Genuchten-Mualem model and implications for parameter estimation and numerical simulation. Advances in Water Resources, 29, 1780-1789. doi:10.1016/j.advwatres.2005.12.011
