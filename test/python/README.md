# parflow/test/python

This directory contains tools for testing Python scripts and examples of the ParFlow
test files converted to Python. The folders divide the tests according to the modules
they require, based on the parflow/test/CMakeLists.txt.

Tests with an asterisk (*) have additional functionality that was included in the TCL
tests, but is not used in the Python tests (e.g. different options for key settings).
These options are commented out in these tests. Also, output comparison has not been
implemented in the Python tests.


- crater2D
- crater2D_vangtable_spline
- crater2D_vangtable_linear
- default_overland
- default_overland_only
- FJC_overland_only_mannings
- default_overland.pfmg.jac
- default_overland.pfmg_octree.jac
- default_overland.pfmg_octree.fulljac
- default_richards
- default_richards_with_netcdf (INCOMPLETE)
- default_richards_nocluster
- default_richards_wells
- default_single
- forsyth2
- harvey_flow *
- harvey_flow_pgs *
- indicator_field (INCOMPLETE)
- LW_var_dz
- LW_var_dz_spinup
- octree-large-domain
- octree-simple
- overland_FlatICP *
- overland_slopingslab_DWE *
- overland_slopingslab_KWE *
- overland_tiltedV_DWE *
- overland_tiltedV_KWE *
- pfmg
- pfmg_galerkin
- pfmg_octree
- richards_box_proctest
- richards_box_proctest_vardz
- richards_FBx
- richards_FBy
- richards_hydrostatic_equilibrium
- small_domain
- smg
- terrain_following_grid_overland
- van-genuchten-file
- var_dz_1D

## OverlandOnly mass-balance validation

`default_overland_only.py` is a numerical validation test for `Solver.OverlandOnly=True`.
It uses a small multi-layer rectangular sloping plane with an impermeable inactive
subsurface and one downstream outlet. The top pressure cells are the only active
storage state.

The test runs six cases:

- `OverlandFlow` with a constant top rainfall flux.
- `OverlandFlowPFB` with an equivalent uniform rainfall PFB file.
- `OverlandFlow` with equivalent steady `Solver.EvapTransFile` forcing.
- `OverlandKinematic` with a constant top rainfall flux.
- `OverlandDiffusive` with a constant top rainfall flux.
- `OverlandFlow` with a finite rainfall pulse followed by drain-down.

Passing requires:

- all cells below the top surface keep their initial pressure,
- timestep surface storage change satisfies `input - output = change in storage`,
- final discharge approaches the known steady solution `rainfall_rate * plan_area`,
- the baseline `OverlandFlow` depth and `qx_overland` profiles approach the Manning
  sloping-plane solution away from boundary cells,
- the `OverlandFlowPFB` and `EvapTransFile` cases match the constant-flux baseline
  final storage and final outflow,
- the finite pulse volume leaves the domain by the end of the run, up to residual
  surface storage and numerical tolerance.

For `OverlandFlow` and `OverlandKinematic`, the mass balance uses the printed
downstream `qx_overland` outlet flux. For `OverlandDiffusive`, the global outflow is
inferred from the storage change because ghost-side diffusive boundary terms are not
fully represented in the printed edge flux array for this compact box test.

`overland_only_validation.py` verifies Python preprocessor errors for unsupported
OverlandOnly combinations before ParFlow execution. These include CLM, wells,
reservoirs, missing overland BCs, `FluxFile`, `SurfacePredictor`, `Spinup`,
`ResetSurfacePressure`, and file-backed forcing for `OverlandKinematic` or
`OverlandDiffusive`.

`LW_overland_only_overland_flow.py`, `LW_overland_only_kinematic.py`, and
`LW_overland_only_diffusive.py` are distributed Little Washita network-routing
checks. They reuse the LW terrain, slopes, indicator, and initial-pressure inputs,
strip CLM/NLDAS forcing, pin the subsurface with `Solver.OverlandOnly=True`, and
verify finite nonnegative surface storage plus a routing response. Final pressure
known-output comparisons run when the reviewed reference PFBs are present in
`test/correct_output`.

`FJC_overland_only_mannings.py` is a single-processor Fourth of July Creek
network-routing check for `OverlandOnly + OverlandFlow` with spatially variable
Manning's coefficients. It builds a temporary solid domain from `FJC_Mask.pfb`,
uses PFB-backed `SlopeX`, `SlopeY`, and `Mannings`, runs a short rainfall pulse
with `PFMG` and `UseJacobian=True`, verifies pressure below the overland surface
remains unchanged, checks that top-layer pressure and `qx_overland`/`qy_overland`
form a concentrated stream-network-like routing signature, and compares pressure
and overland flux fields to reviewed known output.

Subdirectories:

### clm

- clm
- clm-reuse
- clm.jac
- clm_4levels
- clm_forc_veg
- clm_varDZ
- clm_vtk (not part of the main test suite)
- clm_samrai (INCOMPLETE)

### new_features

> Tests for Python script functions and new features

- default_db
- enum_versioning
- full_clone
- hyphen_test
- os_function
- pfset_test
- prefix_naming
- serial_runs
- write_check
- asc_write
- image-as-mask
- pfb_mask
- simple-mask
- table_loading
- tables_LW

### washita/py_scripts

- Dist_Forcings (INCOMPLETE)
- LW_NetCDF_Test (INCOMPLETE)
- LW_Test (INCOMPLETE)
- LW_Timing (INCOMPLETE)
