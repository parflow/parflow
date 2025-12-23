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
