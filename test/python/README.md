# parflow/test/python

This directory contains tools for testing Python scripts and examples of the ParFlow
test files converted to Python. The folders divide the tests according to the modules
they require, based on the parflow/test/CMakeLists.txt. 

Tests with an asterisk (*) have additional functionality that was included in the TCL
tests, but is not used in the Python tests (e.g. different options for key settings). 
These options are commented out in these tests. Also, output comparison has not been 
implemented in the Python tests. 

## Folders:

### base

- crater2D
- crater2D_vangtable_spline
- crater2D_vangtable_linear
- default_richards_nocluster
- default_richards_wells
- forsyth2
- harvey_flow *
- harvey_flow_pgs *
- octree-large-domain
- octree-simple
- overland_FlatICP *
- overland_slopingslab_DWE *
- overland_tiltedV_DWE * 
- overland_tiltedV_KWE * 
- pfmg
- pfmg_galerkin
- pfmg_octree
- richards_box_proctest
- richards_FBx
- richards_FBy
- richards_hydrostatic_equilibrium
- small_domain
- smg
- terrain_following_grid_overland
- van-genuchten-file
- var_dz_1D

### base_2d

- default_overland
- default_overland.pfmg.jac
- default_overland.pfmg_octree.jac
- default_overland.pfmg_octree.fulljac
- LW_var_dz
- LW_var_dz_spinup
- overland_slopingslab_KWE *
- richards_box_proctest_vardz

### base_3d

- default_richards
- default_single

### clm

- clm
- clm-reuse
- clm.jac
- clm_4levels
- clm_forc_veg
- clm_varDZ
- clm_vtk (not part of the main test suite)

### clm-samrai

- clm_samrai (INCOMPLETE)

### netcdf

- default_richards_with_netcdf (INCOMPLETE)

### new_features

> Tests for Python script functions

- enum_versioning
- full_clone
- hyphen_test
- os_function
- pfset_test
- prefix_naming
- serial_runs
- write_check
- yaml_output

### samrai

- samrai_compute_domain (INCOMPLETE)
- samrai (INCOMPLETE)

### scenarios

> Tests for Python script build helper functions

- ecoslim_gen (INCOMPLETE)

### silo

- default_richards_with_silo
- indicator_field
- water_balance_x
- water_balance_x.hardflow.jac
- water_balance_x.hardflow.nojac
- water_balance_y