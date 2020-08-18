# ParFlow-integration/test/python

This directory contains tools for testing Python scripts and examples of the ParFlow
test files converted to Python. The folders divide the tests according to the modules
they require, based on the parflow/test/CMakeLists.txt. The "other" folder holds
tests that additional to the standard ParFlow test suite.

## Folders:

### base

- default_richards_wells.tcl
- forsyth2.tcl
- harvey_flow.tcl
- harvey_flow_pgs.tcl
- crater2D.tcl
- crater2D_vangtable_spline.tcl
- crater2D_vangtable_linear.tcl
- small_domain.tcl
- richards_hydrostatic_equalibrium.tcl
- default_richards_nocluster.tcl
- octree-simple.tcl
- octree-large-domain.tcl
- terrain_following_grid_overland.tcl
- var_dz_1D.tcl
- pfmg.tcl
- pfmg_galerkin.tcl
- smg.tcl
- pfmg_octree.tcl
- van-genuchten-file.tcl NEEDDIST
- overland_tiltedV_KWE.tcl NEEDUPDATE
- overland_slopingslab_DWE.tcl
- overland_tiltedV_DWE.tcl NEEDUPDATE
- overland_FlatICP.tcl NEEDUPDATE
- richards_FBx.tcl
- richards_FBy.tcl
- richards_box_proctest.tcl

### base_2d

- default_overland.tcl
- default_overland.pfmg.jac.tcl
- default_overland.pfmg_octree.jac.tcl
- default_overland.pfmg_octree.fulljac.tcl
- LW_var_dz.tcl NEEDDIST
- LW_var_dz_spinup.tcl NEEDDIST
- overland_slopingslab_KWE.tcl NEEDUPDATE
- richards_box_proctest_vardz.tcl NEEDUPDATE

### base_3d

- default_single.tcl
- default_richards.tcl

### clm

TBD

### clm-samrai

- clm_samrai.tcl TODO

### netcdf

- default_richards_with_netcdf.tcl TODO

### samrai

- samrai_compute_domain.tcl TODO
- samrai.tcl TODO

### silo

- indicator_field NEEDDIST
- water_balance_y
- water_balance_x
- water_balance_x.hardflow.nojac
- water_balance_x.hardflow.jac
- default_richards_with_silo

### other
