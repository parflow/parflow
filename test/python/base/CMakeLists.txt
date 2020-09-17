include(ParflowTest)

set(TESTS
  default_richards_wells
  forsyth2
  harvey_flow
  harvey_flow_pgs
  crater2D
  crater2D_vangtable_spline
  crater2D_vangtable_linear
  small_domain
  richards_hydrostatic_equilibrium
)

if(PARFLOW_HAVE_HYPRE OR DEFINED ENV{PARFLOW_DIR})
  list(APPEND TESTS
    default_richards_nocluster
    octree-simple
    octree-large-domain
    terrain_following_grid_overland
    var_dz_1D
    pfmg
    pfmg_galerkin
    smg
    pfmg_octree
    van-genuchten-file
    overland_tiltedV_KWE
    overland_slopingslab_DWE
    overland_tiltedV_DWE
    overland_FlatICP
    richards_FBx
    richards_FBy
    richards_box_proctest)
endif()


foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()
