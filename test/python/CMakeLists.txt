include(ParflowTest)

set(TESTS
  default_single
  default_richards_wells
  forsyth2
  harvey_flow
  harvey_flow_pgs
  crater2D
  crater2D_vangtable_spline
  crater2D_vangtable_linear
  small_domain
  richards_hydrostatic_equilibrium
  indicator_field
)

if(${PARFLOW_HAVE_HYPRE})
  list(APPEND TESTS
    default_richards
    default_richards_nocluster
    default_overland
    octree-simple
    octree-large-domain
    LW_var_dz
    LW_var_dz_spinup
    terrain_following_grid_overland
    var_dz_1D
    pfmg
    pfmg_galerkin
    smg
    pfmg_octree
    van-genuchten-file
    overland_slopingslab_KWE
    overland_tiltedV_KWE
    overland_slopingslab_DWE
    overland_tiltedV_DWE
    overland_FlatICP
    richards_FBx
    richards_FBy
    richards_box_proctest
    richards_box_proctest_vardz
    water_balance_y
    water_balance_x
    water_balance_x.hardflow.nojac
    water_balance_x.hardflow.jac)
endif()

if(${PARFLOW_HAVE_SILO})
  list(APPEND TESTS
    # indicator_field
    )

  if(${PARFLOW_HAVE_HYPRE})
    list(APPEND TESTS
      # default_richards_with_silo
      )
  endif()
endif()

if(${PARFLOW_HAVE_NETCDF})
  if(${PARFLOW_HAVE_HYPRE})
    #This test is failing on several platforms
    list(APPEND TESTS
#    	default_richards_with_netcdf
        )
  endif()
endif()

set(SAMRAI_TESTS)
set(SAMRAI_TESTS_WITH_PATCH_COUNT)

if(${PARFLOW_HAVE_SAMRAI})
  list(APPEND SAMRAI_TESTS
    samrai_compute_domain)

  list(APPEND SAMRAI_TESTS_WITH_PATCH_COUNT
    samrai)
endif()

list(APPEND PARALLEL_3DTOPO_TESTS "")
list(APPEND PARALLEL_2DTOPO_TESTS "")

if ( ${PARFLOW_AMPS_LAYER} IN_LIST PARFLOW_AMPS_LAYER_REQUIRE_MPI )
  list(APPEND PARALLEL_3DTOPO_TESTS
    default_single)

  if(${PARFLOW_HAVE_HYPRE})
    list(APPEND PARALLEL_3DTOPO_TESTS
      default_richards)

    list(APPEND PARALLEL_2DTOPO_TESTS
      default_overland
      default_overland.pfmg.jac
      default_overland.pfmg_octree.fulljac
      LW_var_dz
      LW_var_dz_spinup
      overland_slopingslab_KWE
      richards_box_proctest_vardz)

    if(${PARFLOW_HAVE_SILO})
      list(APPEND PARALLEL_2DTOPO_TESTS
	default_overland.pfmg_octree.jac)
    endif()
  endif()

endif()

foreach(inputfile ${TESTS})
  pf_add_py_test(${inputfile})
endforeach()

###################################################################


foreach(inputfile ${PARALLEL_2DTOPO_TESTS})

  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)

endforeach()


foreach(inputfile ${PARALLEL_3DTOPO_TESTS})

  pf_add_py_parallel_test(${inputfile} 1 1 2)
  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 1 1 4)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)
  pf_add_py_parallel_test(${inputfile} 2 2 2)
  pf_add_py_parallel_test(${inputfile} 3 3 3)

endforeach()

###################################################################

add_subdirectory (new_features)
# add_subdirectory (scenarios)
add_subdirectory (key_validation)

if (${PARFLOW_HAVE_CLM})
  add_subdirectory (clm)
#  add_subdirectory (washita)
endif(${PARFLOW_HAVE_CLM})
