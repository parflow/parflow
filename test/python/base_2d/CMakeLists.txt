include(ParflowTest)

if(PARFLOW_HAVE_HYPRE OR DEFINED ENV{PARFLOW_DIR})
  set(2D_TESTS
    default_overland
    default_overland.pfmg.jac
    default_overland.pfmg_octree.jac
    default_overland.pfmg_octree.fulljac
    LW_var_dz
    LW_var_dz_spinup
    overland_slopingslab_KWE
    richards_box_proctest.vardz
  )
endif()

foreach(inputfile ${2D_TESTS})
  pf_add_py_parallel_test(${inputfile} 1 1 1)
  pf_add_py_parallel_test(${inputfile} 1 2 1)
  pf_add_py_parallel_test(${inputfile} 2 1 1)
  pf_add_py_parallel_test(${inputfile} 1 4 1)
  pf_add_py_parallel_test(${inputfile} 4 1 1)
endforeach()
