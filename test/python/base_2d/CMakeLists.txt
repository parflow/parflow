include(ParflowTest)

if(${PARFLOW_HAVE_HYPRE})
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
  pf_add_py_test(${inputfile})
  foreach(processor_topology "1 2 1" "2 1 1" "1 4 1" "4 1 1")
    continue()
  endforeach()
endforeach()