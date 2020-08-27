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

set(TOPO_VALS
  1
  2
  4
)

foreach(inputfile ${2D_TESTS})
  foreach(process_topo_p ${TOPO_VALS})
      foreach(process_topo_q ${TOPO_VALS})
         pf_add_py_parallel_test(${inputfile} ${process_topo_p} ${process_topo_q} 1)
      endforeach()
  endforeach()
endforeach()