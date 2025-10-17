set (SRC_FILES_CONST advect.F
  advection_godunov.c
  bc_lb.c
  bc_pressure.c
  bc_pressure_package.c
  cghs.c
  char_vector.c
  chebyshev.c
  comm_pkg.c
  communication.c
  computation.c
  compute_maximums.c
  compute_top_and_bottom.c
  compute_patch_top.c
  compute_total_concentration.c
  constantRF.c
  constant_porosity.c
  create_grid.c
  diag_scale.c
  diffuse_lb.c
  discretize_pressure.c
  distribute_usergrid.c
  dpofa.c
  dposl.c
  evaptranssum.c
  gauinv.c
  general.c
  geom_t_solid.c
  geometry.c
  grgeom_list.c
  grgeom_octree.c
  grid.c
  hbt.c
  index_space.c
  infinity_norm.c
  inputRF.c
  input_checks.c
  input_database.c
  input_porosity.c
  kinsol_nonlin_solver.c
  kinsol_pc.c
  l2_error_norm.c
  line_process.c
  logging.c
  matdiag_scale.c
  max_field_value.c
  metadata.c
  n_vector.c
  new_endpts.c
  nodiag_scale.c
  overlandsum.c
  pcg.c
  permeability_face.c
  perturb_lb.c
  pf_hypre.c
  pf_n_vector.c
  pf_pfmg.c
  pf_pfmg_octree.c
  pf_smg.c
  pfield.c
  pgsRF.c
  ppcg.c
  printgrid.c
  printmatrix.c
  printvector.c
  problem.c
  problem_bc_internal.c
  problem_bc_phase_saturation.c
  problem_capillary_pressure.c
  problem_domain.c
  problem_dz_scale.c
  problem_eval.c
  problem_FBx.c
  problem_FBy.c
  problem_FBz.c
  problem_geometries.c
  problem_ic_phase_concen.c
  problem_ic_phase_satur.c
  problem_mannings.c
  problem_phase_heat_capacity.c
  problem_phase_internal_energy.c
  problem_phase_mobility.c
  problem_phase_viscosity.c
  problem_porosity.c
  problem_real_space_z.c
  problem_retardation.c
  problem_richards_bc_internal.c
  problem_saturation_constitutive.c
  problem_spec_storage.c
  problem_thermal_conductivity.c
  problem_toposlope_x.c
  problem_toposlope_y.c
  problem_wc_x.c
  problem_wc_y.c
  process_grid.c
  random.c
  ratqr.c
  read_parflow_binary.c
  read_parflow_netcdf.c
  reg_from_stenc.c
  sadvect.F
  sadvection_godunov.c
  scale.c
  select_time_step.c
  set_problem_data.c
  sim_shear.c
  solver.c
  solver_impes.c
  solver_lb.c
  solver_richards.c
  subsrf_sim.c
  time_cycle_data.c
  timing.c
  total_velocity_face.c
  turning_bandsRF.c
  usergrid_input.c
  w_jacobi.c
  well.c
  well_package.c
  wells_lb.c
  reservoir.c
  reservoir_package.c
  wrf_parflow.c
  cpl_parflow.c
  write_clm_netcdf.c
  write_parflow_binary.c
  write_parflow_netcdf.c
  write_parflow_pdi.c
  write_parflow_silo.c
  write_parflow_silo_pmpio.c
)

if( (${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}) OR (${PARFLOW_HAVE_OMP}) )

  set (SRC_FILES_ARCH_TYPE_0
    clustering.cpp
    grgeometry.cpp
  )
  set (SRC_FILES_ARCH_TYPE_1
    axpy.cpp
    copy.cpp
    innerprod.cpp
    matrix.cpp
    matvec.cpp
    mg_semi.cpp
    mg_semi_restrict.cpp
    mg_semi_prolong.cpp
    nl_function_eval.cpp
    overlandflow_eval.cpp
    overlandflow_eval_diffusive.cpp
    overlandflow_eval_Kin.cpp
    phase_velocity_face.cpp
    problem_phase_density.cpp
    problem_phase_rel_perm.cpp
    problem_phase_source.cpp
    problem_saturation.cpp
    rb_GS_point.cpp
    richards_jacobian_eval.cpp
    vector.cpp
    vector_utilities.cpp
  )
  set (SRC_FILES_ARCH_TYPE_2
    background.cpp
    calc_elevations.cpp
    globals.cpp
    pf_module.cpp
    problem_bc.cpp
    problem_bc_pressure.cpp
    problem_ic_phase_pressure.cpp
    region.cpp
  )

if(${PARFLOW_HAVE_KOKKOS})
  list(APPEND SRC_FILES_ARCH_TYPE_0 pf_kokkos.cpp)
endif(${PARFLOW_HAVE_KOKKOS})

  #  PF_COMP_UNIT_TYPE determines the behavior of the accelerated compilation unit
	#  ------------------------------------------------------------
  #  CUDA
	#  ------------------------------------------------------------
	#  1:     NVCC compiler, Unified Memory allocation, Parallel loops on GPUs
	#  2:     NVCC compiler, Unified Memory allocation, Sequential loops on host
	#  Other: NVCC compiler, Standard heap allocation, Sequential loops on host
  #
	#  ------------------------------------------------------------
  #  OpenMP
	#  ------------------------------------------------------------
  #  1:     CXX compiler, Standard heap allocation, Parallel loops on CPU
	#  2:     CXX compiler, Standard heap allocation, Sequential loops on CPU
	#  Other: CXX compiler, Standard heap allocation, Sequential loops on CPU

  set_source_files_properties(${SRC_FILES_ARCH_TYPE_0} PROPERTIES COMPILE_DEFINITIONS "PF_COMP_UNIT_TYPE=0")
  set_source_files_properties(${SRC_FILES_ARCH_TYPE_1} PROPERTIES COMPILE_DEFINITIONS "PF_COMP_UNIT_TYPE=1")
  set_source_files_properties(${SRC_FILES_ARCH_TYPE_2} PROPERTIES COMPILE_DEFINITIONS "PF_COMP_UNIT_TYPE=2")

  set (SRC_FILES_ARCH ${SRC_FILES_ARCH_TYPE_0} ${SRC_FILES_ARCH_TYPE_1} ${SRC_FILES_ARCH_TYPE_2})

  if((${PARFLOW_HAVE_CUDA}) AND (NOT (${PARFLOW_HAVE_KOKKOS})))
    set_source_files_properties(${SRC_FILES_ARCH} PROPERTIES LANGUAGE CUDA)
  else((${PARFLOW_HAVE_CUDA}) AND (NOT (${PARFLOW_HAVE_KOKKOS})))
    set_source_files_properties(${SRC_FILES_ARCH} PROPERTIES LANGUAGE CXX)
  endif((${PARFLOW_HAVE_CUDA}) AND (NOT (${PARFLOW_HAVE_KOKKOS})))


  add_library(pfsimulator ${SRC_FILES_ARCH} ${SRC_FILES_CONST})

  if ( ${PARFLOW_HAVE_CUDA} )
    target_link_libraries(pfsimulator nvToolsExt)
    target_include_directories(pfsimulator PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  endif ( ${PARFLOW_HAVE_CUDA} )

  if( ${PARFLOW_HAVE_KOKKOS} )
    target_link_libraries(pfsimulator Kokkos::kokkos)
    target_include_directories(pfsimulator PUBLIC ${KOKKOS_INCLUDE})
  endif( ${PARFLOW_HAVE_KOKKOS} )

  if( ${PARFLOW_HAVE_RMM} )
    target_link_libraries(pfsimulator rmm::rmm)
  endif( ${PARFLOW_HAVE_RMM} )

  if( ${PARFLOW_HAVE_UMPIRE} )
    target_link_libraries(pfsimulator umpire)
  endif( ${PARFLOW_HAVE_UMPIRE} )  

else( (${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}) OR (${PARFLOW_HAVE_OMP}) )

  set (SRC_FILES_ARCH
    axpy.c
    background.c
    calc_elevations.c
    clustering.c
    copy.c
    globals.c
    grgeometry.c
    innerprod.c
    matrix.c
    matvec.c
    mg_semi.c
    mg_semi_restrict.c
    mg_semi_prolong.c
    nl_function_eval.c
    overlandflow_eval.c
    overlandflow_eval_diffusive.c
    overlandflow_eval_Kin.c
    phase_velocity_face.c
    pf_module.c
    problem_bc.c
    problem_bc_pressure.c
    problem_ic_phase_pressure.c
    problem_phase_density.c
    problem_phase_rel_perm.c
    problem_phase_source.c
    problem_saturation.c
    rb_GS_point.c
    region.c
    richards_jacobian_eval.c
    vector.c
    vector_utilities.c
  )
  add_library(pfsimulator ${SRC_FILES_ARCH} ${SRC_FILES_CONST})

endif( (${PARFLOW_HAVE_CUDA}) OR (${PARFLOW_HAVE_KOKKOS}) OR (${PARFLOW_HAVE_OMP}) )

if( ${PARFLOW_BUILD_WITH_CPP} )
  set_source_files_properties(${SRC_FILES_ARCH} ${SRC_FILES_CONST}
    PROPERTIES LANGUAGE CXX)
endif()

if (${PARFLOW_HAVE_SUNDIALS})
  target_include_directories (pfsimulator PUBLIC ${SUNDIALS_INCLUDE_DIR})
  target_link_libraries (pfsimulator ${SUNDIALS_LIBRARIES})
else ()
  target_link_libraries(pfsimulator pfkinsol)
endif (${PARFLOW_HAVE_SUNDIALS})

target_link_libraries(pfsimulator amps cjson ${PARFLOW_ETRACE_LIBRARY})
target_include_directories(pfsimulator PUBLIC "../third_party/cjson")

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

install (TARGETS pfsimulator DESTINATION lib)

if(${PARFLOW_HAVE_CLM})
  target_link_libraries (pfsimulator pfclm)
endif(${PARFLOW_HAVE_CLM})

if (${PARFLOW_HAVE_HYPRE})
  target_include_directories (pfsimulator PUBLIC ${HYPRE_INCLUDE_DIR})
  target_link_libraries (pfsimulator Hypre::Hypre)
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
  target_link_libraries (pfsimulator ${MPI_LIBRARIES})
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_HDF5})
  target_include_directories (pfsimulator PUBLIC "${HDF5_INCLUDE_DIRS}")
  target_link_libraries (pfsimulator ${HDF5_LIBRARIES})
  if (${PARFLOW_HAVE_NETCDF})
    target_link_libraries (pfsimulator ${HDF5_HL_LIBRARIES})
  endif (${PARFLOW_HAVE_NETCDF})
endif (${PARFLOW_HAVE_HDF5})

if (${PARFLOW_HAVE_SILO})
  target_include_directories (pfsimulator PUBLIC "${SILO_INCLUDE_DIRS}")
  target_link_libraries (pfsimulator ${SILO_LIBRARIES})
endif (${PARFLOW_HAVE_SILO})

if (${PARFLOW_HAVE_PDI})
  target_include_directories (pfsimulator PUBLIC "${PDI_INCLUDE_DIRS}")
  target_include_directories (pfsimulator PUBLIC "${PARACONF_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_PDI})

if (${PARFLOW_HAVE_NETCDF})
  target_link_libraries (pfsimulator NetCDF::NetCDF)
endif (${PARFLOW_HAVE_NETCDF})
