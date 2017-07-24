set (SRC_FILES advect.F
  advection_godunov.c
  axpy.c
  background.c
  bc_lb.c
  bc_pressure.c
  bc_pressure_package.c
  calc_elevations.c
  cghs.c
  char_vector.c
  chebyshev.c
  comm_pkg.c
  communication.c
  computation.c
  compute_maximums.c
  compute_total_concentration.c
  constantRF.c
  constant_porosity.c
  copy.c
  create_grid.c
  diag_scale.c
  diffuse_lb.c
  discretize_pressure.c
  distribute_usergrid.c
  dpofa.c
  dposl.c
  gauinv.c
  general.c
  geom_t_solid.c
  geometry.c
  globals.c
  grgeom_list.c
  grgeom_octree.c
  grgeometry.c
  grid.c
  hbt.c
  infinity_norm.c
  innerprod.c
  input_porosity.c
  inputRF.c
  input_database.c
  kinsol_nonlin_solver.c
  kinsol_pc.c
  l2_error_norm.c
  line_process.c
  logging.c
  matdiag_scale.c
  matrix.c
  matvec.c
  max_field_value.c
  mg_semi.c
  mg_semi_prolong.c
  mg_semi_restrict.c
  new_endpts.c
  nodiag_scale.c
  pcg.c
  permeability_face.c
  perturb_lb.c
  pf_module.c
  pfield.c
  pf_pfmg.c
  pf_pfmg_octree.c
  pf_smg.c
  pgsRF.c
  phase_velocity_face.c
  ppcg.c
  nl_function_eval.c
  printgrid.c
  printmatrix.c
  printvector.c
  problem.c
  problem_bc.c
  problem_bc_internal.c
  problem_bc_phase_saturation.c
  problem_bc_pressure.c
  problem_capillary_pressure.c
  problem_domain.c
  problem_eval.c
  problem_geometries.c
  problem_phase_heat_capacity.c
  problem_ic_phase_concen.c
  problem_ic_phase_pressure.c
  problem_ic_phase_satur.c
  problem_mannings.c
  problem_phase_density.c
  problem_phase_internal_energy.c
  problem_phase_mobility.c
  problem_phase_rel_perm.c
  problem_phase_source.c
  problem_phase_viscosity.c
  problem_porosity.c
  problem_real_space_z.c
  problem_retardation.c
  problem_richards_bc_internal.c
  problem_saturation.c
  problem_thermal_conductivity.c
  problem_saturation_constitutive.c
  problem_spec_storage.c
  problem_dz_scale.c
  problem_toposlope_x.c
  problem_toposlope_y.c
  process_grid.c
  random.c
  ratqr.c
  rb_GS_point.c
  read_parflow_binary.c
  reg_from_stenc.c
  region.c
  richards_jacobian_eval.c
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
  vector.c
  vector_utilities.c
  w_jacobi.c
  well.c
  well_package.c
  wells_lb.c
  write_parflow_binary.c
  write_parflow_silo.c
  write_parflow_silo_pmpio.c
  wrf_parflow.c
  n_vector.c
  nl_function_eval.c
  evaptranssum.c
  overlandsum.c
  overlandflow_eval.c
  overlandflow_eval_diffusive.c
  compute_top.c
  input_checks.c
  )

add_library(pfsimulator ${SRC_FILES})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})


install (TARGETS pfsimulator DESTINATION lib)

if (${PARFLOW_HAVE_HYPRE})
  target_include_directories (pfsimulator PUBLIC ${HYPRE_INCLUDE_DIR})
endif (${PARFLOW_HAVE_HYPRE})

if (${PARFLOW_HAVE_MPI})
  target_include_directories (pfsimulator PUBLIC "${MPI_C_INCLUDE_PATH}")
endif (${PARFLOW_HAVE_MPI})

if (${PARFLOW_HAVE_HDF5})
  target_include_directories (pfsimulator PUBLIC "${HDF5_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_HDF5})

if (${PARFLOW_HAVE_SILO})
  target_include_directories (pfsimulator PUBLIC "${SILO_INCLUDE_DIRS}")
endif (${PARFLOW_HAVE_SILO})

