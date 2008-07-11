# Microsoft Developer Studio Generated NMAKE File, Based on ParFlow.dsp
!IF "$(CFG)" == ""
CFG=ParFlow - Win32 Debug
!MESSAGE No configuration specified. Defaulting to ParFlow - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "ParFlow - Win32 Release" && "$(CFG)" != "ParFlow - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ParFlow.mak" CFG="ParFlow - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ParFlow - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ParFlow - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "ParFlow - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\ParFlow.exe"


CLEAN :
	-@erase "$(INTDIR)\_rand48.obj"
	-@erase "$(INTDIR)\advect.win32.obj"
	-@erase "$(INTDIR)\advection_godunov.obj"
	-@erase "$(INTDIR)\amps_allreduce.obj"
	-@erase "$(INTDIR)\amps_bcast.obj"
	-@erase "$(INTDIR)\amps_clear.obj"
	-@erase "$(INTDIR)\amps_createinvoice.obj"
	-@erase "$(INTDIR)\amps_exchange.obj"
	-@erase "$(INTDIR)\amps_ffopen.obj"
	-@erase "$(INTDIR)\amps_finalize.obj"
	-@erase "$(INTDIR)\amps_find_powers.obj"
	-@erase "$(INTDIR)\amps_fopen.obj"
	-@erase "$(INTDIR)\amps_init.obj"
	-@erase "$(INTDIR)\amps_invoice.obj"
	-@erase "$(INTDIR)\amps_io.obj"
	-@erase "$(INTDIR)\amps_irecv.obj"
	-@erase "$(INTDIR)\amps_newhandle.obj"
	-@erase "$(INTDIR)\amps_newpackage.obj"
	-@erase "$(INTDIR)\amps_pack.obj"
	-@erase "$(INTDIR)\amps_recv.obj"
	-@erase "$(INTDIR)\amps_send.obj"
	-@erase "$(INTDIR)\amps_sfbcast.obj"
	-@erase "$(INTDIR)\amps_sfclose.obj"
	-@erase "$(INTDIR)\amps_sfopen.obj"
	-@erase "$(INTDIR)\amps_sizeofinvoice.obj"
	-@erase "$(INTDIR)\amps_sync.obj"
	-@erase "$(INTDIR)\amps_test.obj"
	-@erase "$(INTDIR)\amps_unpack.obj"
	-@erase "$(INTDIR)\amps_vector.obj"
	-@erase "$(INTDIR)\amps_wait.obj"
	-@erase "$(INTDIR)\axpy.obj"
	-@erase "$(INTDIR)\background.obj"
	-@erase "$(INTDIR)\bc_lb.obj"
	-@erase "$(INTDIR)\bc_pressure.obj"
	-@erase "$(INTDIR)\bc_pressure_package.obj"
	-@erase "$(INTDIR)\calc_elevations.obj"
	-@erase "$(INTDIR)\cghs.obj"
	-@erase "$(INTDIR)\char_vector.obj"
	-@erase "$(INTDIR)\chebyshev.obj"
	-@erase "$(INTDIR)\comm_pkg.obj"
	-@erase "$(INTDIR)\communication.obj"
	-@erase "$(INTDIR)\computation.obj"
	-@erase "$(INTDIR)\compute_maximums.obj"
	-@erase "$(INTDIR)\compute_total_concentration.obj"
	-@erase "$(INTDIR)\constant_porosity.obj"
	-@erase "$(INTDIR)\constantRF.obj"
	-@erase "$(INTDIR)\copy.obj"
	-@erase "$(INTDIR)\create_grid.obj"
	-@erase "$(INTDIR)\diag_scale.obj"
	-@erase "$(INTDIR)\diffuse_lb.obj"
	-@erase "$(INTDIR)\discretize_pressure.obj"
	-@erase "$(INTDIR)\distribute_usergrid.obj"
	-@erase "$(INTDIR)\dpofa.obj"
	-@erase "$(INTDIR)\dposl.obj"
	-@erase "$(INTDIR)\drand48.obj"
	-@erase "$(INTDIR)\erand48.obj"
	-@erase "$(INTDIR)\gauinv.obj"
	-@erase "$(INTDIR)\general.obj"
	-@erase "$(INTDIR)\geom_t_solid.obj"
	-@erase "$(INTDIR)\geometry.obj"
	-@erase "$(INTDIR)\globals.obj"
	-@erase "$(INTDIR)\grgeom_list.obj"
	-@erase "$(INTDIR)\grgeom_octree.obj"
	-@erase "$(INTDIR)\grgeometry.obj"
	-@erase "$(INTDIR)\grid.obj"
	-@erase "$(INTDIR)\hbt.obj"
	-@erase "$(INTDIR)\Header.obj"
	-@erase "$(INTDIR)\infinity_norm.obj"
	-@erase "$(INTDIR)\innerprod.obj"
	-@erase "$(INTDIR)\input_database.obj"
	-@erase "$(INTDIR)\inputRF.obj"
	-@erase "$(INTDIR)\iterativ.obj"
	-@erase "$(INTDIR)\jrand48.obj"
	-@erase "$(INTDIR)\kinsol.obj"
	-@erase "$(INTDIR)\kinsol_nonlin_solver.obj"
	-@erase "$(INTDIR)\kinsol_pc.obj"
	-@erase "$(INTDIR)\kinspgmr.obj"
	-@erase "$(INTDIR)\l2_error_norm.obj"
	-@erase "$(INTDIR)\line_process.obj"
	-@erase "$(INTDIR)\llnlmath.obj"
	-@erase "$(INTDIR)\logging.obj"
	-@erase "$(INTDIR)\lrand48.obj"
	-@erase "$(INTDIR)\matdiag_scale.obj"
	-@erase "$(INTDIR)\matrix.obj"
	-@erase "$(INTDIR)\matvec.obj"
	-@erase "$(INTDIR)\max_field_value.obj"
	-@erase "$(INTDIR)\mg_semi.obj"
	-@erase "$(INTDIR)\mg_semi_prolong.obj"
	-@erase "$(INTDIR)\mg_semi_restrict.obj"
	-@erase "$(INTDIR)\mrand48.obj"
	-@erase "$(INTDIR)\n_vector.obj"
	-@erase "$(INTDIR)\new_endpts.obj"
	-@erase "$(INTDIR)\nl_function_eval.obj"
	-@erase "$(INTDIR)\nodiag_scale.obj"
	-@erase "$(INTDIR)\nrand48.obj"
	-@erase "$(INTDIR)\parflow.obj"
	-@erase "$(INTDIR)\pcg.obj"
	-@erase "$(INTDIR)\permeability_face.obj"
	-@erase "$(INTDIR)\perturb_lb.obj"
	-@erase "$(INTDIR)\pf_module.obj"
	-@erase "$(INTDIR)\pfield.obj"
	-@erase "$(INTDIR)\pgsRF.obj"
	-@erase "$(INTDIR)\phase_velocity_face.obj"
	-@erase "$(INTDIR)\ppcg.obj"
	-@erase "$(INTDIR)\printgrid.obj"
	-@erase "$(INTDIR)\printmatrix.obj"
	-@erase "$(INTDIR)\printvector.obj"
	-@erase "$(INTDIR)\problem.obj"
	-@erase "$(INTDIR)\problem_bc.obj"
	-@erase "$(INTDIR)\problem_bc_internal.obj"
	-@erase "$(INTDIR)\problem_bc_phase_saturation.obj"
	-@erase "$(INTDIR)\problem_bc_pressure.obj"
	-@erase "$(INTDIR)\problem_capillary_pressure.obj"
	-@erase "$(INTDIR)\problem_domain.obj"
	-@erase "$(INTDIR)\problem_eval.obj"
	-@erase "$(INTDIR)\problem_geometries.obj"
	-@erase "$(INTDIR)\problem_ic_phase_concen.obj"
	-@erase "$(INTDIR)\problem_ic_phase_pressure.obj"
	-@erase "$(INTDIR)\problem_ic_phase_satur.obj"
	-@erase "$(INTDIR)\problem_phase_density.obj"
	-@erase "$(INTDIR)\problem_phase_mobility.obj"
	-@erase "$(INTDIR)\problem_phase_rel_perm.obj"
	-@erase "$(INTDIR)\problem_phase_source.obj"
	-@erase "$(INTDIR)\problem_porosity.obj"
	-@erase "$(INTDIR)\problem_retardation.obj"
	-@erase "$(INTDIR)\problem_richards_bc_internal.obj"
	-@erase "$(INTDIR)\problem_saturation.obj"
	-@erase "$(INTDIR)\problem_saturation_constitutive.obj"
	-@erase "$(INTDIR)\random.obj"
	-@erase "$(INTDIR)\ratqr.obj"
	-@erase "$(INTDIR)\rb_GS_point.obj"
	-@erase "$(INTDIR)\read_parflow_binary.obj"
	-@erase "$(INTDIR)\reg_from_stenc.obj"
	-@erase "$(INTDIR)\region.obj"
	-@erase "$(INTDIR)\richards_jacobian_eval.obj"
	-@erase "$(INTDIR)\sadvect.win32.obj"
	-@erase "$(INTDIR)\sadvection_godunov.obj"
	-@erase "$(INTDIR)\scale.obj"
	-@erase "$(INTDIR)\select_time_step.obj"
	-@erase "$(INTDIR)\set_problem_data.obj"
	-@erase "$(INTDIR)\signal.obj"
	-@erase "$(INTDIR)\sim_shear.obj"
	-@erase "$(INTDIR)\solver.obj"
	-@erase "$(INTDIR)\solver_impes.obj"
	-@erase "$(INTDIR)\solver_lb.obj"
	-@erase "$(INTDIR)\solver_richards.obj"
	-@erase "$(INTDIR)\spgmr.obj"
	-@erase "$(INTDIR)\srand48.obj"
	-@erase "$(INTDIR)\subsrf_sim.obj"
	-@erase "$(INTDIR)\time_cycle_data.obj"
	-@erase "$(INTDIR)\timing.obj"
	-@erase "$(INTDIR)\total_velocity_face.obj"
	-@erase "$(INTDIR)\turning_bandsRF.obj"
	-@erase "$(INTDIR)\unix_port.obj"
	-@erase "$(INTDIR)\usergrid_input.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vector.obj"
	-@erase "$(INTDIR)\vector_utilities.obj"
	-@erase "$(INTDIR)\w_jacobi.obj"
	-@erase "$(INTDIR)\well.obj"
	-@erase "$(INTDIR)\well_package.obj"
	-@erase "$(INTDIR)\wells_lb.obj"
	-@erase "$(INTDIR)\write_parflow_binary.obj"
	-@erase "$(OUTDIR)\ParFlow.exe"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

F90=df.exe
CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GX /O2 /Ob2 /I "..\..\amps\win32" /I "..\..\config" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /Fp"$(INTDIR)\ParFlow.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\ParFlow.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:no /pdb:"$(OUTDIR)\ParFlow.pdb" /machine:I386 /out:"$(OUTDIR)\ParFlow.exe" 
LINK32_OBJS= \
	"$(INTDIR)\_rand48.obj" \
	"$(INTDIR)\advect.win32.obj" \
	"$(INTDIR)\advection_godunov.obj" \
	"$(INTDIR)\amps_allreduce.obj" \
	"$(INTDIR)\amps_bcast.obj" \
	"$(INTDIR)\amps_clear.obj" \
	"$(INTDIR)\amps_createinvoice.obj" \
	"$(INTDIR)\amps_exchange.obj" \
	"$(INTDIR)\amps_ffopen.obj" \
	"$(INTDIR)\amps_finalize.obj" \
	"$(INTDIR)\amps_find_powers.obj" \
	"$(INTDIR)\amps_fopen.obj" \
	"$(INTDIR)\amps_init.obj" \
	"$(INTDIR)\amps_invoice.obj" \
	"$(INTDIR)\amps_io.obj" \
	"$(INTDIR)\amps_irecv.obj" \
	"$(INTDIR)\amps_newhandle.obj" \
	"$(INTDIR)\amps_newpackage.obj" \
	"$(INTDIR)\amps_pack.obj" \
	"$(INTDIR)\amps_recv.obj" \
	"$(INTDIR)\amps_send.obj" \
	"$(INTDIR)\amps_sfbcast.obj" \
	"$(INTDIR)\amps_sfclose.obj" \
	"$(INTDIR)\amps_sfopen.obj" \
	"$(INTDIR)\amps_sizeofinvoice.obj" \
	"$(INTDIR)\amps_sync.obj" \
	"$(INTDIR)\amps_test.obj" \
	"$(INTDIR)\amps_unpack.obj" \
	"$(INTDIR)\amps_vector.obj" \
	"$(INTDIR)\amps_wait.obj" \
	"$(INTDIR)\axpy.obj" \
	"$(INTDIR)\background.obj" \
	"$(INTDIR)\bc_lb.obj" \
	"$(INTDIR)\bc_pressure.obj" \
	"$(INTDIR)\bc_pressure_package.obj" \
	"$(INTDIR)\calc_elevations.obj" \
	"$(INTDIR)\cghs.obj" \
	"$(INTDIR)\char_vector.obj" \
	"$(INTDIR)\chebyshev.obj" \
	"$(INTDIR)\comm_pkg.obj" \
	"$(INTDIR)\communication.obj" \
	"$(INTDIR)\computation.obj" \
	"$(INTDIR)\compute_maximums.obj" \
	"$(INTDIR)\compute_total_concentration.obj" \
	"$(INTDIR)\constant_porosity.obj" \
	"$(INTDIR)\constantRF.obj" \
	"$(INTDIR)\copy.obj" \
	"$(INTDIR)\create_grid.obj" \
	"$(INTDIR)\diag_scale.obj" \
	"$(INTDIR)\diffuse_lb.obj" \
	"$(INTDIR)\discretize_pressure.obj" \
	"$(INTDIR)\distribute_usergrid.obj" \
	"$(INTDIR)\dpofa.obj" \
	"$(INTDIR)\dposl.obj" \
	"$(INTDIR)\drand48.obj" \
	"$(INTDIR)\erand48.obj" \
	"$(INTDIR)\gauinv.obj" \
	"$(INTDIR)\general.obj" \
	"$(INTDIR)\geom_t_solid.obj" \
	"$(INTDIR)\geometry.obj" \
	"$(INTDIR)\globals.obj" \
	"$(INTDIR)\grgeom_list.obj" \
	"$(INTDIR)\grgeom_octree.obj" \
	"$(INTDIR)\grgeometry.obj" \
	"$(INTDIR)\grid.obj" \
	"$(INTDIR)\hbt.obj" \
	"$(INTDIR)\Header.obj" \
	"$(INTDIR)\infinity_norm.obj" \
	"$(INTDIR)\innerprod.obj" \
	"$(INTDIR)\input_database.obj" \
	"$(INTDIR)\inputRF.obj" \
	"$(INTDIR)\iterativ.obj" \
	"$(INTDIR)\jrand48.obj" \
	"$(INTDIR)\kinsol.obj" \
	"$(INTDIR)\kinsol_nonlin_solver.obj" \
	"$(INTDIR)\kinsol_pc.obj" \
	"$(INTDIR)\kinspgmr.obj" \
	"$(INTDIR)\l2_error_norm.obj" \
	"$(INTDIR)\line_process.obj" \
	"$(INTDIR)\llnlmath.obj" \
	"$(INTDIR)\logging.obj" \
	"$(INTDIR)\lrand48.obj" \
	"$(INTDIR)\matdiag_scale.obj" \
	"$(INTDIR)\matrix.obj" \
	"$(INTDIR)\matvec.obj" \
	"$(INTDIR)\max_field_value.obj" \
	"$(INTDIR)\mg_semi.obj" \
	"$(INTDIR)\mg_semi_prolong.obj" \
	"$(INTDIR)\mg_semi_restrict.obj" \
	"$(INTDIR)\mrand48.obj" \
	"$(INTDIR)\n_vector.obj" \
	"$(INTDIR)\new_endpts.obj" \
	"$(INTDIR)\nl_function_eval.obj" \
	"$(INTDIR)\nodiag_scale.obj" \
	"$(INTDIR)\nrand48.obj" \
	"$(INTDIR)\parflow.obj" \
	"$(INTDIR)\pcg.obj" \
	"$(INTDIR)\permeability_face.obj" \
	"$(INTDIR)\perturb_lb.obj" \
	"$(INTDIR)\pf_module.obj" \
	"$(INTDIR)\pfield.obj" \
	"$(INTDIR)\pgsRF.obj" \
	"$(INTDIR)\phase_velocity_face.obj" \
	"$(INTDIR)\ppcg.obj" \
	"$(INTDIR)\printgrid.obj" \
	"$(INTDIR)\printmatrix.obj" \
	"$(INTDIR)\printvector.obj" \
	"$(INTDIR)\problem.obj" \
	"$(INTDIR)\problem_bc.obj" \
	"$(INTDIR)\problem_bc_internal.obj" \
	"$(INTDIR)\problem_bc_phase_saturation.obj" \
	"$(INTDIR)\problem_bc_pressure.obj" \
	"$(INTDIR)\problem_capillary_pressure.obj" \
	"$(INTDIR)\problem_domain.obj" \
	"$(INTDIR)\problem_eval.obj" \
	"$(INTDIR)\problem_geometries.obj" \
	"$(INTDIR)\problem_ic_phase_concen.obj" \
	"$(INTDIR)\problem_ic_phase_pressure.obj" \
	"$(INTDIR)\problem_ic_phase_satur.obj" \
	"$(INTDIR)\problem_phase_density.obj" \
	"$(INTDIR)\problem_phase_mobility.obj" \
	"$(INTDIR)\problem_phase_rel_perm.obj" \
	"$(INTDIR)\problem_phase_source.obj" \
	"$(INTDIR)\problem_porosity.obj" \
	"$(INTDIR)\problem_retardation.obj" \
	"$(INTDIR)\problem_richards_bc_internal.obj" \
	"$(INTDIR)\problem_saturation.obj" \
	"$(INTDIR)\problem_saturation_constitutive.obj" \
	"$(INTDIR)\random.obj" \
	"$(INTDIR)\ratqr.obj" \
	"$(INTDIR)\rb_GS_point.obj" \
	"$(INTDIR)\read_parflow_binary.obj" \
	"$(INTDIR)\reg_from_stenc.obj" \
	"$(INTDIR)\region.obj" \
	"$(INTDIR)\richards_jacobian_eval.obj" \
	"$(INTDIR)\sadvect.win32.obj" \
	"$(INTDIR)\sadvection_godunov.obj" \
	"$(INTDIR)\scale.obj" \
	"$(INTDIR)\select_time_step.obj" \
	"$(INTDIR)\set_problem_data.obj" \
	"$(INTDIR)\signal.obj" \
	"$(INTDIR)\sim_shear.obj" \
	"$(INTDIR)\solver.obj" \
	"$(INTDIR)\solver_impes.obj" \
	"$(INTDIR)\solver_lb.obj" \
	"$(INTDIR)\solver_richards.obj" \
	"$(INTDIR)\spgmr.obj" \
	"$(INTDIR)\srand48.obj" \
	"$(INTDIR)\subsrf_sim.obj" \
	"$(INTDIR)\time_cycle_data.obj" \
	"$(INTDIR)\timing.obj" \
	"$(INTDIR)\total_velocity_face.obj" \
	"$(INTDIR)\turning_bandsRF.obj" \
	"$(INTDIR)\unix_port.obj" \
	"$(INTDIR)\usergrid_input.obj" \
	"$(INTDIR)\vector.obj" \
	"$(INTDIR)\vector_utilities.obj" \
	"$(INTDIR)\w_jacobi.obj" \
	"$(INTDIR)\well.obj" \
	"$(INTDIR)\well_package.obj" \
	"$(INTDIR)\wells_lb.obj" \
	"$(INTDIR)\write_parflow_binary.obj"

"$(OUTDIR)\ParFlow.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\ParFlow.exe" "$(OUTDIR)\ParFlow.bsc"


CLEAN :
	-@erase "$(INTDIR)\_rand48.obj"
	-@erase "$(INTDIR)\_rand48.sbr"
	-@erase "$(INTDIR)\advect.win32.obj"
	-@erase "$(INTDIR)\advect.win32.sbr"
	-@erase "$(INTDIR)\advection_godunov.obj"
	-@erase "$(INTDIR)\advection_godunov.sbr"
	-@erase "$(INTDIR)\amps_allreduce.obj"
	-@erase "$(INTDIR)\amps_allreduce.sbr"
	-@erase "$(INTDIR)\amps_bcast.obj"
	-@erase "$(INTDIR)\amps_bcast.sbr"
	-@erase "$(INTDIR)\amps_clear.obj"
	-@erase "$(INTDIR)\amps_clear.sbr"
	-@erase "$(INTDIR)\amps_createinvoice.obj"
	-@erase "$(INTDIR)\amps_createinvoice.sbr"
	-@erase "$(INTDIR)\amps_exchange.obj"
	-@erase "$(INTDIR)\amps_exchange.sbr"
	-@erase "$(INTDIR)\amps_ffopen.obj"
	-@erase "$(INTDIR)\amps_ffopen.sbr"
	-@erase "$(INTDIR)\amps_finalize.obj"
	-@erase "$(INTDIR)\amps_finalize.sbr"
	-@erase "$(INTDIR)\amps_find_powers.obj"
	-@erase "$(INTDIR)\amps_find_powers.sbr"
	-@erase "$(INTDIR)\amps_fopen.obj"
	-@erase "$(INTDIR)\amps_fopen.sbr"
	-@erase "$(INTDIR)\amps_init.obj"
	-@erase "$(INTDIR)\amps_init.sbr"
	-@erase "$(INTDIR)\amps_invoice.obj"
	-@erase "$(INTDIR)\amps_invoice.sbr"
	-@erase "$(INTDIR)\amps_io.obj"
	-@erase "$(INTDIR)\amps_io.sbr"
	-@erase "$(INTDIR)\amps_irecv.obj"
	-@erase "$(INTDIR)\amps_irecv.sbr"
	-@erase "$(INTDIR)\amps_newhandle.obj"
	-@erase "$(INTDIR)\amps_newhandle.sbr"
	-@erase "$(INTDIR)\amps_newpackage.obj"
	-@erase "$(INTDIR)\amps_newpackage.sbr"
	-@erase "$(INTDIR)\amps_pack.obj"
	-@erase "$(INTDIR)\amps_pack.sbr"
	-@erase "$(INTDIR)\amps_recv.obj"
	-@erase "$(INTDIR)\amps_recv.sbr"
	-@erase "$(INTDIR)\amps_send.obj"
	-@erase "$(INTDIR)\amps_send.sbr"
	-@erase "$(INTDIR)\amps_sfbcast.obj"
	-@erase "$(INTDIR)\amps_sfbcast.sbr"
	-@erase "$(INTDIR)\amps_sfclose.obj"
	-@erase "$(INTDIR)\amps_sfclose.sbr"
	-@erase "$(INTDIR)\amps_sfopen.obj"
	-@erase "$(INTDIR)\amps_sfopen.sbr"
	-@erase "$(INTDIR)\amps_sizeofinvoice.obj"
	-@erase "$(INTDIR)\amps_sizeofinvoice.sbr"
	-@erase "$(INTDIR)\amps_sync.obj"
	-@erase "$(INTDIR)\amps_sync.sbr"
	-@erase "$(INTDIR)\amps_test.obj"
	-@erase "$(INTDIR)\amps_test.sbr"
	-@erase "$(INTDIR)\amps_unpack.obj"
	-@erase "$(INTDIR)\amps_unpack.sbr"
	-@erase "$(INTDIR)\amps_vector.obj"
	-@erase "$(INTDIR)\amps_vector.sbr"
	-@erase "$(INTDIR)\amps_wait.obj"
	-@erase "$(INTDIR)\amps_wait.sbr"
	-@erase "$(INTDIR)\axpy.obj"
	-@erase "$(INTDIR)\axpy.sbr"
	-@erase "$(INTDIR)\background.obj"
	-@erase "$(INTDIR)\background.sbr"
	-@erase "$(INTDIR)\bc_lb.obj"
	-@erase "$(INTDIR)\bc_lb.sbr"
	-@erase "$(INTDIR)\bc_pressure.obj"
	-@erase "$(INTDIR)\bc_pressure.sbr"
	-@erase "$(INTDIR)\bc_pressure_package.obj"
	-@erase "$(INTDIR)\bc_pressure_package.sbr"
	-@erase "$(INTDIR)\calc_elevations.obj"
	-@erase "$(INTDIR)\calc_elevations.sbr"
	-@erase "$(INTDIR)\cghs.obj"
	-@erase "$(INTDIR)\cghs.sbr"
	-@erase "$(INTDIR)\char_vector.obj"
	-@erase "$(INTDIR)\char_vector.sbr"
	-@erase "$(INTDIR)\chebyshev.obj"
	-@erase "$(INTDIR)\chebyshev.sbr"
	-@erase "$(INTDIR)\comm_pkg.obj"
	-@erase "$(INTDIR)\comm_pkg.sbr"
	-@erase "$(INTDIR)\communication.obj"
	-@erase "$(INTDIR)\communication.sbr"
	-@erase "$(INTDIR)\computation.obj"
	-@erase "$(INTDIR)\computation.sbr"
	-@erase "$(INTDIR)\compute_maximums.obj"
	-@erase "$(INTDIR)\compute_maximums.sbr"
	-@erase "$(INTDIR)\compute_total_concentration.obj"
	-@erase "$(INTDIR)\compute_total_concentration.sbr"
	-@erase "$(INTDIR)\constant_porosity.obj"
	-@erase "$(INTDIR)\constant_porosity.sbr"
	-@erase "$(INTDIR)\constantRF.obj"
	-@erase "$(INTDIR)\constantRF.sbr"
	-@erase "$(INTDIR)\copy.obj"
	-@erase "$(INTDIR)\copy.sbr"
	-@erase "$(INTDIR)\create_grid.obj"
	-@erase "$(INTDIR)\create_grid.sbr"
	-@erase "$(INTDIR)\diag_scale.obj"
	-@erase "$(INTDIR)\diag_scale.sbr"
	-@erase "$(INTDIR)\diffuse_lb.obj"
	-@erase "$(INTDIR)\diffuse_lb.sbr"
	-@erase "$(INTDIR)\discretize_pressure.obj"
	-@erase "$(INTDIR)\discretize_pressure.sbr"
	-@erase "$(INTDIR)\distribute_usergrid.obj"
	-@erase "$(INTDIR)\distribute_usergrid.sbr"
	-@erase "$(INTDIR)\dpofa.obj"
	-@erase "$(INTDIR)\dpofa.sbr"
	-@erase "$(INTDIR)\dposl.obj"
	-@erase "$(INTDIR)\dposl.sbr"
	-@erase "$(INTDIR)\drand48.obj"
	-@erase "$(INTDIR)\drand48.sbr"
	-@erase "$(INTDIR)\erand48.obj"
	-@erase "$(INTDIR)\erand48.sbr"
	-@erase "$(INTDIR)\gauinv.obj"
	-@erase "$(INTDIR)\gauinv.sbr"
	-@erase "$(INTDIR)\general.obj"
	-@erase "$(INTDIR)\general.sbr"
	-@erase "$(INTDIR)\geom_t_solid.obj"
	-@erase "$(INTDIR)\geom_t_solid.sbr"
	-@erase "$(INTDIR)\geometry.obj"
	-@erase "$(INTDIR)\geometry.sbr"
	-@erase "$(INTDIR)\globals.obj"
	-@erase "$(INTDIR)\globals.sbr"
	-@erase "$(INTDIR)\grgeom_list.obj"
	-@erase "$(INTDIR)\grgeom_list.sbr"
	-@erase "$(INTDIR)\grgeom_octree.obj"
	-@erase "$(INTDIR)\grgeom_octree.sbr"
	-@erase "$(INTDIR)\grgeometry.obj"
	-@erase "$(INTDIR)\grgeometry.sbr"
	-@erase "$(INTDIR)\grid.obj"
	-@erase "$(INTDIR)\grid.sbr"
	-@erase "$(INTDIR)\hbt.obj"
	-@erase "$(INTDIR)\hbt.sbr"
	-@erase "$(INTDIR)\Header.obj"
	-@erase "$(INTDIR)\Header.sbr"
	-@erase "$(INTDIR)\infinity_norm.obj"
	-@erase "$(INTDIR)\infinity_norm.sbr"
	-@erase "$(INTDIR)\innerprod.obj"
	-@erase "$(INTDIR)\innerprod.sbr"
	-@erase "$(INTDIR)\input_database.obj"
	-@erase "$(INTDIR)\input_database.sbr"
	-@erase "$(INTDIR)\inputRF.obj"
	-@erase "$(INTDIR)\inputRF.sbr"
	-@erase "$(INTDIR)\iterativ.obj"
	-@erase "$(INTDIR)\iterativ.sbr"
	-@erase "$(INTDIR)\jrand48.obj"
	-@erase "$(INTDIR)\jrand48.sbr"
	-@erase "$(INTDIR)\kinsol.obj"
	-@erase "$(INTDIR)\kinsol.sbr"
	-@erase "$(INTDIR)\kinsol_nonlin_solver.obj"
	-@erase "$(INTDIR)\kinsol_nonlin_solver.sbr"
	-@erase "$(INTDIR)\kinsol_pc.obj"
	-@erase "$(INTDIR)\kinsol_pc.sbr"
	-@erase "$(INTDIR)\kinspgmr.obj"
	-@erase "$(INTDIR)\kinspgmr.sbr"
	-@erase "$(INTDIR)\l2_error_norm.obj"
	-@erase "$(INTDIR)\l2_error_norm.sbr"
	-@erase "$(INTDIR)\line_process.obj"
	-@erase "$(INTDIR)\line_process.sbr"
	-@erase "$(INTDIR)\llnlmath.obj"
	-@erase "$(INTDIR)\llnlmath.sbr"
	-@erase "$(INTDIR)\logging.obj"
	-@erase "$(INTDIR)\logging.sbr"
	-@erase "$(INTDIR)\lrand48.obj"
	-@erase "$(INTDIR)\lrand48.sbr"
	-@erase "$(INTDIR)\matdiag_scale.obj"
	-@erase "$(INTDIR)\matdiag_scale.sbr"
	-@erase "$(INTDIR)\matrix.obj"
	-@erase "$(INTDIR)\matrix.sbr"
	-@erase "$(INTDIR)\matvec.obj"
	-@erase "$(INTDIR)\matvec.sbr"
	-@erase "$(INTDIR)\max_field_value.obj"
	-@erase "$(INTDIR)\max_field_value.sbr"
	-@erase "$(INTDIR)\mg_semi.obj"
	-@erase "$(INTDIR)\mg_semi.sbr"
	-@erase "$(INTDIR)\mg_semi_prolong.obj"
	-@erase "$(INTDIR)\mg_semi_prolong.sbr"
	-@erase "$(INTDIR)\mg_semi_restrict.obj"
	-@erase "$(INTDIR)\mg_semi_restrict.sbr"
	-@erase "$(INTDIR)\mrand48.obj"
	-@erase "$(INTDIR)\mrand48.sbr"
	-@erase "$(INTDIR)\n_vector.obj"
	-@erase "$(INTDIR)\n_vector.sbr"
	-@erase "$(INTDIR)\new_endpts.obj"
	-@erase "$(INTDIR)\new_endpts.sbr"
	-@erase "$(INTDIR)\nl_function_eval.obj"
	-@erase "$(INTDIR)\nl_function_eval.sbr"
	-@erase "$(INTDIR)\nodiag_scale.obj"
	-@erase "$(INTDIR)\nodiag_scale.sbr"
	-@erase "$(INTDIR)\nrand48.obj"
	-@erase "$(INTDIR)\nrand48.sbr"
	-@erase "$(INTDIR)\parflow.obj"
	-@erase "$(INTDIR)\parflow.sbr"
	-@erase "$(INTDIR)\pcg.obj"
	-@erase "$(INTDIR)\pcg.sbr"
	-@erase "$(INTDIR)\permeability_face.obj"
	-@erase "$(INTDIR)\permeability_face.sbr"
	-@erase "$(INTDIR)\perturb_lb.obj"
	-@erase "$(INTDIR)\perturb_lb.sbr"
	-@erase "$(INTDIR)\pf_module.obj"
	-@erase "$(INTDIR)\pf_module.sbr"
	-@erase "$(INTDIR)\pfield.obj"
	-@erase "$(INTDIR)\pfield.sbr"
	-@erase "$(INTDIR)\pgsRF.obj"
	-@erase "$(INTDIR)\pgsRF.sbr"
	-@erase "$(INTDIR)\phase_velocity_face.obj"
	-@erase "$(INTDIR)\phase_velocity_face.sbr"
	-@erase "$(INTDIR)\ppcg.obj"
	-@erase "$(INTDIR)\ppcg.sbr"
	-@erase "$(INTDIR)\printgrid.obj"
	-@erase "$(INTDIR)\printgrid.sbr"
	-@erase "$(INTDIR)\printmatrix.obj"
	-@erase "$(INTDIR)\printmatrix.sbr"
	-@erase "$(INTDIR)\printvector.obj"
	-@erase "$(INTDIR)\printvector.sbr"
	-@erase "$(INTDIR)\problem.obj"
	-@erase "$(INTDIR)\problem.sbr"
	-@erase "$(INTDIR)\problem_bc.obj"
	-@erase "$(INTDIR)\problem_bc.sbr"
	-@erase "$(INTDIR)\problem_bc_internal.obj"
	-@erase "$(INTDIR)\problem_bc_internal.sbr"
	-@erase "$(INTDIR)\problem_bc_phase_saturation.obj"
	-@erase "$(INTDIR)\problem_bc_phase_saturation.sbr"
	-@erase "$(INTDIR)\problem_bc_pressure.obj"
	-@erase "$(INTDIR)\problem_bc_pressure.sbr"
	-@erase "$(INTDIR)\problem_capillary_pressure.obj"
	-@erase "$(INTDIR)\problem_capillary_pressure.sbr"
	-@erase "$(INTDIR)\problem_domain.obj"
	-@erase "$(INTDIR)\problem_domain.sbr"
	-@erase "$(INTDIR)\problem_eval.obj"
	-@erase "$(INTDIR)\problem_eval.sbr"
	-@erase "$(INTDIR)\problem_geometries.obj"
	-@erase "$(INTDIR)\problem_geometries.sbr"
	-@erase "$(INTDIR)\problem_ic_phase_concen.obj"
	-@erase "$(INTDIR)\problem_ic_phase_concen.sbr"
	-@erase "$(INTDIR)\problem_ic_phase_pressure.obj"
	-@erase "$(INTDIR)\problem_ic_phase_pressure.sbr"
	-@erase "$(INTDIR)\problem_ic_phase_satur.obj"
	-@erase "$(INTDIR)\problem_ic_phase_satur.sbr"
	-@erase "$(INTDIR)\problem_phase_density.obj"
	-@erase "$(INTDIR)\problem_phase_density.sbr"
	-@erase "$(INTDIR)\problem_phase_mobility.obj"
	-@erase "$(INTDIR)\problem_phase_mobility.sbr"
	-@erase "$(INTDIR)\problem_phase_rel_perm.obj"
	-@erase "$(INTDIR)\problem_phase_rel_perm.sbr"
	-@erase "$(INTDIR)\problem_phase_source.obj"
	-@erase "$(INTDIR)\problem_phase_source.sbr"
	-@erase "$(INTDIR)\problem_porosity.obj"
	-@erase "$(INTDIR)\problem_porosity.sbr"
	-@erase "$(INTDIR)\problem_retardation.obj"
	-@erase "$(INTDIR)\problem_retardation.sbr"
	-@erase "$(INTDIR)\problem_richards_bc_internal.obj"
	-@erase "$(INTDIR)\problem_richards_bc_internal.sbr"
	-@erase "$(INTDIR)\problem_saturation.obj"
	-@erase "$(INTDIR)\problem_saturation.sbr"
	-@erase "$(INTDIR)\problem_saturation_constitutive.obj"
	-@erase "$(INTDIR)\problem_saturation_constitutive.sbr"
	-@erase "$(INTDIR)\random.obj"
	-@erase "$(INTDIR)\random.sbr"
	-@erase "$(INTDIR)\ratqr.obj"
	-@erase "$(INTDIR)\ratqr.sbr"
	-@erase "$(INTDIR)\rb_GS_point.obj"
	-@erase "$(INTDIR)\rb_GS_point.sbr"
	-@erase "$(INTDIR)\read_parflow_binary.obj"
	-@erase "$(INTDIR)\read_parflow_binary.sbr"
	-@erase "$(INTDIR)\reg_from_stenc.obj"
	-@erase "$(INTDIR)\reg_from_stenc.sbr"
	-@erase "$(INTDIR)\region.obj"
	-@erase "$(INTDIR)\region.sbr"
	-@erase "$(INTDIR)\richards_jacobian_eval.obj"
	-@erase "$(INTDIR)\richards_jacobian_eval.sbr"
	-@erase "$(INTDIR)\sadvect.win32.obj"
	-@erase "$(INTDIR)\sadvect.win32.sbr"
	-@erase "$(INTDIR)\sadvection_godunov.obj"
	-@erase "$(INTDIR)\sadvection_godunov.sbr"
	-@erase "$(INTDIR)\scale.obj"
	-@erase "$(INTDIR)\scale.sbr"
	-@erase "$(INTDIR)\select_time_step.obj"
	-@erase "$(INTDIR)\select_time_step.sbr"
	-@erase "$(INTDIR)\set_problem_data.obj"
	-@erase "$(INTDIR)\set_problem_data.sbr"
	-@erase "$(INTDIR)\signal.obj"
	-@erase "$(INTDIR)\signal.sbr"
	-@erase "$(INTDIR)\sim_shear.obj"
	-@erase "$(INTDIR)\sim_shear.sbr"
	-@erase "$(INTDIR)\solver.obj"
	-@erase "$(INTDIR)\solver.sbr"
	-@erase "$(INTDIR)\solver_impes.obj"
	-@erase "$(INTDIR)\solver_impes.sbr"
	-@erase "$(INTDIR)\solver_lb.obj"
	-@erase "$(INTDIR)\solver_lb.sbr"
	-@erase "$(INTDIR)\solver_richards.obj"
	-@erase "$(INTDIR)\solver_richards.sbr"
	-@erase "$(INTDIR)\spgmr.obj"
	-@erase "$(INTDIR)\spgmr.sbr"
	-@erase "$(INTDIR)\srand48.obj"
	-@erase "$(INTDIR)\srand48.sbr"
	-@erase "$(INTDIR)\subsrf_sim.obj"
	-@erase "$(INTDIR)\subsrf_sim.sbr"
	-@erase "$(INTDIR)\time_cycle_data.obj"
	-@erase "$(INTDIR)\time_cycle_data.sbr"
	-@erase "$(INTDIR)\timing.obj"
	-@erase "$(INTDIR)\timing.sbr"
	-@erase "$(INTDIR)\total_velocity_face.obj"
	-@erase "$(INTDIR)\total_velocity_face.sbr"
	-@erase "$(INTDIR)\turning_bandsRF.obj"
	-@erase "$(INTDIR)\turning_bandsRF.sbr"
	-@erase "$(INTDIR)\unix_port.obj"
	-@erase "$(INTDIR)\unix_port.sbr"
	-@erase "$(INTDIR)\usergrid_input.obj"
	-@erase "$(INTDIR)\usergrid_input.sbr"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\vector.obj"
	-@erase "$(INTDIR)\vector.sbr"
	-@erase "$(INTDIR)\vector_utilities.obj"
	-@erase "$(INTDIR)\vector_utilities.sbr"
	-@erase "$(INTDIR)\w_jacobi.obj"
	-@erase "$(INTDIR)\w_jacobi.sbr"
	-@erase "$(INTDIR)\well.obj"
	-@erase "$(INTDIR)\well.sbr"
	-@erase "$(INTDIR)\well_package.obj"
	-@erase "$(INTDIR)\well_package.sbr"
	-@erase "$(INTDIR)\wells_lb.obj"
	-@erase "$(INTDIR)\wells_lb.sbr"
	-@erase "$(INTDIR)\write_parflow_binary.obj"
	-@erase "$(INTDIR)\write_parflow_binary.sbr"
	-@erase "$(OUTDIR)\ParFlow.bsc"
	-@erase "$(OUTDIR)\ParFlow.exe"
	-@erase "$(OUTDIR)\ParFlow.ilk"
	-@erase "$(OUTDIR)\ParFlow.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

F90=df.exe
CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /Gm /GX /ZI /Od /I "..\..\amps\win32" /I "..\..\config" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /FR"$(INTDIR)\\" /Fp"$(INTDIR)\ParFlow.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\ParFlow.bsc" 
BSC32_SBRS= \
	"$(INTDIR)\_rand48.sbr" \
	"$(INTDIR)\advect.win32.sbr" \
	"$(INTDIR)\advection_godunov.sbr" \
	"$(INTDIR)\amps_allreduce.sbr" \
	"$(INTDIR)\amps_bcast.sbr" \
	"$(INTDIR)\amps_clear.sbr" \
	"$(INTDIR)\amps_createinvoice.sbr" \
	"$(INTDIR)\amps_exchange.sbr" \
	"$(INTDIR)\amps_ffopen.sbr" \
	"$(INTDIR)\amps_finalize.sbr" \
	"$(INTDIR)\amps_find_powers.sbr" \
	"$(INTDIR)\amps_fopen.sbr" \
	"$(INTDIR)\amps_init.sbr" \
	"$(INTDIR)\amps_invoice.sbr" \
	"$(INTDIR)\amps_io.sbr" \
	"$(INTDIR)\amps_irecv.sbr" \
	"$(INTDIR)\amps_newhandle.sbr" \
	"$(INTDIR)\amps_newpackage.sbr" \
	"$(INTDIR)\amps_pack.sbr" \
	"$(INTDIR)\amps_recv.sbr" \
	"$(INTDIR)\amps_send.sbr" \
	"$(INTDIR)\amps_sfbcast.sbr" \
	"$(INTDIR)\amps_sfclose.sbr" \
	"$(INTDIR)\amps_sfopen.sbr" \
	"$(INTDIR)\amps_sizeofinvoice.sbr" \
	"$(INTDIR)\amps_sync.sbr" \
	"$(INTDIR)\amps_test.sbr" \
	"$(INTDIR)\amps_unpack.sbr" \
	"$(INTDIR)\amps_vector.sbr" \
	"$(INTDIR)\amps_wait.sbr" \
	"$(INTDIR)\axpy.sbr" \
	"$(INTDIR)\background.sbr" \
	"$(INTDIR)\bc_lb.sbr" \
	"$(INTDIR)\bc_pressure.sbr" \
	"$(INTDIR)\bc_pressure_package.sbr" \
	"$(INTDIR)\calc_elevations.sbr" \
	"$(INTDIR)\cghs.sbr" \
	"$(INTDIR)\char_vector.sbr" \
	"$(INTDIR)\chebyshev.sbr" \
	"$(INTDIR)\comm_pkg.sbr" \
	"$(INTDIR)\communication.sbr" \
	"$(INTDIR)\computation.sbr" \
	"$(INTDIR)\compute_maximums.sbr" \
	"$(INTDIR)\compute_total_concentration.sbr" \
	"$(INTDIR)\constant_porosity.sbr" \
	"$(INTDIR)\constantRF.sbr" \
	"$(INTDIR)\copy.sbr" \
	"$(INTDIR)\create_grid.sbr" \
	"$(INTDIR)\diag_scale.sbr" \
	"$(INTDIR)\diffuse_lb.sbr" \
	"$(INTDIR)\discretize_pressure.sbr" \
	"$(INTDIR)\distribute_usergrid.sbr" \
	"$(INTDIR)\dpofa.sbr" \
	"$(INTDIR)\dposl.sbr" \
	"$(INTDIR)\drand48.sbr" \
	"$(INTDIR)\erand48.sbr" \
	"$(INTDIR)\gauinv.sbr" \
	"$(INTDIR)\general.sbr" \
	"$(INTDIR)\geom_t_solid.sbr" \
	"$(INTDIR)\geometry.sbr" \
	"$(INTDIR)\globals.sbr" \
	"$(INTDIR)\grgeom_list.sbr" \
	"$(INTDIR)\grgeom_octree.sbr" \
	"$(INTDIR)\grgeometry.sbr" \
	"$(INTDIR)\grid.sbr" \
	"$(INTDIR)\hbt.sbr" \
	"$(INTDIR)\Header.sbr" \
	"$(INTDIR)\infinity_norm.sbr" \
	"$(INTDIR)\innerprod.sbr" \
	"$(INTDIR)\input_database.sbr" \
	"$(INTDIR)\inputRF.sbr" \
	"$(INTDIR)\iterativ.sbr" \
	"$(INTDIR)\jrand48.sbr" \
	"$(INTDIR)\kinsol.sbr" \
	"$(INTDIR)\kinsol_nonlin_solver.sbr" \
	"$(INTDIR)\kinsol_pc.sbr" \
	"$(INTDIR)\kinspgmr.sbr" \
	"$(INTDIR)\l2_error_norm.sbr" \
	"$(INTDIR)\line_process.sbr" \
	"$(INTDIR)\llnlmath.sbr" \
	"$(INTDIR)\logging.sbr" \
	"$(INTDIR)\lrand48.sbr" \
	"$(INTDIR)\matdiag_scale.sbr" \
	"$(INTDIR)\matrix.sbr" \
	"$(INTDIR)\matvec.sbr" \
	"$(INTDIR)\max_field_value.sbr" \
	"$(INTDIR)\mg_semi.sbr" \
	"$(INTDIR)\mg_semi_prolong.sbr" \
	"$(INTDIR)\mg_semi_restrict.sbr" \
	"$(INTDIR)\mrand48.sbr" \
	"$(INTDIR)\n_vector.sbr" \
	"$(INTDIR)\new_endpts.sbr" \
	"$(INTDIR)\nl_function_eval.sbr" \
	"$(INTDIR)\nodiag_scale.sbr" \
	"$(INTDIR)\nrand48.sbr" \
	"$(INTDIR)\parflow.sbr" \
	"$(INTDIR)\pcg.sbr" \
	"$(INTDIR)\permeability_face.sbr" \
	"$(INTDIR)\perturb_lb.sbr" \
	"$(INTDIR)\pf_module.sbr" \
	"$(INTDIR)\pfield.sbr" \
	"$(INTDIR)\pgsRF.sbr" \
	"$(INTDIR)\phase_velocity_face.sbr" \
	"$(INTDIR)\ppcg.sbr" \
	"$(INTDIR)\printgrid.sbr" \
	"$(INTDIR)\printmatrix.sbr" \
	"$(INTDIR)\printvector.sbr" \
	"$(INTDIR)\problem.sbr" \
	"$(INTDIR)\problem_bc.sbr" \
	"$(INTDIR)\problem_bc_internal.sbr" \
	"$(INTDIR)\problem_bc_phase_saturation.sbr" \
	"$(INTDIR)\problem_bc_pressure.sbr" \
	"$(INTDIR)\problem_capillary_pressure.sbr" \
	"$(INTDIR)\problem_domain.sbr" \
	"$(INTDIR)\problem_eval.sbr" \
	"$(INTDIR)\problem_geometries.sbr" \
	"$(INTDIR)\problem_ic_phase_concen.sbr" \
	"$(INTDIR)\problem_ic_phase_pressure.sbr" \
	"$(INTDIR)\problem_ic_phase_satur.sbr" \
	"$(INTDIR)\problem_phase_density.sbr" \
	"$(INTDIR)\problem_phase_mobility.sbr" \
	"$(INTDIR)\problem_phase_rel_perm.sbr" \
	"$(INTDIR)\problem_phase_source.sbr" \
	"$(INTDIR)\problem_porosity.sbr" \
	"$(INTDIR)\problem_retardation.sbr" \
	"$(INTDIR)\problem_richards_bc_internal.sbr" \
	"$(INTDIR)\problem_saturation.sbr" \
	"$(INTDIR)\problem_saturation_constitutive.sbr" \
	"$(INTDIR)\random.sbr" \
	"$(INTDIR)\ratqr.sbr" \
	"$(INTDIR)\rb_GS_point.sbr" \
	"$(INTDIR)\read_parflow_binary.sbr" \
	"$(INTDIR)\reg_from_stenc.sbr" \
	"$(INTDIR)\region.sbr" \
	"$(INTDIR)\richards_jacobian_eval.sbr" \
	"$(INTDIR)\sadvect.win32.sbr" \
	"$(INTDIR)\sadvection_godunov.sbr" \
	"$(INTDIR)\scale.sbr" \
	"$(INTDIR)\select_time_step.sbr" \
	"$(INTDIR)\set_problem_data.sbr" \
	"$(INTDIR)\signal.sbr" \
	"$(INTDIR)\sim_shear.sbr" \
	"$(INTDIR)\solver.sbr" \
	"$(INTDIR)\solver_impes.sbr" \
	"$(INTDIR)\solver_lb.sbr" \
	"$(INTDIR)\solver_richards.sbr" \
	"$(INTDIR)\spgmr.sbr" \
	"$(INTDIR)\srand48.sbr" \
	"$(INTDIR)\subsrf_sim.sbr" \
	"$(INTDIR)\time_cycle_data.sbr" \
	"$(INTDIR)\timing.sbr" \
	"$(INTDIR)\total_velocity_face.sbr" \
	"$(INTDIR)\turning_bandsRF.sbr" \
	"$(INTDIR)\unix_port.sbr" \
	"$(INTDIR)\usergrid_input.sbr" \
	"$(INTDIR)\vector.sbr" \
	"$(INTDIR)\vector_utilities.sbr" \
	"$(INTDIR)\w_jacobi.sbr" \
	"$(INTDIR)\well.sbr" \
	"$(INTDIR)\well_package.sbr" \
	"$(INTDIR)\wells_lb.sbr" \
	"$(INTDIR)\write_parflow_binary.sbr"

"$(OUTDIR)\ParFlow.bsc" : "$(OUTDIR)" $(BSC32_SBRS)
    $(BSC32) @<<
  $(BSC32_FLAGS) $(BSC32_SBRS)
<<

LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:yes /pdb:"$(OUTDIR)\ParFlow.pdb" /debug /machine:I386 /out:"$(OUTDIR)\ParFlow.exe" /pdbtype:sept 
LINK32_OBJS= \
	"$(INTDIR)\_rand48.obj" \
	"$(INTDIR)\advect.win32.obj" \
	"$(INTDIR)\advection_godunov.obj" \
	"$(INTDIR)\amps_allreduce.obj" \
	"$(INTDIR)\amps_bcast.obj" \
	"$(INTDIR)\amps_clear.obj" \
	"$(INTDIR)\amps_createinvoice.obj" \
	"$(INTDIR)\amps_exchange.obj" \
	"$(INTDIR)\amps_ffopen.obj" \
	"$(INTDIR)\amps_finalize.obj" \
	"$(INTDIR)\amps_find_powers.obj" \
	"$(INTDIR)\amps_fopen.obj" \
	"$(INTDIR)\amps_init.obj" \
	"$(INTDIR)\amps_invoice.obj" \
	"$(INTDIR)\amps_io.obj" \
	"$(INTDIR)\amps_irecv.obj" \
	"$(INTDIR)\amps_newhandle.obj" \
	"$(INTDIR)\amps_newpackage.obj" \
	"$(INTDIR)\amps_pack.obj" \
	"$(INTDIR)\amps_recv.obj" \
	"$(INTDIR)\amps_send.obj" \
	"$(INTDIR)\amps_sfbcast.obj" \
	"$(INTDIR)\amps_sfclose.obj" \
	"$(INTDIR)\amps_sfopen.obj" \
	"$(INTDIR)\amps_sizeofinvoice.obj" \
	"$(INTDIR)\amps_sync.obj" \
	"$(INTDIR)\amps_test.obj" \
	"$(INTDIR)\amps_unpack.obj" \
	"$(INTDIR)\amps_vector.obj" \
	"$(INTDIR)\amps_wait.obj" \
	"$(INTDIR)\axpy.obj" \
	"$(INTDIR)\background.obj" \
	"$(INTDIR)\bc_lb.obj" \
	"$(INTDIR)\bc_pressure.obj" \
	"$(INTDIR)\bc_pressure_package.obj" \
	"$(INTDIR)\calc_elevations.obj" \
	"$(INTDIR)\cghs.obj" \
	"$(INTDIR)\char_vector.obj" \
	"$(INTDIR)\chebyshev.obj" \
	"$(INTDIR)\comm_pkg.obj" \
	"$(INTDIR)\communication.obj" \
	"$(INTDIR)\computation.obj" \
	"$(INTDIR)\compute_maximums.obj" \
	"$(INTDIR)\compute_total_concentration.obj" \
	"$(INTDIR)\constant_porosity.obj" \
	"$(INTDIR)\constantRF.obj" \
	"$(INTDIR)\copy.obj" \
	"$(INTDIR)\create_grid.obj" \
	"$(INTDIR)\diag_scale.obj" \
	"$(INTDIR)\diffuse_lb.obj" \
	"$(INTDIR)\discretize_pressure.obj" \
	"$(INTDIR)\distribute_usergrid.obj" \
	"$(INTDIR)\dpofa.obj" \
	"$(INTDIR)\dposl.obj" \
	"$(INTDIR)\drand48.obj" \
	"$(INTDIR)\erand48.obj" \
	"$(INTDIR)\gauinv.obj" \
	"$(INTDIR)\general.obj" \
	"$(INTDIR)\geom_t_solid.obj" \
	"$(INTDIR)\geometry.obj" \
	"$(INTDIR)\globals.obj" \
	"$(INTDIR)\grgeom_list.obj" \
	"$(INTDIR)\grgeom_octree.obj" \
	"$(INTDIR)\grgeometry.obj" \
	"$(INTDIR)\grid.obj" \
	"$(INTDIR)\hbt.obj" \
	"$(INTDIR)\Header.obj" \
	"$(INTDIR)\infinity_norm.obj" \
	"$(INTDIR)\innerprod.obj" \
	"$(INTDIR)\input_database.obj" \
	"$(INTDIR)\inputRF.obj" \
	"$(INTDIR)\iterativ.obj" \
	"$(INTDIR)\jrand48.obj" \
	"$(INTDIR)\kinsol.obj" \
	"$(INTDIR)\kinsol_nonlin_solver.obj" \
	"$(INTDIR)\kinsol_pc.obj" \
	"$(INTDIR)\kinspgmr.obj" \
	"$(INTDIR)\l2_error_norm.obj" \
	"$(INTDIR)\line_process.obj" \
	"$(INTDIR)\llnlmath.obj" \
	"$(INTDIR)\logging.obj" \
	"$(INTDIR)\lrand48.obj" \
	"$(INTDIR)\matdiag_scale.obj" \
	"$(INTDIR)\matrix.obj" \
	"$(INTDIR)\matvec.obj" \
	"$(INTDIR)\max_field_value.obj" \
	"$(INTDIR)\mg_semi.obj" \
	"$(INTDIR)\mg_semi_prolong.obj" \
	"$(INTDIR)\mg_semi_restrict.obj" \
	"$(INTDIR)\mrand48.obj" \
	"$(INTDIR)\n_vector.obj" \
	"$(INTDIR)\new_endpts.obj" \
	"$(INTDIR)\nl_function_eval.obj" \
	"$(INTDIR)\nodiag_scale.obj" \
	"$(INTDIR)\nrand48.obj" \
	"$(INTDIR)\parflow.obj" \
	"$(INTDIR)\pcg.obj" \
	"$(INTDIR)\permeability_face.obj" \
	"$(INTDIR)\perturb_lb.obj" \
	"$(INTDIR)\pf_module.obj" \
	"$(INTDIR)\pfield.obj" \
	"$(INTDIR)\pgsRF.obj" \
	"$(INTDIR)\phase_velocity_face.obj" \
	"$(INTDIR)\ppcg.obj" \
	"$(INTDIR)\printgrid.obj" \
	"$(INTDIR)\printmatrix.obj" \
	"$(INTDIR)\printvector.obj" \
	"$(INTDIR)\problem.obj" \
	"$(INTDIR)\problem_bc.obj" \
	"$(INTDIR)\problem_bc_internal.obj" \
	"$(INTDIR)\problem_bc_phase_saturation.obj" \
	"$(INTDIR)\problem_bc_pressure.obj" \
	"$(INTDIR)\problem_capillary_pressure.obj" \
	"$(INTDIR)\problem_domain.obj" \
	"$(INTDIR)\problem_eval.obj" \
	"$(INTDIR)\problem_geometries.obj" \
	"$(INTDIR)\problem_ic_phase_concen.obj" \
	"$(INTDIR)\problem_ic_phase_pressure.obj" \
	"$(INTDIR)\problem_ic_phase_satur.obj" \
	"$(INTDIR)\problem_phase_density.obj" \
	"$(INTDIR)\problem_phase_mobility.obj" \
	"$(INTDIR)\problem_phase_rel_perm.obj" \
	"$(INTDIR)\problem_phase_source.obj" \
	"$(INTDIR)\problem_porosity.obj" \
	"$(INTDIR)\problem_retardation.obj" \
	"$(INTDIR)\problem_richards_bc_internal.obj" \
	"$(INTDIR)\problem_saturation.obj" \
	"$(INTDIR)\problem_saturation_constitutive.obj" \
	"$(INTDIR)\random.obj" \
	"$(INTDIR)\ratqr.obj" \
	"$(INTDIR)\rb_GS_point.obj" \
	"$(INTDIR)\read_parflow_binary.obj" \
	"$(INTDIR)\reg_from_stenc.obj" \
	"$(INTDIR)\region.obj" \
	"$(INTDIR)\richards_jacobian_eval.obj" \
	"$(INTDIR)\sadvect.win32.obj" \
	"$(INTDIR)\sadvection_godunov.obj" \
	"$(INTDIR)\scale.obj" \
	"$(INTDIR)\select_time_step.obj" \
	"$(INTDIR)\set_problem_data.obj" \
	"$(INTDIR)\signal.obj" \
	"$(INTDIR)\sim_shear.obj" \
	"$(INTDIR)\solver.obj" \
	"$(INTDIR)\solver_impes.obj" \
	"$(INTDIR)\solver_lb.obj" \
	"$(INTDIR)\solver_richards.obj" \
	"$(INTDIR)\spgmr.obj" \
	"$(INTDIR)\srand48.obj" \
	"$(INTDIR)\subsrf_sim.obj" \
	"$(INTDIR)\time_cycle_data.obj" \
	"$(INTDIR)\timing.obj" \
	"$(INTDIR)\total_velocity_face.obj" \
	"$(INTDIR)\turning_bandsRF.obj" \
	"$(INTDIR)\unix_port.obj" \
	"$(INTDIR)\usergrid_input.obj" \
	"$(INTDIR)\vector.obj" \
	"$(INTDIR)\vector_utilities.obj" \
	"$(INTDIR)\w_jacobi.obj" \
	"$(INTDIR)\well.obj" \
	"$(INTDIR)\well_package.obj" \
	"$(INTDIR)\wells_lb.obj" \
	"$(INTDIR)\write_parflow_binary.obj"

"$(OUTDIR)\ParFlow.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("ParFlow.dep")
!INCLUDE "ParFlow.dep"
!ELSE 
!MESSAGE Warning: cannot find "ParFlow.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "ParFlow - Win32 Release" || "$(CFG)" == "ParFlow - Win32 Debug"
SOURCE=..\..\amps\win32\_rand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\_rand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\_rand48.obj"	"$(INTDIR)\_rand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\advect.win32.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\advect.win32.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\advect.win32.obj"	"$(INTDIR)\advect.win32.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\advection_godunov.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\advection_godunov.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\advection_godunov.obj"	"$(INTDIR)\advection_godunov.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_allreduce.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_allreduce.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_allreduce.obj"	"$(INTDIR)\amps_allreduce.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_bcast.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_bcast.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_bcast.obj"	"$(INTDIR)\amps_bcast.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_clear.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_clear.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_clear.obj"	"$(INTDIR)\amps_clear.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_createinvoice.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_createinvoice.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_createinvoice.obj"	"$(INTDIR)\amps_createinvoice.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_exchange.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_exchange.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_exchange.obj"	"$(INTDIR)\amps_exchange.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_ffopen.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_ffopen.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_ffopen.obj"	"$(INTDIR)\amps_ffopen.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_finalize.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_finalize.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_finalize.obj"	"$(INTDIR)\amps_finalize.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_find_powers.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_find_powers.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_find_powers.obj"	"$(INTDIR)\amps_find_powers.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_fopen.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_fopen.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_fopen.obj"	"$(INTDIR)\amps_fopen.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_init.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_init.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_init.obj"	"$(INTDIR)\amps_init.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_invoice.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_invoice.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_invoice.obj"	"$(INTDIR)\amps_invoice.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_io.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_io.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_io.obj"	"$(INTDIR)\amps_io.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_irecv.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_irecv.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_irecv.obj"	"$(INTDIR)\amps_irecv.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_newhandle.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_newhandle.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_newhandle.obj"	"$(INTDIR)\amps_newhandle.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_newpackage.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_newpackage.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_newpackage.obj"	"$(INTDIR)\amps_newpackage.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_pack.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_pack.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_pack.obj"	"$(INTDIR)\amps_pack.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_recv.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_recv.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_recv.obj"	"$(INTDIR)\amps_recv.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_send.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_send.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_send.obj"	"$(INTDIR)\amps_send.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_sfbcast.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_sfbcast.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_sfbcast.obj"	"$(INTDIR)\amps_sfbcast.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_sfclose.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_sfclose.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_sfclose.obj"	"$(INTDIR)\amps_sfclose.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\amps_sfopen.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_sfopen.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_sfopen.obj"	"$(INTDIR)\amps_sfopen.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_sizeofinvoice.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_sizeofinvoice.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_sizeofinvoice.obj"	"$(INTDIR)\amps_sizeofinvoice.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_sync.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_sync.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_sync.obj"	"$(INTDIR)\amps_sync.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_test.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_test.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_test.obj"	"$(INTDIR)\amps_test.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_unpack.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_unpack.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_unpack.obj"	"$(INTDIR)\amps_unpack.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_vector.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_vector.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_vector.obj"	"$(INTDIR)\amps_vector.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\amps_wait.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\amps_wait.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\amps_wait.obj"	"$(INTDIR)\amps_wait.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\axpy.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\axpy.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\axpy.obj"	"$(INTDIR)\axpy.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\background.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\background.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\background.obj"	"$(INTDIR)\background.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\bc_lb.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\bc_lb.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\bc_lb.obj"	"$(INTDIR)\bc_lb.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\bc_pressure.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\bc_pressure.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\bc_pressure.obj"	"$(INTDIR)\bc_pressure.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\bc_pressure_package.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\bc_pressure_package.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\bc_pressure_package.obj"	"$(INTDIR)\bc_pressure_package.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\calc_elevations.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\calc_elevations.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\calc_elevations.obj"	"$(INTDIR)\calc_elevations.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\cghs.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\cghs.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\cghs.obj"	"$(INTDIR)\cghs.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\char_vector.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\char_vector.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\char_vector.obj"	"$(INTDIR)\char_vector.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\chebyshev.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\chebyshev.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\chebyshev.obj"	"$(INTDIR)\chebyshev.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\comm_pkg.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\comm_pkg.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\comm_pkg.obj"	"$(INTDIR)\comm_pkg.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\communication.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\communication.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\communication.obj"	"$(INTDIR)\communication.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\computation.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\computation.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\computation.obj"	"$(INTDIR)\computation.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\compute_maximums.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\compute_maximums.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\compute_maximums.obj"	"$(INTDIR)\compute_maximums.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\compute_total_concentration.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\compute_total_concentration.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\compute_total_concentration.obj"	"$(INTDIR)\compute_total_concentration.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\constant_porosity.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\constant_porosity.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\constant_porosity.obj"	"$(INTDIR)\constant_porosity.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\constantRF.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\constantRF.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\constantRF.obj"	"$(INTDIR)\constantRF.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\copy.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\copy.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\copy.obj"	"$(INTDIR)\copy.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\create_grid.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\create_grid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\create_grid.obj"	"$(INTDIR)\create_grid.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\diag_scale.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\diag_scale.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\diag_scale.obj"	"$(INTDIR)\diag_scale.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\diffuse_lb.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\diffuse_lb.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\diffuse_lb.obj"	"$(INTDIR)\diffuse_lb.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\discretize_pressure.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\discretize_pressure.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\discretize_pressure.obj"	"$(INTDIR)\discretize_pressure.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\distribute_usergrid.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\distribute_usergrid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\distribute_usergrid.obj"	"$(INTDIR)\distribute_usergrid.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\dpofa.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\dpofa.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\dpofa.obj"	"$(INTDIR)\dpofa.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\dposl.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\dposl.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\dposl.obj"	"$(INTDIR)\dposl.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\drand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\drand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\drand48.obj"	"$(INTDIR)\drand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\erand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\erand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\erand48.obj"	"$(INTDIR)\erand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\gauinv.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\gauinv.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\gauinv.obj"	"$(INTDIR)\gauinv.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\general.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\general.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\general.obj"	"$(INTDIR)\general.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\geom_t_solid.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\geom_t_solid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\geom_t_solid.obj"	"$(INTDIR)\geom_t_solid.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\geometry.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\geometry.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\geometry.obj"	"$(INTDIR)\geometry.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\globals.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\globals.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\globals.obj"	"$(INTDIR)\globals.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\grgeom_list.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\grgeom_list.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\grgeom_list.obj"	"$(INTDIR)\grgeom_list.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\grgeom_octree.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\grgeom_octree.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\grgeom_octree.obj"	"$(INTDIR)\grgeom_octree.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\grgeometry.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\grgeometry.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\grgeometry.obj"	"$(INTDIR)\grgeometry.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\grid.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\grid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\grid.obj"	"$(INTDIR)\grid.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\hbt.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\hbt.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\hbt.obj"	"$(INTDIR)\hbt.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\Header.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\Header.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\Header.obj"	"$(INTDIR)\Header.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\infinity_norm.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\infinity_norm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\infinity_norm.obj"	"$(INTDIR)\infinity_norm.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\innerprod.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\innerprod.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\innerprod.obj"	"$(INTDIR)\innerprod.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\input_database.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\input_database.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\input_database.obj"	"$(INTDIR)\input_database.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\inputRF.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\inputRF.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\inputRF.obj"	"$(INTDIR)\inputRF.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\kinsol\iterativ.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\iterativ.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\iterativ.obj"	"$(INTDIR)\iterativ.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\jrand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\jrand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\jrand48.obj"	"$(INTDIR)\jrand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\kinsol\kinsol.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\kinsol.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\kinsol.obj"	"$(INTDIR)\kinsol.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\kinsol_nonlin_solver.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\kinsol_nonlin_solver.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\kinsol_nonlin_solver.obj"	"$(INTDIR)\kinsol_nonlin_solver.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\kinsol_pc.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\kinsol_pc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\kinsol_pc.obj"	"$(INTDIR)\kinsol_pc.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\kinsol\kinspgmr.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\kinspgmr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\kinspgmr.obj"	"$(INTDIR)\kinspgmr.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\l2_error_norm.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\l2_error_norm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\l2_error_norm.obj"	"$(INTDIR)\l2_error_norm.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\line_process.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\line_process.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\line_process.obj"	"$(INTDIR)\line_process.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\kinsol\llnlmath.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\llnlmath.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\llnlmath.obj"	"$(INTDIR)\llnlmath.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\logging.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\logging.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\logging.obj"	"$(INTDIR)\logging.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\lrand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\lrand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\lrand48.obj"	"$(INTDIR)\lrand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\matdiag_scale.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\matdiag_scale.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\matdiag_scale.obj"	"$(INTDIR)\matdiag_scale.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\matrix.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\matrix.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\matrix.obj"	"$(INTDIR)\matrix.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\matvec.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\matvec.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\matvec.obj"	"$(INTDIR)\matvec.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\max_field_value.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\max_field_value.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\max_field_value.obj"	"$(INTDIR)\max_field_value.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\mg_semi.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\mg_semi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\mg_semi.obj"	"$(INTDIR)\mg_semi.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\mg_semi_prolong.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\mg_semi_prolong.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\mg_semi_prolong.obj"	"$(INTDIR)\mg_semi_prolong.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\mg_semi_restrict.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\mg_semi_restrict.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\mg_semi_restrict.obj"	"$(INTDIR)\mg_semi_restrict.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\mrand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\mrand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\mrand48.obj"	"$(INTDIR)\mrand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\n_vector.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\n_vector.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\n_vector.obj"	"$(INTDIR)\n_vector.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\new_endpts.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\new_endpts.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\new_endpts.obj"	"$(INTDIR)\new_endpts.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\nl_function_eval.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\nl_function_eval.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\nl_function_eval.obj"	"$(INTDIR)\nl_function_eval.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\nodiag_scale.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\nodiag_scale.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\nodiag_scale.obj"	"$(INTDIR)\nodiag_scale.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\nrand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\nrand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\nrand48.obj"	"$(INTDIR)\nrand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\parflow.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\parflow.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\parflow.obj"	"$(INTDIR)\parflow.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\pcg.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\pcg.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\pcg.obj"	"$(INTDIR)\pcg.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\permeability_face.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\permeability_face.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\permeability_face.obj"	"$(INTDIR)\permeability_face.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\perturb_lb.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\perturb_lb.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\perturb_lb.obj"	"$(INTDIR)\perturb_lb.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\pf_module.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\pf_module.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\pf_module.obj"	"$(INTDIR)\pf_module.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\pfield.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\pfield.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\pfield.obj"	"$(INTDIR)\pfield.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\pgsRF.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\pgsRF.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\pgsRF.obj"	"$(INTDIR)\pgsRF.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\phase_velocity_face.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\phase_velocity_face.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\phase_velocity_face.obj"	"$(INTDIR)\phase_velocity_face.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\ppcg.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\ppcg.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\ppcg.obj"	"$(INTDIR)\ppcg.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\printgrid.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\printgrid.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\printgrid.obj"	"$(INTDIR)\printgrid.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\printmatrix.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\printmatrix.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\printmatrix.obj"	"$(INTDIR)\printmatrix.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\printvector.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\printvector.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\printvector.obj"	"$(INTDIR)\printvector.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem.obj"	"$(INTDIR)\problem.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_bc.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_bc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_bc.obj"	"$(INTDIR)\problem_bc.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_bc_internal.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_bc_internal.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_bc_internal.obj"	"$(INTDIR)\problem_bc_internal.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_bc_phase_saturation.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_bc_phase_saturation.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_bc_phase_saturation.obj"	"$(INTDIR)\problem_bc_phase_saturation.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_bc_pressure.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_bc_pressure.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_bc_pressure.obj"	"$(INTDIR)\problem_bc_pressure.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_capillary_pressure.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_capillary_pressure.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_capillary_pressure.obj"	"$(INTDIR)\problem_capillary_pressure.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_domain.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_domain.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_domain.obj"	"$(INTDIR)\problem_domain.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_eval.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_eval.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_eval.obj"	"$(INTDIR)\problem_eval.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_geometries.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_geometries.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_geometries.obj"	"$(INTDIR)\problem_geometries.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_ic_phase_concen.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_ic_phase_concen.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_ic_phase_concen.obj"	"$(INTDIR)\problem_ic_phase_concen.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_ic_phase_pressure.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_ic_phase_pressure.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_ic_phase_pressure.obj"	"$(INTDIR)\problem_ic_phase_pressure.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_ic_phase_satur.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_ic_phase_satur.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_ic_phase_satur.obj"	"$(INTDIR)\problem_ic_phase_satur.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_phase_density.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_phase_density.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_phase_density.obj"	"$(INTDIR)\problem_phase_density.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_phase_mobility.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_phase_mobility.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_phase_mobility.obj"	"$(INTDIR)\problem_phase_mobility.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_phase_rel_perm.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_phase_rel_perm.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_phase_rel_perm.obj"	"$(INTDIR)\problem_phase_rel_perm.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_phase_source.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_phase_source.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_phase_source.obj"	"$(INTDIR)\problem_phase_source.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_porosity.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_porosity.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_porosity.obj"	"$(INTDIR)\problem_porosity.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_retardation.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_retardation.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_retardation.obj"	"$(INTDIR)\problem_retardation.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_richards_bc_internal.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_richards_bc_internal.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_richards_bc_internal.obj"	"$(INTDIR)\problem_richards_bc_internal.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_saturation.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_saturation.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_saturation.obj"	"$(INTDIR)\problem_saturation.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\problem_saturation_constitutive.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\problem_saturation_constitutive.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\problem_saturation_constitutive.obj"	"$(INTDIR)\problem_saturation_constitutive.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\random.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\random.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\random.obj"	"$(INTDIR)\random.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\ratqr.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\ratqr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\ratqr.obj"	"$(INTDIR)\ratqr.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\rb_GS_point.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\rb_GS_point.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\rb_GS_point.obj"	"$(INTDIR)\rb_GS_point.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\read_parflow_binary.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\read_parflow_binary.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\read_parflow_binary.obj"	"$(INTDIR)\read_parflow_binary.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\reg_from_stenc.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\reg_from_stenc.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\reg_from_stenc.obj"	"$(INTDIR)\reg_from_stenc.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\region.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\region.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\region.obj"	"$(INTDIR)\region.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\richards_jacobian_eval.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\richards_jacobian_eval.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\richards_jacobian_eval.obj"	"$(INTDIR)\richards_jacobian_eval.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\sadvect.win32.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\sadvect.win32.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\sadvect.win32.obj"	"$(INTDIR)\sadvect.win32.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\sadvection_godunov.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\sadvection_godunov.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\sadvection_godunov.obj"	"$(INTDIR)\sadvection_godunov.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\scale.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\scale.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\scale.obj"	"$(INTDIR)\scale.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\select_time_step.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\select_time_step.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\select_time_step.obj"	"$(INTDIR)\select_time_step.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\set_problem_data.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\set_problem_data.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\set_problem_data.obj"	"$(INTDIR)\set_problem_data.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\common\signal.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\signal.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\signal.obj"	"$(INTDIR)\signal.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\sim_shear.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\sim_shear.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\sim_shear.obj"	"$(INTDIR)\sim_shear.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\solver.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\solver.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\solver.obj"	"$(INTDIR)\solver.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\solver_impes.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\solver_impes.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\solver_impes.obj"	"$(INTDIR)\solver_impes.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\solver_lb.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\solver_lb.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\solver_lb.obj"	"$(INTDIR)\solver_lb.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\solver_richards.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\solver_richards.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\solver_richards.obj"	"$(INTDIR)\solver_richards.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\kinsol\spgmr.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\spgmr.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\spgmr.obj"	"$(INTDIR)\spgmr.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\srand48.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\srand48.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\srand48.obj"	"$(INTDIR)\srand48.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\subsrf_sim.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\subsrf_sim.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\subsrf_sim.obj"	"$(INTDIR)\subsrf_sim.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\time_cycle_data.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\time_cycle_data.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\time_cycle_data.obj"	"$(INTDIR)\time_cycle_data.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\timing.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\timing.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\timing.obj"	"$(INTDIR)\timing.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\total_velocity_face.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\total_velocity_face.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\total_velocity_face.obj"	"$(INTDIR)\total_velocity_face.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\turning_bandsRF.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\turning_bandsRF.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\turning_bandsRF.obj"	"$(INTDIR)\turning_bandsRF.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\amps\win32\unix_port.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\unix_port.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\unix_port.obj"	"$(INTDIR)\unix_port.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\usergrid_input.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\usergrid_input.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\usergrid_input.obj"	"$(INTDIR)\usergrid_input.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\vector.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\vector.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\vector.obj"	"$(INTDIR)\vector.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\vector_utilities.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\vector_utilities.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\vector_utilities.obj"	"$(INTDIR)\vector_utilities.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\w_jacobi.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\w_jacobi.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\w_jacobi.obj"	"$(INTDIR)\w_jacobi.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\well.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\well.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\well.obj"	"$(INTDIR)\well.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\well_package.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\well_package.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\well_package.obj"	"$(INTDIR)\well_package.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\wells_lb.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\wells_lb.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\wells_lb.obj"	"$(INTDIR)\wells_lb.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\parflow\write_parflow_binary.c

!IF  "$(CFG)" == "ParFlow - Win32 Release"


"$(INTDIR)\write_parflow_binary.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"


"$(INTDIR)\write_parflow_binary.obj"	"$(INTDIR)\write_parflow_binary.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 


!ENDIF 

