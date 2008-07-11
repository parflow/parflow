# Microsoft Developer Studio Project File - Name="ParFlow" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=ParFlow - Win32 Insure
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "ParFlow.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "ParFlow.mak" CFG="ParFlow - Win32 Insure"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "ParFlow - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "ParFlow - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE "ParFlow - Win32 Insure" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "ParFlow - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
F90=df.exe
# ADD BASE F90 /include:"Release/" /compile_only /nologo
# ADD F90 /include:"Release/" /compile_only /nologo /iface:cref
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /Ob2 /I "..\..\amps\win32" /I "..\..\config" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386

!ELSEIF  "$(CFG)" == "ParFlow - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
F90=df.exe
# ADD BASE F90 /include:"Debug/" /compile_only /nologo /debug:full /optimize:0
# ADD F90 /include:"Debug/" /compile_only /nologo /debug:full /optimize:0 /iface:cref
# ADD BASE CPP /nologo /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /Gm /GX /ZI /Od /I "..\..\amps\win32" /I "..\..\config" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ELSEIF  "$(CFG)" == "ParFlow - Win32 Insure"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "ParFlow___Win32_Insure"
# PROP BASE Intermediate_Dir "ParFlow___Win32_Insure"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Insure"
# PROP Intermediate_Dir "Insure"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
F90=df.exe
# ADD BASE F90 /include:"Debug/" /compile_only /nologo /debug:full /optimize:0 /iface:cref
# ADD F90 /include:"Debug/" /compile_only /nologo /debug:full /optimize:0 /iface:cref
# ADD BASE CPP /nologo /MD /W3 /Gm /GX /ZI /Od /I "..\..\amps\win32" /I "..\..\config" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /FR /YX /FD /c
# ADD CPP /nologo /MD /W3 /Gm /GX /ZI /Od /I "..\..\amps\win32" /I "..\..\config" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D "PF_TIMING" /FR /YX /FD /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept

!ENDIF 

# Begin Target

# Name "ParFlow - Win32 Release"
# Name "ParFlow - Win32 Debug"
# Name "ParFlow - Win32 Insure"
# Begin Source File

SOURCE=..\..\amps\win32\_rand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\advect.win32.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\advection_godunov.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_allreduce.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_bcast.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_clear.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_createinvoice.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_exchange.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_ffopen.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_finalize.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_find_powers.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_fopen.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_init.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_invoice.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_io.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_irecv.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_newhandle.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_newpackage.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_pack.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_recv.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_send.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_sfbcast.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_sfclose.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\amps_sfopen.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_sizeofinvoice.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_sync.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_test.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_unpack.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_vector.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\amps_wait.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\axpy.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\background.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\bc_lb.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\bc_pressure.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\bc_pressure_package.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\calc_elevations.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\cghs.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\char_vector.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\chebyshev.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\comm_pkg.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\communication.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\computation.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\compute_maximums.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\compute_total_concentration.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\constant_porosity.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\constantRF.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\copy.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\create_grid.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\diag_scale.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\diffuse_lb.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\discretize_pressure.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\distribute_usergrid.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\dpofa.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\dposl.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\drand48.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\erand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\gauinv.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\general.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\geom_t_solid.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\geometry.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\globals.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\grgeom_list.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\grgeom_octree.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\grgeometry.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\grid.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\hbt.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\Header.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\infinity_norm.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\innerprod.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\input_database.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\inputRF.c
# End Source File
# Begin Source File

SOURCE=..\..\kinsol\iterativ.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\jrand48.c
# End Source File
# Begin Source File

SOURCE=..\..\kinsol\kinsol.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\kinsol_nonlin_solver.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\kinsol_pc.c
# End Source File
# Begin Source File

SOURCE=..\..\kinsol\kinspgmr.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\l2_error_norm.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\line_process.c
# End Source File
# Begin Source File

SOURCE=..\..\kinsol\llnlmath.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\logging.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\lrand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\matdiag_scale.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\matrix.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\matvec.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\max_field_value.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\mg_semi.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\mg_semi_prolong.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\mg_semi_restrict.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\mrand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\n_vector.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\new_endpts.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\nl_function_eval.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\nodiag_scale.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\nrand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\parflow.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\pcg.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\permeability_face.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\perturb_lb.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\pf_module.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\pfield.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\pgsRF.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\phase_velocity_face.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\ppcg.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\printgrid.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\printmatrix.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\printvector.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_bc.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_bc_internal.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_bc_phase_saturation.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_bc_pressure.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_capillary_pressure.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_domain.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_eval.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_geometries.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_ic_phase_concen.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_ic_phase_pressure.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_ic_phase_satur.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_phase_density.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_phase_mobility.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_phase_rel_perm.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_phase_source.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_porosity.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_retardation.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_richards_bc_internal.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_saturation.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\problem_saturation_constitutive.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\random.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\ratqr.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\rb_GS_point.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\read_parflow_binary.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\reg_from_stenc.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\region.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\richards_jacobian_eval.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\sadvect.win32.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\sadvection_godunov.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\scale.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\select_time_step.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\set_problem_data.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\common\signal.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\sim_shear.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\solver.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\solver_impes.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\solver_lb.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\solver_richards.c
# End Source File
# Begin Source File

SOURCE=..\..\kinsol\spgmr.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\srand48.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\subsrf_sim.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\time_cycle_data.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\timing.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\total_velocity_face.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\turning_bandsRF.c
# End Source File
# Begin Source File

SOURCE=..\..\amps\win32\unix_port.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\usergrid_input.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\vector.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\vector_utilities.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\w_jacobi.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\well.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\well_package.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\wells_lb.c
# End Source File
# Begin Source File

SOURCE=..\..\parflow\write_parflow_binary.c
# End Source File
# End Target
# End Project
