#include <define.h>

subroutine CoLMINI(idate, numpatch)

! ======================================================================
! Initialization of Land Characteristic Parameters and Initial State Variables
!
! Reference:
!     [1] Dai et al., 2003: The Common Land Model (CoLM).
!         Bull. of Amer. Meter. Soc., 84: 1013-1023
!     [2] Dai et al., 2004: A two-big-leaf model for canopy temperature,
!         photosynthesis and stomatal conductance. Journal of Climate
!     [3] Dai et al., 2014: The Terrestrial Modeling System (TMS).
!
!     Created by Yongjiu Dai Februay 2004
!     Revised by Yongjiu Dai Februay 2014
! ======================================================================

   USE MOD_Precision
   USE MOD_Namelist
   USE MOD_SPMD_Task
   !USE MOD_Block
   !USE MOD_Pixel
   !USE MOD_Mesh
   !USE MOD_LandElm
#ifdef CATCHMENT
   USE MOD_LandHRU
#endif
   !USE MOD_LandPatch
   !USE MOD_SrfdataRestart
   USE MOD_Vars_Global
   USE MOD_Const_LC
   USE MOD_Const_PFT
   USE MOD_TimeManager
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_LandPFT
#endif
#ifdef URBAN_MODEL
   USE MOD_LandUrban
#endif
#ifdef SinglePoint
   USE MOD_SingleSrfdata
#endif
#if (defined UNSTRUCTURED || defined CATCHMENT)
   USE MOD_ElmVector
#endif
#ifdef CATCHMENT
   USE MOD_HRUVector
#endif
   USE MOD_Initialize
   ! SNICAR
   !USE MOD_SnowSnicar, only: SnowAge_init, SnowOptics_init
   IMPLICIT NONE

   ! ----------------local variables ---------------------------------
   character(len=256) :: nlfile
   character(len=256) :: casename ! case name
   character(len=256) :: dir_landdata
   character(len=256) :: dir_restart
   character(len=256) :: fsrfdata
   integer  :: s_year      ! starting date for run in year
   integer  :: s_month     ! starting date for run in month
   integer  :: s_day       ! starting date for run in day
   integer  :: s_julian    ! starting date for run in julian day
   integer  :: s_seconds   ! starting time of day for run in seconds
   integer, intent(in) :: idate(3)    ! starting date
   integer  :: lc_year     ! land cover year
   logical  :: greenwich   ! true: greenwich time, false: local time

   integer*8 :: start_time, end_time, c_per_sec, time_used
   integer, intent(in) :: numpatch

#ifdef USEMPI
      CALL spmd_init ()
#endif

      !IF (p_is_master) THEN
      !   CALL system_clock (start_time)
      !ENDIF

      ! ----------------------------------------------------------------------
      !CALL getarg (1, nlfile)
      !CALL read_namelist (nlfile)

      !casename     = DEF_CASE_NAME
      !dir_landdata = DEF_dir_landdata
      !dir_restart  = DEF_dir_restart
      greenwich    = DEF_simulation_time%greenwich
      !s_year       = DEF_simulation_time%start_year
      !s_month      = DEF_simulation_time%start_month
      !s_day        = DEF_simulation_time%start_day
      !s_seconds    = DEF_simulation_time%start_sec

#ifdef SinglePoint
      fsrfdata = trim(dir_landdata) // '/srfdata.nc'
#ifndef URBAN_MODEL
      !CALL read_surface_data_single (fsrfdata, mksrfdata=.false.)
#else
      !CALL read_urban_surface_data_single (fsrfdata, mksrfdata=.false., mkrun=.true.)
#endif
#endif

      !CALL monthday2julian(s_year,s_month,s_day,s_julian)
      !idate(1) = s_year; idate(2) = s_julian; idate(3) = s_seconds
      !CALL adj2begin(idate)

#ifdef LULCC
      lc_year = idate(1)
#else
      lc_year = DEF_LC_YEAR
#endif

      !CALL Init_GlobalVars
      !CALL Init_LC_Const
      !CALL Init_PFT_Const

      !CALL pixel%load_from_file  (dir_landdata)
      !CALL gblock%load_from_file (dir_landdata)
      !CALL mesh_load_from_file   (dir_landdata, lc_year)

      !CALL pixelset_load_from_file (dir_landdata, 'landelm', landelm, numelm, lc_year)

#ifdef CATCHMENT
      CALL pixelset_load_from_file (dir_landdata, 'landhru', landhru, numhru, lc_year)
#endif

      !CALL pixelset_load_from_file (dir_landdata, 'landpatch', landpatch, numpatch, lc_year)

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL pixelset_load_from_file (dir_landdata, 'landpft', landpft, numpft, lc_year)
      CALL map_patch_to_pft
#endif

#ifdef URBAN_MODEL
      CALL pixelset_load_from_file (dir_landdata, 'landurban', landurban, numurban, lc_year)
      CALL map_patch_to_urban
#endif
#if (defined UNSTRUCTURED || defined CATCHMENT)
      CALL elm_vector_init ()
#ifdef CATCHMENT
      CALL hru_vector_init ()
#endif
#endif

      !! Read in SNICAR optical and aging parameters
      !CALL SnowOptics_init( DEF_file_snowoptics ) ! SNICAR optical parameters
      !CALL SnowAge_init( DEF_file_snowaging )     ! SNICAR aging   parameters

      CALL initialize (casename, dir_landdata, dir_restart, idate, lc_year, greenwich, numpatch)

#ifdef SinglePoint
      CALL single_srfdata_final ()
#endif

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

      !IF (p_is_master) THEN
      !   CALL system_clock (end_time, count_rate = c_per_sec)
      !   time_used = (end_time - start_time) / c_per_sec
      !   IF (time_used >= 3600) THEN
      !      write(*,101) time_used/3600, mod(time_used,3600)/60, mod(time_used,60)
      !      101 format (/,'Overall system time used:', I4, ' hours', I3, ' minutes', I3, ' seconds.')
      !   ELSEIF (time_used >= 60) THEN
      !      write(*,102) time_used/60, mod(time_used,60)
      !      102 format (/,'Overall system time used:', I3, ' minutes', I3, ' seconds.')
      !   ELSE
      !      write(*,103) time_used
      !      103 format (/,'Overall system time used:', I3, ' seconds.')
      !   ENDIF

      !   write(*,*) 'CoLM Initialization Execution Completed'
      !ENDIF

#ifdef USEMPI
      CALL spmd_exit
#endif

END subroutine CoLMINI
! ----------------------------------------------------------------------
! EOP
