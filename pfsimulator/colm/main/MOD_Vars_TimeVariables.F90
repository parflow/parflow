#include <define.h>

! -------------------------------
! Created by Yongjiu Dai, 03/2014
! -------------------------------
MODULE MOD_Vars_PFTimeVariables
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)

! -----------------------------------------------------------------
! !DESCRIPTION:
! Define PFT time variables
!
! Added by Hua Yuan, 08/2019
! -----------------------------------------------------------------

   USE MOD_Precision
   USE MOD_TimeManager
#ifdef BGC
   USE MOD_BGC_Vars_PFTimeVariables
#endif

   IMPLICIT NONE
   SAVE
! -----------------------------------------------------------------
! Time-varying state variables which reaquired by restart run

   ! for LULC_IGBP_PFT or LULC_IGBP_PC
   real(r8), allocatable :: tleaf_p      (:) !shaded leaf temperature [K]
   real(r8), allocatable :: ldew_p       (:) !depth of water on foliage [mm]
   real(r8), allocatable :: ldew_rain_p  (:) !depth of rain on foliage [mm]
   real(r8), allocatable :: ldew_snow_p  (:) !depth of snow on foliage [mm]
   real(r8), allocatable :: sigf_p       (:) !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), allocatable :: tlai_p       (:) !leaf area index
   real(r8), allocatable :: lai_p        (:) !leaf area index
   real(r8), allocatable :: laisun_p     (:) !sunlit leaf area index
   real(r8), allocatable :: laisha_p     (:) !shaded leaf area index
   real(r8), allocatable :: tsai_p       (:) !stem area index
   real(r8), allocatable :: sai_p        (:) !stem area index
   real(r8), allocatable :: ssun_p   (:,:,:) !sunlit canopy absorption for solar radiation (0-1)
   real(r8), allocatable :: ssha_p   (:,:,:) !shaded canopy absorption for solar radiation (0-1)
   real(r8), allocatable :: thermk_p     (:) !canopy gap fraction for tir radiation
   real(r8), allocatable :: fshade_p     (:) !canopy shade fraction for tir radiation
   real(r8), allocatable :: extkb_p      (:) !(k, g(mu)/mu) direct solar extinction coefficient
   real(r8), allocatable :: extkd_p      (:) !diffuse and scattered diffuse PAR extinction coefficient
   !TODO@yuan: to check the below for PC whether they are needed
   real(r8), allocatable :: tref_p       (:) !2 m height air temperature [kelvin]
   real(r8), allocatable :: qref_p       (:) !2 m height air specific humidity
   real(r8), allocatable :: rst_p        (:) !canopy stomatal resistance (s/m)
   real(r8), allocatable :: z0m_p        (:) !effective roughness [m]
! Plant Hydraulic variables
   real(r8), allocatable :: vegwp_p    (:,:) !vegetation water potential [mm]
   real(r8), allocatable :: gs0sun_p     (:) !working copy of sunlit stomata conductance
   real(r8), allocatable :: gs0sha_p     (:) !working copy of shalit stomata conductance
! END plant hydraulic variables
! Ozone Stress Variables
   real(r8), allocatable :: o3coefv_sun_p(:) !Ozone stress factor for photosynthesis on sunlit leaf
   real(r8), allocatable :: o3coefv_sha_p(:) !Ozone stress factor for photosynthesis on shaded leaf
   real(r8), allocatable :: o3coefg_sun_p(:) !Ozone stress factor for stomata on sunlit leaf
   real(r8), allocatable :: o3coefg_sha_p(:) !Ozone stress factor for stomata on shaded leaf
   real(r8), allocatable :: lai_old_p    (:) !lai in last time step
   real(r8), allocatable :: o3uptakesun_p(:) !Ozone does, sunlit leaf (mmol O3/m^2)
   real(r8), allocatable :: o3uptakesha_p(:) !Ozone does, shaded leaf (mmol O3/m^2)
! END Ozone Stress Variables
! irrigation variables
   integer , allocatable :: irrig_method_p(:)!irrigation method
! END irrigation variables

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: allocate_PFTimeVariables
   PUBLIC :: deallocate_PFTimeVariables
   PUBLIC :: READ_PFTimeVariables
   PUBLIC :: WRITE_PFTimeVariables
#ifdef RangeCheck
   PUBLIC :: check_PFTimeVariables
#endif

! PRIVATE MEMBER FUNCTIONS:

!-----------------------------------------------------------------------

CONTAINS

!-----------------------------------------------------------------------

   SUBROUTINE allocate_PFTimeVariables ()
! ------------------------------------------------------
! Allocates memory for CoLM 1d [numpft] variables
! ------------------------------------------------------
   USE MOD_Precision
   USE MOD_SPMD_Task
   USE MOD_LandPFT
   USE MOD_Vars_Global
   IMPLICIT NONE

      IF (p_is_worker) THEN
         IF (numpft > 0) THEN
            allocate (tleaf_p      (numpft)) ; tleaf_p      (:) = spval !leaf temperature [K]
            allocate (ldew_p       (numpft)) ; ldew_p       (:) = spval !depth of water on foliage [mm]
            allocate (ldew_rain_p  (numpft)) ; ldew_rain_p  (:) = spval !depth of rain on foliage [mm]
            allocate (ldew_snow_p  (numpft)) ; ldew_snow_p  (:) = spval !depth of snow on foliage [mm]
            allocate (sigf_p       (numpft)) ; sigf_p       (:) = spval !fraction of veg cover, excluding snow-covered veg [-]
            allocate (tlai_p       (numpft)) ; tlai_p       (:) = spval !leaf area index
            allocate (lai_p        (numpft)) ; lai_p        (:) = spval !leaf area index
            allocate (laisun_p     (numpft)) ; laisun_p     (:) = spval !leaf area index
            allocate (laisha_p     (numpft)) ; laisha_p     (:) = spval !leaf area index
            allocate (tsai_p       (numpft)) ; tsai_p       (:) = spval !stem area index
            allocate (sai_p        (numpft)) ; sai_p        (:) = spval !stem area index
            allocate (ssun_p   (2,2,numpft)) ; ssun_p   (:,:,:) = spval !sunlit canopy absorption for solar radiation (0-1)
            allocate (ssha_p   (2,2,numpft)) ; ssha_p   (:,:,:) = spval !shaded canopy absorption for solar radiation (0-1)
            allocate (thermk_p     (numpft)) ; thermk_p     (:) = spval !canopy gap fraction for tir radiation
            allocate (fshade_p     (numpft)) ; fshade_p     (:) = spval !canopy shade fraction for tir radiation
            allocate (extkb_p      (numpft)) ; extkb_p      (:) = spval !(k, g(mu)/mu) direct solar extinction coefficient
            allocate (extkd_p      (numpft)) ; extkd_p      (:) = spval !diffuse and scattered diffuse PAR extinction coefficient
            allocate (tref_p       (numpft)) ; tref_p       (:) = spval !2 m height air temperature [kelvin]
            allocate (qref_p       (numpft)) ; qref_p       (:) = spval !2 m height air specific humidity
            allocate (rst_p        (numpft)) ; rst_p        (:) = spval !canopy stomatal resistance (s/m)
            allocate (z0m_p        (numpft)) ; z0m_p        (:) = spval !effective roughness [m]
! Plant Hydraulic variables; draulic variables
            allocate (vegwp_p(1:nvegwcs,numpft)); vegwp_p (:,:) = spval
            allocate (gs0sun_p     (numpft)); gs0sun_p      (:) = spval
            allocate (gs0sha_p     (numpft)); gs0sha_p      (:) = spval
! END plant hydraulic variables
! Allocate Ozone Stress Variables
            allocate (o3coefv_sun_p(numpft)) ; o3coefv_sun_p(:) = spval !Ozone stress factor for photosynthesis on sunlit leaf
            allocate (o3coefv_sha_p(numpft)) ; o3coefv_sha_p(:) = spval !Ozone stress factor for photosynthesis on shaded leaf
            allocate (o3coefg_sun_p(numpft)) ; o3coefg_sun_p(:) = spval !Ozone stress factor for stomata on sunlit leaf
            allocate (o3coefg_sha_p(numpft)) ; o3coefg_sha_p(:) = spval !Ozone stress factor for stomata on shaded leaf
            allocate (lai_old_p    (numpft)) ; lai_old_p    (:) = spval !lai in last time step
            allocate (o3uptakesun_p(numpft)) ; o3uptakesun_p(:) = spval !Ozone does, sunlit leaf (mmol O3/m^2)
            allocate (o3uptakesha_p(numpft)) ; o3uptakesha_p(:) = spval !Ozone does, shaded leaf (mmol O3/m^2)
! END allocate Ozone Stress Variables
            allocate (irrig_method_p(numpft))! irrigation method

         ENDIF
      ENDIF

#ifdef BGC
      CALL allocate_BGCPFTimeVariables
#endif

   END SUBROUTINE allocate_PFTimeVariables

   SUBROUTINE READ_PFTimeVariables (file_restart)

   USE MOD_Namelist, only: DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, DEF_USE_IRRIGATION
   USE MOD_NetCDFVector
   USE MOD_LandPFT
   USE MOD_Vars_Global

   IMPLICIT NONE

      character(len=*), intent(in) :: file_restart

      CALL ncio_read_vector (file_restart, 'tleaf_p  ', landpft, tleaf_p    ) !
      CALL ncio_read_vector (file_restart, 'ldew_p   ', landpft, ldew_p     ) !
      CALL ncio_read_vector (file_restart, 'ldew_rain_p', landpft, ldew_rain_p) !depth of rain on foliage [mm]
      CALL ncio_read_vector (file_restart, 'ldew_snow_p', landpft, ldew_snow_p) !depth of snow on foliage [mm]
      CALL ncio_read_vector (file_restart, 'sigf_p   ', landpft, sigf_p     ) !
      CALL ncio_read_vector (file_restart, 'tlai_p   ', landpft, tlai_p     ) !
      CALL ncio_read_vector (file_restart, 'lai_p    ', landpft, lai_p      ) !
!     CALL ncio_read_vector (file_restart, 'laisun_p ', landpft, laisun_p   ) !
!     CALL ncio_read_vector (file_restart, 'laisha_p ', landpft, laisha_p   ) !
      CALL ncio_read_vector (file_restart, 'tsai_p   ', landpft, tsai_p     ) !
      CALL ncio_read_vector (file_restart, 'sai_p    ', landpft, sai_p      ) !
      CALL ncio_read_vector (file_restart, 'ssun_p   ', 2,2, landpft, ssun_p) !
      CALL ncio_read_vector (file_restart, 'ssha_p   ', 2,2, landpft, ssha_p) !
      CALL ncio_read_vector (file_restart, 'thermk_p ', landpft, thermk_p   ) !
      CALL ncio_read_vector (file_restart, 'fshade_p ', landpft, fshade_p   ) !
      CALL ncio_read_vector (file_restart, 'extkb_p  ', landpft, extkb_p    ) !
      CALL ncio_read_vector (file_restart, 'extkd_p  ', landpft, extkd_p    ) !
      CALL ncio_read_vector (file_restart, 'tref_p   ', landpft, tref_p     ) !
      CALL ncio_read_vector (file_restart, 'qref_p   ', landpft, qref_p     ) !
      CALL ncio_read_vector (file_restart, 'rst_p    ', landpft, rst_p      ) !
      CALL ncio_read_vector (file_restart, 'z0m_p    ', landpft, z0m_p      ) !
IF(DEF_USE_PLANTHYDRAULICS)THEN
      CALL ncio_read_vector (file_restart, 'vegwp_p  ', nvegwcs, landpft, vegwp_p ) !
      CALL ncio_read_vector (file_restart, 'gs0sun_p ', landpft, gs0sun_p   ) !
      CALL ncio_read_vector (file_restart, 'gs0sha_p ', landpft, gs0sha_p   ) !
ENDIF
IF(DEF_USE_OZONESTRESS)THEN
      CALL ncio_read_vector (file_restart, 'lai_old_p    ', landpft, lai_old_p    , defval = 0._r8)
      CALL ncio_read_vector (file_restart, 'o3uptakesun_p', landpft, o3uptakesun_p, defval = 0._r8)
      CALL ncio_read_vector (file_restart, 'o3uptakesha_p', landpft, o3uptakesha_p, defval = 0._r8)
ENDIF
IF(DEF_USE_IRRIGATION)THEN
      CALL ncio_read_vector (file_restart,'irrig_method_p', landpft,irrig_method_p, defval = 1)
ENDIF

#ifdef BGC
      CALL read_BGCPFTimeVariables (file_restart)
#endif

   END SUBROUTINE READ_PFTimeVariables

   SUBROUTINE WRITE_PFTimeVariables (file_restart)

   USE MOD_Namelist, only : DEF_REST_CompressLevel, DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, &
                            DEF_USE_IRRIGATION
   USE MOD_LandPFT
   USE MOD_NetCDFVector
   USE MOD_Vars_Global
   IMPLICIT NONE

   character(len=*), intent(in) :: file_restart

   ! Local variables
   integer :: compress

      compress = DEF_REST_CompressLevel

      CALL ncio_create_file_vector (file_restart, landpft)
      CALL ncio_define_dimension_vector (file_restart, landpft, 'pft')
      CALL ncio_define_dimension_vector (file_restart, landpft, 'band', 2)
      CALL ncio_define_dimension_vector (file_restart, landpft, 'rtyp', 2)
IF(DEF_USE_PLANTHYDRAULICS)THEN
      CALL ncio_define_dimension_vector (file_restart, landpft, 'vegnodes', nvegwcs)
ENDIF

      CALL ncio_write_vector (file_restart, 'tleaf_p  ', 'pft', landpft, tleaf_p  , compress) !
      CALL ncio_write_vector (file_restart, 'ldew_p   ', 'pft', landpft, ldew_p   , compress) !
      CALL ncio_write_vector (file_restart, 'ldew_rain_p', 'pft', landpft, ldew_rain_p, compress) !depth of rain on foliage [mm]
      CALL ncio_write_vector (file_restart, 'ldew_snow_p', 'pft', landpft, ldew_snow_p, compress) !depth of snow on foliage [mm]
      CALL ncio_write_vector (file_restart, 'sigf_p   ', 'pft', landpft, sigf_p   , compress) !
      CALL ncio_write_vector (file_restart, 'tlai_p   ', 'pft', landpft, tlai_p   , compress) !
      CALL ncio_write_vector (file_restart, 'lai_p    ', 'pft', landpft, lai_p    , compress) !
!     CALL ncio_write_vector (file_restart, 'laisun_p ', 'pft', landpft, laisun_p , compress) !
!     CALL ncio_write_vector (file_restart, 'laisha_p ', 'pft', landpft, laisha_p , compress) !
      CALL ncio_write_vector (file_restart, 'tsai_p   ', 'pft', landpft, tsai_p   , compress) !
      CALL ncio_write_vector (file_restart, 'sai_p    ', 'pft', landpft, sai_p    , compress) !
      CALL ncio_write_vector (file_restart, 'ssun_p   ', 'band', 2, 'rtyp', 2, 'pft', landpft, ssun_p, compress) !
      CALL ncio_write_vector (file_restart, 'ssha_p   ', 'band', 2, 'rtyp', 2, 'pft', landpft, ssha_p, compress) !
      CALL ncio_write_vector (file_restart, 'thermk_p ', 'pft', landpft, thermk_p , compress) !
      CALL ncio_write_vector (file_restart, 'fshade_p ', 'pft', landpft, fshade_p , compress) !
      CALL ncio_write_vector (file_restart, 'extkb_p  ', 'pft', landpft, extkb_p  , compress) !
      CALL ncio_write_vector (file_restart, 'extkd_p  ', 'pft', landpft, extkd_p  , compress) !
      CALL ncio_write_vector (file_restart, 'tref_p   ', 'pft', landpft, tref_p   , compress) !
      CALL ncio_write_vector (file_restart, 'qref_p   ', 'pft', landpft, qref_p   , compress) !
      CALL ncio_write_vector (file_restart, 'rst_p    ', 'pft', landpft, rst_p    , compress) !
      CALL ncio_write_vector (file_restart, 'z0m_p    ', 'pft', landpft, z0m_p    , compress) !
IF(DEF_USE_PLANTHYDRAULICS)THEN
      CALL ncio_write_vector (file_restart, 'vegwp_p  '  , 'vegnodes', nvegwcs, 'pft',   landpft, vegwp_p, compress)
      CALL ncio_write_vector (file_restart, 'gs0sun_p '  , 'pft', landpft, gs0sun_p   , compress) !
      CALL ncio_write_vector (file_restart, 'gs0sha_p '  , 'pft', landpft, gs0sha_p   , compress) !
ENDIF
IF(DEF_USE_OZONESTRESS)THEN
      CALL ncio_write_vector (file_restart, 'lai_old_p    ', 'pft', landpft, lai_old_p    , compress)
      CALL ncio_write_vector (file_restart, 'o3uptakesun_p', 'pft', landpft, o3uptakesun_p, compress)
      CALL ncio_write_vector (file_restart, 'o3uptakesha_p', 'pft', landpft, o3uptakesha_p, compress)
ENDIF
IF(DEF_USE_IRRIGATION)THEN
      CALL ncio_write_vector (file_restart,'irrig_method_p','pft', landpft, irrig_method_p, compress)
ENDIF

#ifdef BGC
      CALL WRITE_BGCPFTimeVariables (file_restart)
#endif

   END SUBROUTINE WRITE_PFTimeVariables


   SUBROUTINE deallocate_PFTimeVariables
! --------------------------------------------------
! Deallocates memory for CoLM 1d [numpft/numpc] variables
! --------------------------------------------------
   USE MOD_SPMD_Task
   USE MOD_LandPFT

      IF (p_is_worker) THEN
         IF (numpft > 0) THEN
            deallocate (tleaf_p  ) !leaf temperature [K]
            deallocate (ldew_p   ) !depth of water on foliage [mm]
            deallocate (ldew_rain_p)
            deallocate (ldew_snow_p)
            deallocate (sigf_p   ) !fraction of veg cover, excluding snow-covered veg [-]
            deallocate (tlai_p   ) !leaf area index
            deallocate (lai_p    ) !leaf area index
            deallocate (laisun_p ) !leaf area index
            deallocate (laisha_p ) !leaf area index
            deallocate (tsai_p   ) !stem area index
            deallocate (sai_p    ) !stem area index
            deallocate (ssun_p   ) !sunlit canopy absorption for solar radiation (0-1)
            deallocate (ssha_p   ) !shaded canopy absorption for solar radiation (0-1)
            deallocate (thermk_p ) !canopy gap fraction for tir radiation
            deallocate (fshade_p ) !canopy gap fraction for tir radiation
            deallocate (extkb_p  ) !(k, g(mu)/mu) direct solar extinction coefficient
            deallocate (extkd_p  ) !diffuse and scattered diffuse PAR extinction coefficient
            deallocate (tref_p   ) !2 m height air temperature [kelvin]
            deallocate (qref_p   ) !2 m height air specific humidity
            deallocate (rst_p    ) !canopy stomatal resistance (s/m)
            deallocate (z0m_p    ) !effective roughness [m]
! Plant Hydraulic variables
            deallocate (vegwp_p  ) !vegetation water potential [mm]
            deallocate (gs0sun_p ) !working copy of sunlit stomata conductance
            deallocate (gs0sha_p ) !working copy of shalit stomata conductance
! END plant hydraulic variables
! Ozone Stress variables
            deallocate (o3coefv_sun_p ) !Ozone stress factor for photosynthesis on sunlit leaf
            deallocate (o3coefv_sha_p ) !Ozone stress factor for photosynthesis on shaded leaf
            deallocate (o3coefg_sun_p ) !Ozone stress factor for stomata on sunlit leaf
            deallocate (o3coefg_sha_p ) !Ozone stress factor for stomata on shaded leaf
            deallocate (lai_old_p     ) !lai in last time step
            deallocate (o3uptakesun_p ) !Ozone does, sunlit leaf (mmol O3/m^2)
            deallocate (o3uptakesha_p ) !Ozone does, shaded leaf (mmol O3/m^2)
            deallocate (irrig_method_p)
! Ozone Stress variables
         ENDIF
      ENDIF

#ifdef BGC
      CALL deallocate_BGCPFTimeVariables
#endif

   END SUBROUTINE deallocate_PFTimeVariables

#ifdef RangeCheck
   SUBROUTINE check_PFTimeVariables

   USE MOD_RangeCheck
   USE MOD_Namelist, only : DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, DEF_USE_IRRIGATION

   IMPLICIT NONE

      CALL check_vector_data ('tleaf_p  ', tleaf_p  )      !
      CALL check_vector_data ('ldew_p   ', ldew_p   )      !
      CALL check_vector_data ('ldew_rain_p', ldew_rain_p ) !depth of rain on foliage [mm]
      CALL check_vector_data ('ldew_snow_p', ldew_snow_p ) !depth of snow on foliage [mm]
      CALL check_vector_data ('sigf_p   ', sigf_p   )      !
      CALL check_vector_data ('tlai_p   ', tlai_p   )      !
      CALL check_vector_data ('lai_p    ', lai_p    )      !
      CALL check_vector_data ('laisun_p ', lai_p    )      !
      CALL check_vector_data ('laisha_p ', lai_p    )      !
      CALL check_vector_data ('tsai_p   ', tsai_p   )      !
      CALL check_vector_data ('sai_p    ', sai_p    )      !
      CALL check_vector_data ('ssun_p   ', ssun_p   )      !
      CALL check_vector_data ('ssha_p   ', ssha_p   )      !
      CALL check_vector_data ('thermk_p ', thermk_p )      !
      CALL check_vector_data ('fshade_p ', fshade_p )      !
      CALL check_vector_data ('extkb_p  ', extkb_p  )      !
      CALL check_vector_data ('extkd_p  ', extkd_p  )      !
      CALL check_vector_data ('tref_p   ', tref_p   )      !
      CALL check_vector_data ('qref_p   ', qref_p   )      !
      CALL check_vector_data ('rst_p    ', rst_p    )      !
      CALL check_vector_data ('z0m_p    ', z0m_p    )      !
IF(DEF_USE_PLANTHYDRAULICS)THEN
      CALL check_vector_data ('vegwp_p  ', vegwp_p  )      !
      CALL check_vector_data ('gs0sun_p ', gs0sun_p )      !
      CALL check_vector_data ('gs0sha_p ', gs0sha_p )      !
ENDIF
IF(DEF_USE_OZONESTRESS)THEN
      CALL check_vector_data ('o3coefv_sun_p', o3coefv_sun_p)
      CALL check_vector_data ('o3coefv_sha_p', o3coefv_sha_p)
      CALL check_vector_data ('o3coefg_sun_p', o3coefg_sun_p)
      CALL check_vector_data ('o3coefg_sha_p', o3coefg_sha_p)
      CALL check_vector_data ('lai_old_p    ', lai_old_p    )
      CALL check_vector_data ('o3uptakesun_p', o3uptakesun_p)
      CALL check_vector_data ('o3uptakesha_p', o3uptakesha_p)
ENDIF
IF(DEF_USE_IRRIGATION)THEN
      CALL check_vector_data ('irrig_method_p', irrig_method_p)
ENDIF

#ifdef BGC
      CALL check_BGCPFTimeVariables
#endif

   END SUBROUTINE check_PFTimeVariables
#endif

#endif
END MODULE MOD_Vars_PFTimeVariables


MODULE MOD_Vars_TimeVariables
! -------------------------------
! Created by Yongjiu Dai, 03/2014
! -------------------------------

   USE MOD_Precision
   USE MOD_TimeManager
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_Vars_PFTimeVariables
#endif
#ifdef BGC
   USE MOD_BGC_Vars_TimeVariables
#endif
#ifdef CatchLateralFlow
   USE MOD_Hydro_Vars_TimeVariables
#endif
#ifdef URBAN_MODEL
   USE MOD_Urban_Vars_TimeVariables
#endif

   IMPLICIT NONE
   SAVE
! -----------------------------------------------------------------
! Time-varying state variables which reaquired by restart run
   real(r8), allocatable :: z_sno       (:,:) ! node depth [m]
   real(r8), allocatable :: dz_sno      (:,:) ! interface depth [m]
   real(r8), allocatable :: t_soisno    (:,:) ! soil temperature [K]
   real(r8), allocatable :: wliq_soisno (:,:) ! liquid water in layers [kg/m2]
   real(r8), allocatable :: wice_soisno (:,:) ! ice lens in layers [kg/m2]
   real(r8), allocatable :: h2osoi      (:,:) ! volumetric soil water in layers [m3/m3]
   real(r8), allocatable :: smp         (:,:) ! soil matrix potential [mm]
   real(r8), allocatable :: hk          (:,:) ! hydraulic conductivity [mm h2o/s]
   real(r8), allocatable :: rootr       (:,:) ! transpiration contribution fraction from different layers
   real(r8), allocatable :: rootflux    (:,:) ! water exchange between soil and root. Positive: soil->root [?]
!Plant Hydraulic variables
   real(r8), allocatable :: vegwp       (:,:) ! vegetation water potential [mm]
   real(r8), allocatable :: gs0sun        (:) ! working copy of sunlit stomata conductance
   real(r8), allocatable :: gs0sha        (:) ! working copy of shalit stomata conductance
!END plant hydraulic variables
!Ozone stress variables
   real(r8), allocatable :: o3coefv_sun   (:) ! Ozone stress factor for photosynthesis on sunlit leaf
   real(r8), allocatable :: o3coefv_sha   (:) ! Ozone stress factor for photosynthesis on shaded leaf
   real(r8), allocatable :: o3coefg_sun   (:) ! Ozone stress factor for stomata on sunlit leaf
   real(r8), allocatable :: o3coefg_sha   (:) ! Ozone stress factor for stomata on shaded leaf
   real(r8), allocatable :: lai_old       (:) ! lai in last time step
   real(r8), allocatable :: o3uptakesun   (:) ! Ozone does, sunlit leaf (mmol O3/m^2)
   real(r8), allocatable :: o3uptakesha   (:) ! Ozone does, shaded leaf (mmol O3/m^2)
!END ozone stress variables
   real(r8), allocatable :: rstfacsun_out (:) ! factor of soil water stress on sunlit leaf
   real(r8), allocatable :: rstfacsha_out (:) ! factor of soil water stress on shaded leaf
   real(r8), allocatable :: gssun_out     (:) ! stomata conductance on sunlit leaf
   real(r8), allocatable :: gssha_out     (:) ! stomata conductance on shaded leaf
   real(r8), allocatable :: t_grnd        (:) ! ground surface temperature [K]

   real(r8), allocatable :: assimsun_out  (:) ! diagnostic sunlit leaf assim value for output
   real(r8), allocatable :: assimsha_out  (:) ! diagnostic sunlit leaf etr value for output
   real(r8), allocatable :: etrsun_out    (:) ! diagnostic shaded leaf assim for output
   real(r8), allocatable :: etrsha_out    (:) ! diagnostic shaded leaf etr for output

   real(r8), allocatable :: tleaf         (:) ! leaf temperature [K]
   real(r8), allocatable :: ldew          (:) ! depth of water on foliage [mm]
   real(r8), allocatable :: ldew_rain     (:) ! depth of rain on foliage [mm]
   real(r8), allocatable :: ldew_snow     (:) ! depth of rain on foliage [mm]
   real(r8), allocatable :: sag           (:) ! non dimensional snow age [-]
   real(r8), allocatable :: scv           (:) ! snow cover, water equivalent [mm]
   real(r8), allocatable :: snowdp        (:) ! snow depth [meter]
   real(r8), allocatable :: fveg          (:) ! fraction of vegetation cover
   real(r8), allocatable :: fsno          (:) ! fraction of snow cover on ground
   real(r8), allocatable :: sigf          (:) ! fraction of veg cover, excluding snow-covered veg [-]
   real(r8), allocatable :: green         (:) ! leaf greenness
   real(r8), allocatable :: tlai          (:) ! leaf area index
   real(r8), allocatable :: lai           (:) ! leaf area index
   real(r8), allocatable :: laisun        (:) ! leaf area index for sunlit leaf
   real(r8), allocatable :: laisha        (:) ! leaf area index for shaded leaf
   real(r8), allocatable :: tsai          (:) ! stem area index
   real(r8), allocatable :: sai           (:) ! stem area index
   real(r8), allocatable :: coszen        (:) ! cosine of solar zenith angle
   real(r8), allocatable :: alb       (:,:,:) ! averaged albedo [-]
   real(r8), allocatable :: ssun      (:,:,:) ! sunlit canopy absorption for solar radiation (0-1)
   real(r8), allocatable :: ssha      (:,:,:) ! shaded canopy absorption for solar radiation (0-1)
   real(r8), allocatable :: ssoi      (:,:,:) ! soil absorption for solar radiation (0-1)
   real(r8), allocatable :: ssno      (:,:,:) ! snow absorption for solar radiation (0-1)
   real(r8), allocatable :: thermk        (:) ! canopy gap fraction for tir radiation
   real(r8), allocatable :: extkb         (:) ! (k, g(mu)/mu) direct solar extinction coefficient
   real(r8), allocatable :: extkd         (:) ! diffuse and scattered diffuse PAR extinction coefficient
   real(r8), allocatable :: zwt           (:) ! the depth to water table [m]
   real(r8), allocatable :: wa            (:) ! water storage in aquifer [mm]
   real(r8), allocatable :: wetwat        (:) ! water storage in wetland [mm]
   real(r8), allocatable :: wat           (:) ! total water storage [mm]
   real(r8), allocatable :: wdsrf         (:) ! depth of surface water [mm]
   real(r8), allocatable :: rss           (:) ! soil surface resistance [s/m]

   real(r8), allocatable :: t_lake      (:,:) ! lake layer teperature [K]
   real(r8), allocatable :: lake_icefrac(:,:) ! lake mass fraction of lake layer that is frozen
   real(r8), allocatable :: savedtke1     (:) ! top level eddy conductivity (W/m K)

   real(r8), allocatable :: snw_rds     (:,:) ! effective grain radius (col,lyr) [microns, m-6]
   real(r8), allocatable :: mss_bcpho   (:,:) ! mass of hydrophobic BC in snow  (col,lyr) [kg]
   real(r8), allocatable :: mss_bcphi   (:,:) ! mass of hydrophillic BC in snow (col,lyr) [kg]
   real(r8), allocatable :: mss_ocpho   (:,:) ! mass of hydrophobic OC in snow  (col,lyr) [kg]
   real(r8), allocatable :: mss_ocphi   (:,:) ! mass of hydrophillic OC in snow (col,lyr) [kg]
   real(r8), allocatable :: mss_dst1    (:,:) ! mass of dust species 1 in snow  (col,lyr) [kg]
   real(r8), allocatable :: mss_dst2    (:,:) ! mass of dust species 2 in snow  (col,lyr) [kg]
   real(r8), allocatable :: mss_dst3    (:,:) ! mass of dust species 3 in snow  (col,lyr) [kg]
   real(r8), allocatable :: mss_dst4    (:,:) ! mass of dust species 4 in snow  (col,lyr) [kg]
   real(r8), allocatable :: ssno_lyr(:,:,:,:) ! snow layer absorption [-]

   real(r8), allocatable :: trad          (:) ! radiative temperature of surface [K]
   real(r8), allocatable :: tref          (:) ! 2 m height air temperature [kelvin]
   real(r8), allocatable :: qref          (:) ! 2 m height air specific humidity
   real(r8), allocatable :: rst           (:) ! canopy stomatal resistance (s/m)
   real(r8), allocatable :: emis          (:) ! averaged bulk surface emissivity
   real(r8), allocatable :: z0m           (:) ! effective roughness [m]
   real(r8), allocatable :: displa        (:) ! zero displacement height [m]
   real(r8), allocatable :: zol           (:) ! dimensionless height (z/L) used in Monin-Obukhov theory
   real(r8), allocatable :: rib           (:) ! bulk Richardson number in surface layer
   real(r8), allocatable :: ustar         (:) ! u* in similarity theory [m/s]
   real(r8), allocatable :: qstar         (:) ! q* in similarity theory [kg/kg]
   real(r8), allocatable :: tstar         (:) ! t* in similarity theory [K]
   real(r8), allocatable :: fm            (:) ! integral of profile FUNCTION for momentum
   real(r8), allocatable :: fh            (:) ! integral of profile FUNCTION for heat
   real(r8), allocatable :: fq            (:) ! integral of profile FUNCTION for moisture

   real(r8), allocatable :: irrig_rate          (:) ! irrigation rate [mm s-1]
   real(r8), allocatable :: deficit_irrig       (:) ! irrigation amount [kg/m2]
   real(r8), allocatable :: sum_irrig           (:) ! total irrigation amount [kg/m2]
   real(r8), allocatable :: sum_irrig_count     (:) ! total irrigation counts [-]
   integer , allocatable :: n_irrig_steps_left  (:) ! left steps for once irrigation [-]
   real(r8), allocatable :: tairday                       (:) ! daily mean temperature [degree C]
   real(r8), allocatable :: usday                         (:) ! daily mean wind component in eastward direction [m/s]
   real(r8), allocatable :: vsday                         (:) ! daily mean wind component in northward direction [m/s]
   real(r8), allocatable :: pairday                       (:) ! daily mean pressure [kPa]
   real(r8), allocatable :: rnetday                       (:) ! daily net radiation flux [MJ/m2/day]
   real(r8), allocatable :: fgrndday                      (:) ! daily ground heat flux [MJ/m2/day]
   real(r8), allocatable :: potential_evapotranspiration  (:) ! daily potential evapotranspiration [mm/day]

   integer , allocatable :: irrig_method_corn      (:) ! irrigation method for corn (0-3)
   integer , allocatable :: irrig_method_swheat    (:) ! irrigation method for spring wheat (0-3)
   integer , allocatable :: irrig_method_wwheat    (:) ! irrigation method for winter wheat (0-3)
   integer , allocatable :: irrig_method_soybean   (:) ! irrigation method for soybean (0-3)
   integer , allocatable :: irrig_method_cotton    (:) ! irrigation method for cotton (0-3)
   integer , allocatable :: irrig_method_rice1     (:) ! irrigation method for rice1 (0-3)
   integer , allocatable :: irrig_method_rice2     (:) ! irrigation method for rice2 (0-3)
   integer , allocatable :: irrig_method_sugarcane (:) ! irrigation method for sugarcane (0-3)
   ! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: allocate_TimeVariables
   PUBLIC :: deallocate_TimeVariables
   PUBLIC :: READ_TimeVariables
   PUBLIC :: WRITE_TimeVariables
#ifdef RangeCheck
   PUBLIC :: check_TimeVariables
#endif


!-----------------------------------------------------------------------

CONTAINS

!-----------------------------------------------------------------------

   SUBROUTINE allocate_TimeVariables (numpatch)
! --------------------------------------------------------------------
! Allocates memory for CoLM 1d [numpatch] variables
! ------------------------------------------------------

   USE MOD_Precision
   USE MOD_Vars_Global
   USE MOD_SPMD_Task
   ! USE MOD_LandPatch, only: numpatch
   IMPLICIT NONE
   integer :: numpatch


      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN

            allocate (z_sno      (maxsnl+1:0,      numpatch)); z_sno       (:,:) = spval
            allocate (dz_sno     (maxsnl+1:0,      numpatch)); dz_sno      (:,:) = spval
            allocate (t_soisno   (maxsnl+1:nl_soil,numpatch)); t_soisno    (:,:) = spval
            allocate (wliq_soisno(maxsnl+1:nl_soil,numpatch)); wliq_soisno (:,:) = spval
            allocate (wice_soisno(maxsnl+1:nl_soil,numpatch)); wice_soisno (:,:) = spval
            allocate (smp               (1:nl_soil,numpatch)); smp         (:,:) = spval
            allocate (hk                (1:nl_soil,numpatch)); hk          (:,:) = spval
            allocate (h2osoi            (1:nl_soil,numpatch)); h2osoi      (:,:) = spval
            allocate (rootr             (1:nl_soil,numpatch)); rootr       (:,:) = spval
            allocate (rootflux          (1:nl_soil,numpatch)); rootflux    (:,:) = spval
!Plant Hydraulic variables
            allocate (vegwp             (1:nvegwcs,numpatch)); vegwp       (:,:) = spval
            allocate (gs0sun                      (numpatch)); gs0sun        (:) = spval
            allocate (gs0sha                      (numpatch)); gs0sha        (:) = spval
!END plant hydraulic variables
!Ozone Stress variables
            allocate (o3coefv_sun                 (numpatch)); o3coefv_sun   (:) = spval
            allocate (o3coefv_sha                 (numpatch)); o3coefv_sha   (:) = spval
            allocate (o3coefg_sun                 (numpatch)); o3coefg_sun   (:) = spval
            allocate (o3coefg_sha                 (numpatch)); o3coefg_sha   (:) = spval
            allocate (lai_old                     (numpatch)); lai_old       (:) = spval
            allocate (o3uptakesun                 (numpatch)); o3uptakesun   (:) = spval
            allocate (o3uptakesha                 (numpatch)); o3uptakesha   (:) = spval
!END ozone stress variables

            allocate (rstfacsun_out               (numpatch)); rstfacsun_out (:) = spval
            allocate (rstfacsha_out               (numpatch)); rstfacsha_out (:) = spval
            allocate (gssun_out                   (numpatch)); gssun_out     (:) = spval
            allocate (gssha_out                   (numpatch)); gssha_out     (:) = spval
            allocate (assimsun_out                (numpatch)); assimsun_out  (:) = spval
            allocate (assimsha_out                (numpatch)); assimsha_out  (:) = spval
            allocate (etrsun_out                  (numpatch)); etrsun_out    (:) = spval
            allocate (etrsha_out                  (numpatch)); etrsha_out    (:) = spval

            allocate (t_grnd                      (numpatch)); t_grnd        (:) = spval
            allocate (tleaf                       (numpatch)); tleaf         (:) = spval
            allocate (ldew                        (numpatch)); ldew          (:) = spval
            allocate (ldew_rain                   (numpatch)); ldew_rain     (:) = spval
            allocate (ldew_snow                   (numpatch)); ldew_snow     (:) = spval
            allocate (sag                         (numpatch)); sag           (:) = spval
            allocate (scv                         (numpatch)); scv           (:) = spval
            allocate (snowdp                      (numpatch)); snowdp        (:) = spval
            allocate (fveg                        (numpatch)); fveg          (:) = spval
            allocate (fsno                        (numpatch)); fsno          (:) = spval
            allocate (sigf                        (numpatch)); sigf          (:) = spval
            allocate (green                       (numpatch)); green         (:) = spval
            allocate (tlai                        (numpatch)); tlai          (:) = spval
            allocate (lai                         (numpatch)); lai           (:) = spval
            allocate (laisun                      (numpatch)); laisun        (:) = spval
            allocate (laisha                      (numpatch)); laisha        (:) = spval
            allocate (tsai                        (numpatch)); tsai          (:) = spval
            allocate (sai                         (numpatch)); sai           (:) = spval
            allocate (coszen                      (numpatch)); coszen        (:) = spval
            allocate (alb                     (2,2,numpatch)); alb       (:,:,:) = spval
            allocate (ssun                    (2,2,numpatch)); ssun      (:,:,:) = spval
            allocate (ssha                    (2,2,numpatch)); ssha      (:,:,:) = spval
            allocate (ssoi                    (2,2,numpatch)); ssoi      (:,:,:) = spval
            allocate (ssno                    (2,2,numpatch)); ssno      (:,:,:) = spval
            allocate (thermk                      (numpatch)); thermk        (:) = spval
            allocate (extkb                       (numpatch)); extkb         (:) = spval
            allocate (extkd                       (numpatch)); extkd         (:) = spval
            allocate (zwt                         (numpatch)); zwt           (:) = spval
            allocate (wa                          (numpatch)); wa            (:) = spval
            allocate (wetwat                      (numpatch)); wetwat        (:) = spval
            allocate (wat                         (numpatch)); wat           (:) = spval
            allocate (wdsrf                       (numpatch)); wdsrf         (:) = spval
            allocate (rss                         (numpatch)); rss           (:) = spval
            allocate (t_lake              (nl_lake,numpatch)); t_lake      (:,:) = spval
            allocate (lake_icefrac        (nl_lake,numpatch)); lake_icefrac(:,:) = spval
            allocate (savedtke1                   (numpatch)); savedtke1     (:) = spval

            allocate (snw_rds          (maxsnl+1:0,numpatch)); snw_rds     (:,:) = spval
            allocate (mss_bcpho        (maxsnl+1:0,numpatch)); mss_bcpho   (:,:) = spval
            allocate (mss_bcphi        (maxsnl+1:0,numpatch)); mss_bcphi   (:,:) = spval
            allocate (mss_ocpho        (maxsnl+1:0,numpatch)); mss_ocpho   (:,:) = spval
            allocate (mss_ocphi        (maxsnl+1:0,numpatch)); mss_ocphi   (:,:) = spval
            allocate (mss_dst1         (maxsnl+1:0,numpatch)); mss_dst1    (:,:) = spval
            allocate (mss_dst2         (maxsnl+1:0,numpatch)); mss_dst2    (:,:) = spval
            allocate (mss_dst3         (maxsnl+1:0,numpatch)); mss_dst3    (:,:) = spval
            allocate (mss_dst4         (maxsnl+1:0,numpatch)); mss_dst4    (:,:) = spval
            allocate (ssno_lyr     (2,2,maxsnl+1:1,numpatch)); ssno_lyr(:,:,:,:) = spval

            allocate (trad                        (numpatch)); trad          (:) = spval
            allocate (tref                        (numpatch)); tref          (:) = spval
            allocate (qref                        (numpatch)); qref          (:) = spval
            allocate (rst                         (numpatch)); rst           (:) = spval
            allocate (emis                        (numpatch)); emis          (:) = spval
            allocate (z0m                         (numpatch)); z0m           (:) = spval
            allocate (displa                      (numpatch)); displa        (:) = spval
            allocate (zol                         (numpatch)); zol           (:) = spval
            allocate (rib                         (numpatch)); rib           (:) = spval
            allocate (ustar                       (numpatch)); ustar         (:) = spval
            allocate (qstar                       (numpatch)); qstar         (:) = spval
            allocate (tstar                       (numpatch)); tstar         (:) = spval
            allocate (fm                          (numpatch)); fm            (:) = spval
            allocate (fh                          (numpatch)); fh            (:) = spval
            allocate (fq                          (numpatch)); fq            (:) = spval

            allocate ( irrig_rate                 (numpatch)); irrig_rate             (:) = spval
            allocate ( deficit_irrig              (numpatch)); deficit_irrig          (:) = spval
            allocate ( sum_irrig                  (numpatch)); sum_irrig              (:) = spval
            allocate ( sum_irrig_count            (numpatch)); sum_irrig_count        (:) = spval
            allocate ( n_irrig_steps_left         (numpatch)); n_irrig_steps_left     (:) = spval_i4
            allocate ( tairday                    (numpatch)); tairday                (:) = spval
            allocate ( usday                      (numpatch)); usday                  (:) = spval
            allocate ( vsday                      (numpatch)); vsday                  (:) = spval
            allocate ( pairday                    (numpatch)); pairday                (:) = spval
            allocate ( rnetday                    (numpatch)); rnetday                (:) = spval
            allocate ( fgrndday                   (numpatch)); fgrndday               (:) = spval
            allocate ( potential_evapotranspiration(numpatch)); potential_evapotranspiration(:) = spval

            allocate ( irrig_method_corn          (numpatch)); irrig_method_corn      (:) = spval_i4
            allocate ( irrig_method_swheat        (numpatch)); irrig_method_swheat    (:) = spval_i4
            allocate ( irrig_method_wwheat        (numpatch)); irrig_method_wwheat    (:) = spval_i4
            allocate ( irrig_method_soybean       (numpatch)); irrig_method_soybean   (:) = spval_i4
            allocate ( irrig_method_cotton        (numpatch)); irrig_method_cotton    (:) = spval_i4
            allocate ( irrig_method_rice1         (numpatch)); irrig_method_rice1     (:) = spval_i4
            allocate ( irrig_method_rice2         (numpatch)); irrig_method_rice2     (:) = spval_i4
            allocate ( irrig_method_sugarcane     (numpatch)); irrig_method_sugarcane (:) = spval_i4

         ENDIF
      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL allocate_PFTimeVariables
#endif

#ifdef BGC
      CALL allocate_BGCTimeVariables
#endif

#ifdef CatchLateralFlow
      CALL allocate_HydroTimeVariables
#endif

#ifdef URBAN_MODEL
      CALL allocate_UrbanTimeVariables
#endif

   END SUBROUTINE allocate_TimeVariables



   SUBROUTINE deallocate_TimeVariables (numpatch)

   USE MOD_SPMD_Task
   !USE MOD_LandPatch, only: numpatch
   IMPLICIT NONE
   integer :: numpatch

      ! --------------------------------------------------
      ! Deallocates memory for CoLM 1d [numpatch] variables
      ! --------------------------------------------------

      IF (p_is_worker) THEN

         IF (numpatch > 0) THEN

            deallocate (z_sno                  )
            deallocate (dz_sno                 )
            deallocate (t_soisno               )
            deallocate (wliq_soisno            )
            deallocate (wice_soisno            )
            deallocate (smp                    )
            deallocate (hk                     )
            deallocate (h2osoi                 )
            deallocate (rootr                  )
            deallocate (rootflux               )
!Plant Hydraulic variables
            deallocate (vegwp                  )
            deallocate (gs0sun                 )
            deallocate (gs0sha                 )
!END plant hydraulic variables
!Ozone stress variables
            deallocate (o3coefv_sun            ) ! Ozone stress factor for photosynthesis on sunlit leaf
            deallocate (o3coefv_sha            ) ! Ozone stress factor for photosynthesis on shaded leaf
            deallocate (o3coefg_sun            ) ! Ozone stress factor for stomata on sunlit leaf
            deallocate (o3coefg_sha            ) ! Ozone stress factor for stomata on shaded leaf
            deallocate (lai_old                ) ! lai in last time step
            deallocate (o3uptakesun            ) ! Ozone does, sunlit leaf (mmol O3/m^2)
            deallocate (o3uptakesha            ) ! Ozone does, shaded leaf (mmol O3/m^2)
!END Ozone stress variables
            deallocate (rstfacsun_out          )
            deallocate (rstfacsha_out          )
            deallocate (gssun_out              )
            deallocate (gssha_out              )
            deallocate (assimsun_out           )
            deallocate (assimsha_out           )
            deallocate (etrsun_out             )
            deallocate (etrsha_out             )

            deallocate (t_grnd                 )
            deallocate (tleaf                  )
            deallocate (ldew                   )
            deallocate (ldew_rain              )
            deallocate (ldew_snow              )
            deallocate (sag                    )
            deallocate (scv                    )
            deallocate (snowdp                 )
            deallocate (fveg                   )
            deallocate (fsno                   )
            deallocate (sigf                   )
            deallocate (green                  )
            deallocate (tlai                   )
            deallocate (lai                    )
            deallocate (laisun                 )
            deallocate (laisha                 )
            deallocate (tsai                   )
            deallocate (sai                    )
            deallocate (coszen                 )
            deallocate (alb                    )
            deallocate (ssun                   )
            deallocate (ssha                   )
            deallocate (ssoi                   )
            deallocate (ssno                   )
            deallocate (thermk                 )
            deallocate (extkb                  )
            deallocate (extkd                  )
            deallocate (zwt                    )
            deallocate (wa                     )
            deallocate (wetwat                 )
            deallocate (wat                    )
            deallocate (wdsrf                  )
            deallocate (rss                    )

            deallocate (t_lake                 ) ! new lake scheme
            deallocate (lake_icefrac           ) ! new lake scheme
            deallocate (savedtke1              ) ! new lake scheme

            deallocate (snw_rds                )
            deallocate (mss_bcpho              )
            deallocate (mss_bcphi              )
            deallocate (mss_ocpho              )
            deallocate (mss_ocphi              )
            deallocate (mss_dst1               )
            deallocate (mss_dst2               )
            deallocate (mss_dst3               )
            deallocate (mss_dst4               )
            deallocate (ssno_lyr               )

            deallocate (trad                   )
            deallocate (tref                   )
            deallocate (qref                   )
            deallocate (rst                    )
            deallocate (emis                   )
            deallocate (z0m                    )
            deallocate (displa                 )
            deallocate (zol                    )
            deallocate (rib                    )
            deallocate (ustar                  )
            deallocate (qstar                  )
            deallocate (tstar                  )
            deallocate (fm                     )
            deallocate (fh                     )
            deallocate (fq                     )

            deallocate (irrig_rate             )
            deallocate (deficit_irrig          )
            deallocate (sum_irrig              )
            deallocate (sum_irrig_count        )
            deallocate (n_irrig_steps_left     )

            deallocate (tairday                )
            deallocate (usday                  )
            deallocate (vsday                  )
            deallocate (pairday                )
            deallocate (rnetday                )
            deallocate (fgrndday               )
            deallocate (potential_evapotranspiration)

            deallocate ( irrig_method_corn     )
            deallocate ( irrig_method_swheat   )
            deallocate ( irrig_method_wwheat   )
            deallocate ( irrig_method_soybean  )
            deallocate ( irrig_method_cotton   )
            deallocate ( irrig_method_rice1    )
            deallocate ( irrig_method_rice2    )
            deallocate ( irrig_method_sugarcane)
         ENDIF
      ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL deallocate_PFTimeVariables
#endif

#if (defined BGC)
      CALL deallocate_BGCTimeVariables
#endif

#ifdef CatchLateralFlow
      CALL deallocate_HydroTimeVariables
#endif

#if (defined URBAN_MODEL)
      CALL deallocate_UrbanTimeVariables
#endif

   END SUBROUTINE deallocate_TimeVariables


   !---------------------------------------
   FUNCTION save_to_restart (idate, deltim, itstamp, ptstamp) result(rwrite)

   USE MOD_Namelist
   IMPLICIT NONE

   logical :: rwrite

   integer,  intent(in) :: idate(3)
   real(r8), intent(in) :: deltim
   type(timestamp), intent(in) :: itstamp, ptstamp


      ! added by yuan, 08/31/2014
      SELECTCASE (trim(adjustl(DEF_WRST_FREQ)))
      CASE ('TIMESTEP')
         rwrite = .true.
      CASE ('HOURLY')
         rwrite = isendofhour (idate, deltim)
      CASE ('DAILY')
         rwrite = isendofday(idate, deltim)
      CASE ('MONTHLY')
         rwrite = isendofmonth(idate, deltim)
      CASE ('YEARLY')
         rwrite = isendofyear(idate, deltim)
      CASE default
         write(*,*) 'Warning: Please USE one of TIMESTEP/HOURLY/DAILY/MONTHLY/YEARLY for restart frequency.'
      ENDSELECT

      IF (rwrite) THEN
         ! rwrite = (ptstamp <= itstamp)
      ENDIF

   END FUNCTION save_to_restart

   !---------------------------------------
   SUBROUTINE WRITE_TimeVariables (istep_pf, rank)

   !=======================================================================
   ! Original version: Yongjiu Dai, September 15, 1999, 03/2014
   !=======================================================================

   USE MOD_SPMD_Task
   USE MOD_Namelist, only : DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, DEF_USE_IRRIGATION

   IMPLICIT NONE

   integer :: istep_pf, rank
   character*100 RI

   if(p_is_master)then
      write(*,*)'Write CoLM Active Restart: istep_pf = ',istep_pf
   endif

   write(RI,*) rank
   open(40,file='CoLM.TimeVariables.rst.'//trim(adjustl(RI)),form='unformatted')

      write(40) z_sno
      write(40) dz_sno
      write(40) t_soisno
      write(40) wliq_soisno
      write(40) wice_soisno
      write(40) smp
      write(40) hk
IF(DEF_USE_PLANTHYDRAULICS)THEN
      write(40) vegwp
      write(40) gs0sun
      write(40) gs0sha
ENDIF
IF(DEF_USE_OZONESTRESS)THEN
      write(40) lai_old
      write(40) o3uptakesun
      write(40) o3uptakesha
ENDIF
      write(40) t_grnd
      write(40) tleaf
      write(40) ldew
      write(40) ldew_rain
      write(40) ldew_snow
      write(40) sag
      write(40) scv
      write(40) snowdp
      write(40) fveg
      write(40) fsno
      write(40) sigf
      write(40) green
      write(40) lai
      write(40) tlai
      write(40) sai
      write(40) tsai
      write(40) coszen
      write(40) alb
      write(40) ssun
      write(40) ssha
      write(40) ssoi
      write(40) ssno
      write(40) thermk
      write(40) extkb
      write(40) extkd
      write(40) zwt
      write(40) wa
      write(40) wetwat
      write(40) wdsrf
      write(40) rss

      write(40) t_lake
      write(40) lake_icefrac
      write(40) savedtke1
      write(40) snw_rds
      write(40) mss_bcpho
      write(40) mss_bcphi
      write(40) mss_ocpho
      write(40) mss_ocphi
      write(40) mss_dst1
      write(40) mss_dst2
      write(40) mss_dst3
      write(40) mss_dst4
      write(40) ssno_lyr

      write(40) trad
      write(40) tref
      write(40) qref
      write(40) rst
      write(40) emis
      write(40) z0m
      write(40) zol
      write(40) rib
      write(40) ustar
      write(40) qstar
      write(40) tstar
      write(40) fm
      write(40) fh
      write(40) fq

IF (DEF_USE_IRRIGATION) THEN
      write(40) irrig_rate
      write(40) deficit_irrig
      write(40) sum_irrig
      write(40) sum_irrig_count
      write(40) n_irrig_steps_left
      write(40) tairday
      write(40) usday
      write(40) vsday
      write(40) pairday
      write(40) rnetday
      write(40) fgrndday
      write(40) potential_evapotranspiration
      write(40) irrig_method_corn
      write(40) irrig_method_swheat
      write(40) irrig_method_wwheat
      write(40) irrig_method_soybean
      write(40) irrig_method_cotton
      write(40) irrig_method_rice1
      write(40) irrig_method_rice2
      write(40) irrig_method_sugarcane
ENDIF

   close(40)

   END SUBROUTINE WRITE_TimeVariables

   SUBROUTINE READ_TimeVariables (rank)

      !=======================================================================
      ! Original version: Yongjiu Dai, September 15, 1999, 03/2014
      !=======================================================================

      USE MOD_SPMD_Task
      USE MOD_Namelist, only : DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, DEF_USE_IRRIGATION

      IMPLICIT NONE

      integer :: rank
      character*100 RI

      write(RI,*) rank
      open(40,file='CoLM.TimeVariables.rst.'//trim(adjustl(RI)),form='unformatted')

         read(40) z_sno
         read(40) dz_sno
         read(40) t_soisno
         read(40) wliq_soisno
         read(40) wice_soisno
         read(40) smp
         read(40) hk
   IF(DEF_USE_PLANTHYDRAULICS)THEN
         read(40) vegwp
         read(40) gs0sun
         read(40) gs0sha
   ENDIF
   IF(DEF_USE_OZONESTRESS)THEN
         read(40) lai_old
         read(40) o3uptakesun
         read(40) o3uptakesha
   ENDIF
         read(40) t_grnd
         read(40) tleaf
         read(40) ldew
         read(40) ldew_rain
         read(40) ldew_snow
         read(40) sag
         read(40) scv
         read(40) snowdp
         read(40) fveg
         read(40) fsno
         read(40) sigf
         read(40) green
         read(40) lai
         read(40) tlai
         read(40) sai
         read(40) tsai
         read(40) coszen
         read(40) alb
         read(40) ssun
         read(40) ssha
         read(40) ssoi
         read(40) ssno
         read(40) thermk
         read(40) extkb
         read(40) extkd
         read(40) zwt
         read(40) wa
         read(40) wetwat
         read(40) wdsrf
         read(40) rss

         read(40) t_lake
         read(40) lake_icefrac
         read(40) savedtke1
         read(40) snw_rds
         read(40) mss_bcpho
         read(40) mss_bcphi
         read(40) mss_ocpho
         read(40) mss_ocphi
         read(40) mss_dst1
         read(40) mss_dst2
         read(40) mss_dst3
         read(40) mss_dst4
         read(40) ssno_lyr

         read(40) trad
         read(40) tref
         read(40) qref
         read(40) rst
         read(40) emis
         read(40) z0m
         read(40) zol
         read(40) rib
         read(40) ustar
         read(40) qstar
         read(40) tstar
         read(40) fm
         read(40) fh
         read(40) fq

   IF (DEF_USE_IRRIGATION) THEN
         read(40) irrig_rate
         read(40) deficit_irrig
         read(40) sum_irrig
         read(40) sum_irrig_count
         read(40) n_irrig_steps_left
         read(40) tairday
         read(40) usday
         read(40) vsday
         read(40) pairday
         read(40) rnetday
         read(40) fgrndday
         read(40) potential_evapotranspiration
         read(40) irrig_method_corn
         read(40) irrig_method_swheat
         read(40) irrig_method_wwheat
         read(40) irrig_method_soybean
         read(40) irrig_method_cotton
         read(40) irrig_method_rice1
         read(40) irrig_method_rice2
         read(40) irrig_method_sugarcane
   ENDIF

      close(40)

      IF (p_is_master) THEN
         write(*,'(A29)') 'Loading Time Variables done.'
      ENDIF

   END SUBROUTINE READ_TimeVariables
  !---------------------------------------
#ifdef RangeCheck
   SUBROUTINE check_TimeVariables ()

   USE MOD_SPMD_Task
   USE MOD_RangeCheck
   USE MOD_Namelist, only: DEF_USE_PLANTHYDRAULICS, DEF_USE_OZONESTRESS, DEF_USE_IRRIGATION, &
                           DEF_USE_SNICAR

   IMPLICIT NONE

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif
      IF (p_is_master) THEN
         write(*,'(/,A27)') 'Checking Time Variables ...'
      ENDIF

      CALL check_vector_data ('t_grnd      [K]    ', t_grnd     ) ! ground surface temperature [K]
      CALL check_vector_data ('tleaf       [K]    ', tleaf      ) ! leaf temperature [K]
      CALL check_vector_data ('ldew        [mm]   ', ldew       ) ! depth of water on foliage [mm]
      CALL check_vector_data ('ldew_rain   [mm]   ', ldew_rain  ) ! depth of rain on foliage [mm]
      CALL check_vector_data ('ldew_snow   [mm]   ', ldew_snow  ) ! depth of snow on foliage [mm]
      CALL check_vector_data ('sag         [-]    ', sag        ) ! non dimensional snow age [-]
      CALL check_vector_data ('scv         [mm]   ', scv        ) ! snow cover, water equivalent [mm]
      CALL check_vector_data ('snowdp      [m]    ', snowdp     ) ! snow depth [meter]
      CALL check_vector_data ('fveg        [-]    ', fveg       ) ! fraction of vegetation cover
      CALL check_vector_data ('fsno        [-]    ', fsno       ) ! fraction of snow cover on ground
      CALL check_vector_data ('sigf        [-]    ', sigf       ) ! fraction of veg cover, excluding snow-covered veg [-]
      CALL check_vector_data ('green       [-]    ', green      ) ! leaf greenness
      CALL check_vector_data ('lai         [-]    ', lai        ) ! leaf area index
      CALL check_vector_data ('tlai        [-]    ', tlai       ) ! leaf area index
      CALL check_vector_data ('sai         [-]    ', sai        ) ! stem area index
      CALL check_vector_data ('tsai        [-]    ', tsai       ) ! stem area index
      CALL check_vector_data ('coszen      [-]    ', coszen     ) ! cosine of solar zenith angle
      CALL check_vector_data ('alb         [-]    ', alb        ) ! averaged albedo [-]
      CALL check_vector_data ('ssun        [-]    ', ssun       ) ! sunlit canopy absorption for solar radiation (0-1)
      CALL check_vector_data ('ssha        [-]    ', ssha       ) ! shaded canopy absorption for solar radiation (0-1)
      CALL check_vector_data ('ssoi        [-]    ', ssoi       ) ! soil absorption for solar radiation (0-1)
      CALL check_vector_data ('ssno        [-]    ', ssno       ) ! snow absorption for solar radiation (0-1)
      CALL check_vector_data ('thermk      [-]    ', thermk     ) ! canopy gap fraction for tir radiation
      CALL check_vector_data ('extkb       [-]    ', extkb      ) ! (k, g(mu)/mu) direct solar extinction coefficient
      CALL check_vector_data ('extkd       [-]    ', extkd      ) ! diffuse and scattered diffuse PAR extinction coefficient
      CALL check_vector_data ('zwt         [m]    ', zwt        ) ! the depth to water table [m]
      CALL check_vector_data ('wa          [mm]   ', wa         ) ! water storage in aquifer [mm]
      CALL check_vector_data ('wetwat      [mm]   ', wetwat     ) ! water storage in wetland [mm]
      CALL check_vector_data ('wdsrf       [mm]   ', wdsrf      ) ! depth of surface water [mm]
      CALL check_vector_data ('rss         [s/m]  ', rss        ) ! soil surface resistance [s/m]
      CALL check_vector_data ('t_lake      [K]    ', t_lake      )!
      CALL check_vector_data ('lake_icefrc [-]    ', lake_icefrac)!
      CALL check_vector_data ('savedtke1   [W/m K]', savedtke1   )!
      CALL check_vector_data ('z_sno       [m]    ', z_sno )      ! node depth [m]
      CALL check_vector_data ('dz_sno      [m]    ', dz_sno)      ! interface depth [m]
      CALL check_vector_data ('t_soisno    [K]    ', t_soisno   ) ! soil temperature [K]
      CALL check_vector_data ('wliq_soisno [kg/m2]', wliq_soisno) ! liquid water in layers [kg/m2]
      CALL check_vector_data ('wice_soisno [kg/m2]', wice_soisno) ! ice lens in layers [kg/m2]
      CALL check_vector_data ('smp         [mm]   ', smp        ) ! soil matrix potential [mm]
      CALL check_vector_data ('hk          [mm/s] ', hk         ) ! hydraulic conductivity [mm h2o/s]
IF(DEF_USE_PLANTHYDRAULICS)THEN
      CALL check_vector_data ('vegwp       [m]    ', vegwp      ) ! vegetation water potential [mm]
      CALL check_vector_data ('gs0sun      []     ', gs0sun     ) ! working copy of sunlit stomata conductance
      CALL check_vector_data ('gs0sha      []     ', gs0sha     ) ! working copy of shalit stomata conductance
ENDIF
IF(DEF_USE_OZONESTRESS)THEN
      CALL check_vector_data ('o3coefv_sun        ', o3coefv_sun)
      CALL check_vector_data ('o3coefv_sha        ', o3coefv_sha)
      CALL check_vector_data ('o3coefg_sun        ', o3coefg_sun)
      CALL check_vector_data ('o3coefg_sha        ', o3coefg_sha)
      CALL check_vector_data ('lai_old            ', lai_old    )
      CALL check_vector_data ('o3uptakesun        ', o3uptakesun)
      CALL check_vector_data ('o3uptakesha        ', o3uptakesha)
ENDIF

IF (DEF_USE_SNICAR) THEN
      CALL check_vector_data ('snw_rds     [m-6]  ',  snw_rds   ) !
      CALL check_vector_data ('mss_bcpho   [Kg]   ',  mss_bcpho ) !
      CALL check_vector_data ('mss_bcphi   [Kg]   ',  mss_bcphi ) !
      CALL check_vector_data ('mss_ocpho   [Kg]   ',  mss_ocpho ) !
      CALL check_vector_data ('mss_ocphi   [Kg]   ',  mss_ocphi ) !
      CALL check_vector_data ('mss_dst1    [Kg]   ',  mss_dst1  ) !
      CALL check_vector_data ('mss_dst2    [Kg]   ',  mss_dst2  ) !
      CALL check_vector_data ('mss_dst3    [Kg]   ',  mss_dst3  ) !
      CALL check_vector_data ('mss_dst4    [Kg]   ',  mss_dst4  ) !
      CALL check_vector_data ('ssno_lyr    [-]    ',  ssno_lyr  ) !
ENDIF

IF (DEF_USE_IRRIGATION) THEN
      CALL check_vector_data ('irrig_rate            ' , irrig_rate            )
      CALL check_vector_data ('deficit_irrig         ' , deficit_irrig         )
      CALL check_vector_data ('sum_irrig             ' , sum_irrig             )
      CALL check_vector_data ('sum_irrig_count       ' , sum_irrig_count       )
      CALL check_vector_data ('n_irrig_steps_left    ' , n_irrig_steps_left    )
      CALL check_vector_data ('tairday               ' , tairday               )
      CALL check_vector_data ('usday                 ' , usday                 )
      CALL check_vector_data ('vsday                 ' , vsday                 )
      CALL check_vector_data ('pairday               ' , pairday               )
      CALL check_vector_data ('rnetday               ' , rnetday               )
      CALL check_vector_data ('fgrndday              ' , fgrndday              )
      CALL check_vector_data ('potential_evapotranspiration' , potential_evapotranspiration)
      CALL check_vector_data ('irrig_method_corn     ' , irrig_method_corn     )
      CALL check_vector_data ('irrig_method_swheat   ' , irrig_method_swheat   )
      CALL check_vector_data ('irrig_method_wwheat   ' , irrig_method_wwheat   )
      CALL check_vector_data ('irrig_method_soybean  ' , irrig_method_soybean  )
      CALL check_vector_data ('irrig_method_cotton   ' , irrig_method_cotton   )
      CALL check_vector_data ('irrig_method_rice1    ' , irrig_method_rice1    )
      CALL check_vector_data ('irrig_method_rice2    ' , irrig_method_rice2    )
      CALL check_vector_data ('irrig_method_sugarcane' , irrig_method_sugarcane)
ENDIF

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
      CALL check_PFTimeVariables
#endif

#if (defined BGC)
      CALL check_BGCTimeVariables
#endif

#ifdef USEMPI
      CALL mpi_barrier (p_comm_glb, p_err)
#endif

  END SUBROUTINE check_TimeVariables
#endif


END MODULE MOD_Vars_TimeVariables
! ---------- EOP ------------
