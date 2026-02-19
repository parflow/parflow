#include <define.h>

#ifdef BGC
MODULE MOD_NdepData
 !-----------------------------------------------------------------------
 ! !DESCRIPTION:
 ! This module read in ndep data.
 !
 ! !ORIGINAL:
 ! Lu Xingjie and Zhang Shupeng, 2023, prepare the original version of the ndep data module.

   USE MOD_Grid
   USE MOD_SpatialMapping
   USE MOD_BGC_Vars_TimeVariables, only : ndep
   USE MOD_BGC_Vars_1DFluxes, only: ndep_to_sminn
   IMPLICIT NONE

   character(len=256) :: file_ndep

   type(grid_type) :: grid_ndep
   type(spatial_mapping_type) :: mg2p_ndep

CONTAINS

   ! ----------
   SUBROUTINE init_ndep_data_annually (YY)

   !----------------------
   ! DESCTIPTION:
   ! open ndep netcdf file from DEF_dir_runtime, read latitude and longitude info.
   ! Initialize ndep data read in.

   USE MOD_TimeManager
   USE MOD_Namelist
   USE MOD_Grid
   USE MOD_NetCDFSerial
   USE MOD_LandPatch
   IMPLICIT NONE

   integer, intent(in) :: YY

   ! Local Variables
   real(r8), allocatable :: lat(:), lon(:)

      file_ndep = trim(DEF_dir_runtime) // '/ndep/fndep_colm_hist_simyr1849-2006_1.9x2.5_c100428.nc'

      CALL ncio_read_bcast_serial (file_ndep, 'lat', lat)
      CALL ncio_read_bcast_serial (file_ndep, 'lon', lon)

      CALL grid_ndep%define_by_center (lat, lon)

      CALL mg2p_ndep%build_arealweighted (grid_ndep, landpatch)

      IF (allocated(lon)) deallocate(lon)
      IF (allocated(lat)) deallocate(lat)

      CALL update_ndep_data_annually (YY, iswrite = .true.)

   END SUBROUTINE init_ndep_data_annually

      ! ----------
   SUBROUTINE init_ndep_data_monthly (YY,MM)  !sf_add

   !----------------------
   ! DESCTIPTION:
   ! open ndep netcdf file from DEF_dir_runtime, read latitude and longitude info.
   ! Initialize ndep data read in.

   USE MOD_TimeManager
   USE MOD_Namelist
   USE MOD_Grid
   USE MOD_NetCDFSerial
   USE MOD_LandPatch
   IMPLICIT NONE

   integer, intent(in) :: YY,MM  !sf_add

   ! Local Variables
   real(r8), allocatable :: lat(:), lon(:)

      file_ndep = trim(DEF_dir_runtime) // '/ndep/fndep_colm_monthly.nc' !sf_add

      CALL ncio_read_bcast_serial (file_ndep, 'lat', lat)
      CALL ncio_read_bcast_serial (file_ndep, 'lon', lon)

      CALL grid_ndep%define_by_center (lat, lon)

      CALL mg2p_ndep%build_arealweighted (grid_ndep, landpatch)

      IF (allocated(lon)) deallocate(lon)
      IF (allocated(lat)) deallocate(lat)

      CALL update_ndep_data_monthly (YY, MM ,iswrite = .true.) !sf_add

   END SUBROUTINE init_ndep_data_monthly

   ! ----------
   SUBROUTINE update_ndep_data_annually (YY, iswrite)
! ===========================================================
!
! !DESCRIPTION:
! Read in the Nitrogen deposition data from CLM5.
!
! !REFERENCE:
! Galloway, J.N., et al. 2004. Nitrogen cycles: past, present, and future. Biogeochem. 70:153-226.
!
! !ORIGINAL:
! Created by Xingjie Lu and Shupeng Zhang, 2022
! ===========================================================

   USE MOD_SPMD_Task
   USE MOD_Namelist, only : DEF_USE_PN
   USE MOD_DataType
   USE MOD_NetCDFBlock
   USE MOD_LandPatch
   USE MOD_Vars_TimeInvariants
   USE MOD_RangeCheck
   IMPLICIT NONE

   integer, intent(in) :: YY
   logical, intent(in) :: iswrite

   ! Local Variables
   type(block_data_real8_2d) :: f_xy_ndep
   integer :: itime, npatch, m

      itime = max(min(YY,2006),1849) - 1848

      IF (p_is_io) THEN
         CALL allocate_block_data  (grid_ndep, f_xy_ndep)
         CALL ncio_read_block_time (file_ndep, 'NDEP_year', grid_ndep, itime, f_xy_ndep)
      ENDIF

      CALL mg2p_ndep%grid2pset (f_xy_ndep, ndep)

      IF (p_is_worker .and. iswrite) THEN
         IF (numpatch > 0) THEN
            DO npatch = 1, numpatch
               m = patchclass(npatch)
               IF(m == 0)THEN
                  ndep_to_sminn(npatch) = 0.
               ELSE
                  IF(DEF_USE_PN)THEN
                     ndep_to_sminn(npatch)  = ndep(npatch) / 3600. / 365. / 24. * 5
                  ELSE
                     ndep_to_sminn(npatch)  = ndep(npatch) / 3600. / 365. / 24.
                  ENDIF
               ENDIF
            ENDDO

         ENDIF
      ENDIF

#ifdef RangeCheck
      CALL check_vector_data ('ndep', ndep)
#endif

   END SUBROUTINE update_ndep_data_annually

   ! ----------
   SUBROUTINE update_ndep_data_monthly (YY, MM, iswrite) !sf_add
! ===========================================================
!
! !DESCRIPTION:
! Read in the Nitrogen deposition data from CLM5.
!
! !REFERENCE:
! Galloway, J.N., et al. 2004. Nitrogen cycles: past, present, and future. Biogeochem. 70:153-226.
!
! !ORIGINAL:
! Created by Xingjie Lu and Shupeng Zhang, 2022
! ===========================================================

   USE MOD_SPMD_Task
   USE MOD_Namelist, only : DEF_USE_PN
   USE MOD_DataType
   USE MOD_NetCDFBlock
   USE MOD_LandPatch
   USE MOD_Vars_TimeInvariants
   USE MOD_RangeCheck
   IMPLICIT NONE

   integer, intent(in) :: YY,MM  ! sf_add
   logical, intent(in) :: iswrite

   ! Local Variables
   type(block_data_real8_2d) :: f_xy_ndep
   integer :: itime, npatch, m

      itime = (max(min(YY,2006),1849) - 1849)*12 + MM ! sf_add
!      print*,"YY=",YY        ! sf_add
!      print*,"MM=",MM        ! sf_add
!      print*,"itime=",itime  ! sf_add

      IF (p_is_io) THEN
         CALL allocate_block_data  (grid_ndep, f_xy_ndep)
         CALL ncio_read_block_time (file_ndep, 'NDEP_month', grid_ndep, itime, f_xy_ndep) ! sf_add
      ENDIF

      CALL mg2p_ndep%grid2pset (f_xy_ndep, ndep)

      IF (p_is_worker .and. iswrite) THEN
         IF (numpatch > 0) THEN
            DO npatch = 1, numpatch
               m = patchclass(npatch)
               IF(m == 0)THEN
                  ndep_to_sminn(npatch) = 0.
               ELSE
                  IF(DEF_USE_PN)THEN
                     ndep_to_sminn(npatch)  = ndep(npatch) / 3600. / 365. / 24. * 5
                  ELSE
                     ndep_to_sminn(npatch)  = ndep(npatch) / 3600. / 365. / 24.
                  ENDIF
               ENDIF
            ENDDO

         ENDIF
      ENDIF

#ifdef RangeCheck
      CALL check_vector_data ('ndep', ndep)
#endif

   END SUBROUTINE update_ndep_data_monthly

END MODULE MOD_NdepData
#endif
