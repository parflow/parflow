#include <define.h>

MODULE MOD_LakeDepthReadin

!------------------------------------------------------------------------------------------
! DESCRIPTION:
! Read in lakedepth and assign lake thickness of each layer.
!
! Original author: Yongjiu Dai, 03/2018
!------------------------------------------------------------------------------------------

   USE MOD_Precision
   IMPLICIT NONE
   SAVE

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: lakedepth_readin


CONTAINS


   SUBROUTINE lakedepth_readin (dir_landdata, lc_year, numpatch)

   USE MOD_Precision
   USE MOD_Vars_Global, only : nl_lake
   USE MOD_SPMD_Task
   !USE MOD_LandPatch
   !USE MOD_NetCDFVector
   USE MOD_Vars_TimeInvariants, only : lakedepth, dz_lake
#ifdef SinglePoint
   USE MOD_SingleSrfdata
#endif

   IMPLICIT NONE

   integer, intent(in) :: lc_year, numpatch    ! which year of land cover data used
   character(len=256), intent(in) :: dir_landdata

   ! Local Variables
   character(len=256) :: lndname, cyear
   real(r8) :: depthratio                  ! ratio of lake depth to standard deep lake depth
   integer  :: ipatch

   ! -----------------------------------------------------------
   ! For global simulations with 10 body layers,
   ! the default (50 m lake) body layer thicknesses are given by :
   ! The node depths zlak located at the center of each layer
   real(r8), dimension(10) :: dzlak
   real(r8), dimension(10) :: zlak

      dzlak = (/0.1, 1., 2., 3., 4., 5., 7., 7., 10.45, 10.45/)  ! m
      zlak  = (/0.05, 0.6, 2.1, 4.6, 8.1, 12.6, 18.6, 25.6, 34.325, 44.775/)

      ! For site simulations with 25 layers, the default thicknesses are (m):
      ! real(r8), dimension(25) :: dzlak
      ! dzlak = (/0.1,                         & ! 0.1 for layer 1;
      !           0.25, 0.25, 0.25, 0.25,      & ! 0.25 for layers 2-5;
      !           0.50, 0.50, 0.50, 0.50,      & ! 0.5 for layers 6-9;
      !           0.75, 0.75, 0.75, 0.75,      & ! 0.75 for layers 10-13;
      !           2.00, 2.00,                  & ! 2 for layers 14-15;
      !           2.50, 2.50,                  & ! 2.5 for layers 16-17;
      !           3.50, 3.50, 3.50, 3.50,      & ! 2.5 for layers 16-17;
      !           5.225, 5.225, 5.225, 5.225/)   ! 5.225 for layers 22-25.
      !
      ! For lakes with depth d /= 50 m and d >= 1 m,
      !                       the top layer is kept at 10 cm and
      !                       the other 9 layer thicknesses are adjusted to maintain fixed proportions.
      !
      ! For lakes with d < 1 m, all layers have equal thickness.
      ! -----------------------------------------------------------

      ! Read lakedepth
#ifdef SinglePoint
      lakedepth(:) = SITE_lakedepth
#else
      write(cyear,'(i4.4)') lc_year
      lndname = trim(dir_landdata)//'/lakedepth/'//trim(cyear)//'/lakedepth_patches.nc'
      !CALL ncio_read_vector (lndname, 'lakedepth_patches', landpatch, lakedepth)
#endif

      lakedepth(:) = sum(dzlak(1:nl_lake))

      ! Define lake levels
      IF (p_is_worker) THEN

         DO ipatch = 1, numpatch

            ! testing 14/05/2021, Zhang
            IF(lakedepth(ipatch) < 0.1) lakedepth(ipatch) = 0.1

            IF(lakedepth(ipatch) > 1. .and. lakedepth(ipatch) < 1000.)THEN
               depthratio = lakedepth(ipatch) / sum(dzlak(1:nl_lake))
               dz_lake(1,ipatch) = dzlak(1)
               dz_lake(2:nl_lake-1,ipatch) = dzlak(2:nl_lake-1)*depthratio
               dz_lake(nl_lake,ipatch) = dzlak(nl_lake)*depthratio - (dz_lake(1,ipatch) - dzlak(1)*depthratio)
            ELSEIF(lakedepth(ipatch) > 0. .and. lakedepth(ipatch) <= 1.)THEN
               dz_lake(:,ipatch) = lakedepth(ipatch) / nl_lake
            ELSE   ! non land water bodies or missing value of the lake depth
               lakedepth(ipatch) = sum(dzlak(1:nl_lake))
               dz_lake(1:nl_lake,ipatch) = dzlak(1:nl_lake)
            ENDIF

         ENDDO

      ENDIF

   END SUBROUTINE lakedepth_readin

END MODULE MOD_LakeDepthReadin
