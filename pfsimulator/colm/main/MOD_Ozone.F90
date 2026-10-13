#include <define.h>

Module MOD_Ozone

 !-----------------------------------------------------------------------
 ! !DESCRIPTION:
 ! This module hold the plant physiological response to the ozone, including vcmax response and stomata response.
 ! Ozone concentration can be either readin through Mod_OzoneData module or set to constant.
 !
 ! !ORIGINAL:
 ! The Community Land Model version 5.0 (CLM5.0)
 !
 ! !REVISION:
 ! Xingjie Lu 2022, revised the CLM5 code to be compatible with CoLM code structure.



   USE MOD_Precision
   USE MOD_Const_Physical, only: rgas
   USE MOD_Const_PFT, only: isevg, leaf_long, woody
   !USE MOD_Grid
   !USE MOD_DataType
   !USE MOD_SpatialMapping
   USE MOD_Vars_1DForcing, only: forc_ozone
   USE MOD_Namelist, only: DEF_USE_OZONEDATA
   IMPLICIT NONE

   character(len=256) :: file_ozone

   !type(grid_type) :: grid_ozone

   !type(block_data_real8_2d) :: f_ozone

   !type(spatial_mapping_type) :: mg2p_ozone

   SAVE

   PUBLIC :: CalcOzoneStress
   !PUBLIC :: init_ozone_data
   !PUBLIC :: update_ozone_data

CONTAINS

   SUBROUTINE CalcOzoneStress (o3coefv,o3coefg, forc_ozone, forc_psrf, th, ram, &
                              rs, rb, lai, lai_old, ivt, o3uptake, deltim)
   !-------------------------------------------------
   ! DESCRIPTION:
   ! Calculate Ozone Stress on both vcmax and stomata conductance.
   !
   ! convert o3 from mol/mol to nmol m^-3
   real(r8), intent(out)   :: o3coefv
   real(r8), intent(out)   :: o3coefg
   real(r8), intent(inout) :: forc_ozone
   real(r8), intent(in)    :: forc_psrf
   real(r8), intent(in)    :: th
   real(r8), intent(in)    :: ram
   real(r8), intent(in)    :: rs
   real(r8), intent(in)    :: rb
   real(r8), intent(in)    :: lai
   real(r8), intent(in)    :: lai_old
   integer , intent(in)    :: ivt
   real(r8), intent(inout) :: o3uptake
   real(r8), intent(in)    :: deltim

   real(r8) :: o3concnmolm3   ! o3 concentration (nmol/m^3)
   real(r8) :: o3flux         ! instantaneous o3 flux (nmol m^-2 s^-1)
   real(r8) :: o3fluxcrit     ! instantaneous o3 flux beyond threshold (nmol m^-2 s^-1)
   real(r8) :: o3fluxperdt    ! o3 flux per timestep (mmol m^-2)
   real(r8) :: heal           ! o3uptake healing rate based on % of new leaves growing (mmol m^-2)
   real(r8) :: leafturn       ! leaf turnover time / mortality rate (per hour)
   real(r8) :: decay          ! o3uptake decay rate based on leaf lifetime (mmol m^-2)
   real(r8) :: photoInt       ! intercept for photosynthesis
   real(r8) :: photoSlope     ! slope for photosynthesis
   real(r8) :: condInt        ! intercept for conductance
   real(r8) :: condSlope      ! slope for conductance

   real(r8), parameter :: ko3 = 1.51_r8  !F. Li

   ! LAI threshold for LAIs that asymptote and don't reach 0
   real(r8), parameter :: lai_thresh = 0.5_r8

   ! threshold below which o3flux is set to 0 (nmol m^-2 s^-1)
   real(r8), parameter :: o3_flux_threshold = 0.5_r8  !F. Li

   ! o3 intercepts and slopes for photosynthesis
   real(r8), parameter :: needleleafPhotoInt   = 0.8390_r8  ! units = unitless
   real(r8), parameter :: needleleafPhotoSlope = 0._r8      ! units = per mmol m^-2
   real(r8), parameter :: broadleafPhotoInt    = 0.8752_r8  ! units = unitless
   real(r8), parameter :: broadleafPhotoSlope  = 0._r8      ! units = per mmol m^-2
   real(r8), parameter :: nonwoodyPhotoInt     = 0.8021_r8  ! units = unitless
   real(r8), parameter :: nonwoodyPhotoSlope   = -0.0009_r8 ! units = per mmol m^-2

   ! o3 intercepts and slopes for conductance
   real(r8), parameter :: needleleafCondInt    = 0.7823_r8  ! units = unitless
   real(r8), parameter :: needleleafCondSlope  = 0.0048_r8  ! units = per mmol m^-2
   real(r8), parameter :: broadleafCondInt     = 0.9125_r8  ! units = unitless
   real(r8), parameter :: broadleafCondSlope   = 0._r8      ! units = per mmol m^-2
   real(r8), parameter :: nonwoodyCondInt      = 0.7511_r8  ! units = unitless
   real(r8), parameter :: nonwoodyCondSlope    = 0._r8      ! units = per mmol m^-2

      IF(.not. DEF_USE_OZONEDATA)THEN
         forc_ozone = 100._r8 * 1.e-9_r8 ! ozone partial pressure [mol/mol]
      ENDIF

      o3concnmolm3 = forc_ozone * 1.e9_r8 * (forc_psrf/(th*rgas*0.001_r8 ))

      ! calculate instantaneous flux
      o3flux = o3concnmolm3/ (ko3*rs+ rb + ram)

      ! apply o3 flux threshold
      IF (o3flux < o3_flux_threshold) THEN
         o3fluxcrit = 0._r8
      ELSE
         o3fluxcrit = o3flux - o3_flux_threshold
      ENDIF

      ! calculate o3 flux per timestep
      o3fluxperdt = o3fluxcrit * deltim * 0.000001_r8

      IF (lai > lai_thresh) THEN
         ! checking IF new leaf area was added
         IF (lai - lai_old > 0) THEN
            ! minimizing o3 damage to new leaves
            heal = max(0._r8,(((lai-lai_old)/lai)*o3fluxperdt))
         ELSE
            heal = 0._r8
         ENDIF

         IF (isevg(ivt)) THEN
            leafturn = 1._r8/(leaf_long(ivt)*365._r8*24._r8)
         ELSE
            leafturn = 0._r8
         ENDIF

         ! o3 uptake decay based on leaf lifetime for evergreen plants
         decay = o3uptake * leafturn * deltim/3600._r8
         !cumulative uptake (mmol m^-2)
         o3uptake = max(0._r8, o3uptake + o3fluxperdt - decay - heal)

      ELSE
         o3uptake = 0._r8
      ENDIF

      IF (o3uptake == 0._r8) THEN
         ! No o3 damage IF no o3 uptake
         o3coefv = 1._r8
         o3coefg = 1._r8
      ELSE
         ! Determine parameter values for this pft
         ! TODO(wjs, 2014-10-01) Once these parameters are moved into the params file,     this
         ! logic can be removed.
         ! add GPAM ozone impact on crop and change logic by F. Li
         IF (ivt>16)THEN
            o3coefv = max(0._r8, min(1._r8, 0.883_r8 - 0.058 * log10(o3uptake)))
            o3coefg = max(0._r8, min(1._r8, 0.951_r8 - 0.109 * tanh(o3uptake)))
         ELSE
            IF (ivt>3) THEN
               IF (woody(ivt)==0) THEN
                  photoInt   = nonwoodyPhotoInt
                  photoSlope = nonwoodyPhotoSlope
                  condInt    = nonwoodyCondInt
                  condSlope  = nonwoodyCondSlope
               ELSE
                  photoInt   = broadleafPhotoInt
                  photoSlope = broadleafPhotoSlope
                  condInt    = broadleafCondInt
                  condSlope  = broadleafCondSlope
               ENDIF
            ELSE
               photoInt   = needleleafPhotoInt
               photoSlope = needleleafPhotoSlope
               condInt    = needleleafCondInt
               condSlope  = needleleafCondSlope
            ENDIF

            ! Apply parameter values to compute o3 coefficients
            o3coefv = max(0._r8, min(1._r8, photoInt + photoSlope * o3uptake))
            o3coefg = max(0._r8, min(1._r8, condInt  + condSlope  * o3uptake))
         ENDIF
      ENDIF

   END SUBROUTINE CalcOzoneStress

END MODULE MOD_Ozone
