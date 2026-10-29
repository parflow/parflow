#include <define.h>
MODULE MOD_LeafInterception
! -----------------------------------------------------------------
! !DESCRIPTION:
! For calculating vegetation canopy preciptation interception.
!
! This MODULE is the coupler for the colm and CaMa-Flood model.

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------
   !* :SUBROUTINE:"LEAF_interception_CoLM2014"   : interception and drainage of precipitation schemes based on colm2014 version
   !* :SUBROUTINE:"LEAF_interception_CoLM202x"   : interception and drainage of precipitation schemes besed on new colm version (under development)
   !* :SUBROUTINE:"LEAF_interception_CLM4"       : interception and drainage of precipitation schemes modified from CLM4
   !* :SUBROUTINE:"LEAF_interception_CLM5"       : interception and drainage of precipitation schemes modified from CLM5
   !* :SUBROUTINE:"LEAF_interception_NOAHMP"     : interception and drainage of precipitation schemes modified from Noah-MP
   !* :SUBROUTINE:"LEAF_interception_MATSIRO"    : interception and drainage of precipitation schemes modified from MATSIRO 2021 version
   !* :SUBROUTINE:"LEAF_interception_VIC"        : interception and drainage of precipitation schemes modified from VIC
   !* :SUBROUTINE:"LEAF_interception_JULES"      : interception and drainage of precipitation schemes modified from JULES
   !* :SUBROUTINE:"LEAF_interception_pftwrap"    : wapper for pft land use classification
   !* :SUBROUTINE:"LEAF_interception_pcwrap"     : wapper for pc land use classification

!REVISION HISTORY:
!----------------
   ! 2023.07     Hua Yuan: remove wrapper PC by using PFT leaf interception
   ! 2023.06     Shupeng Zhang @ SYSU
   ! 2023.02.23  Zhongwang Wei @ SYSU
   ! 2021.12.12  Zhongwang Wei @ SYSU
   ! 2020.10.21  Zhongwang Wei @ SYSU
   ! 2019.06     Hua Yuan: 1) add wrapper for PFT and PC, and 2) remove sigf by using lai+sai
   ! 2014.04     Yongjiu Dai
   ! 2002.08.31  Yongjiu Dai
   USE MOD_Precision
   USE MOD_Const_Physical, only: tfrz, denh2o, denice
   USE MOD_Namelist, only : DEF_Interception_scheme, DEF_USE_IRRIGATION
#ifdef CROP
   USE MOD_Irrigation, only: CalIrrigationApplicationFluxes
#endif

   IMPLICIT NONE

   real(r8), parameter ::  CICE        = 2.094E06  !specific heat capacity of ice (j/m3/k)
   real(r8), parameter ::  bp          = 20.
   real(r8), parameter ::  HFUS        = 0.3336E06 !latent heat of fusion (j/kg)
   real(r8), parameter ::  CWAT        = 4.188E06  !specific heat capacity of water (j/m3/k)
   real(r8), parameter ::  pcoefs(2,2) = reshape((/20.0_r8, 0.206e-8_r8, 0.0001_r8, 0.9999_r8/), (/2,2/))

   !----------------------- Dummy argument --------------------------------
   real(r8) :: satcap                     ! maximum allowed water on canopy [mm]
   real(r8) :: satcap_rain                ! maximum allowed rain on canopy [mm]
   real(r8) :: satcap_snow                ! maximum allowed snow on canopy [mm]
   real(r8) :: lsai                       ! sum of leaf area index and stem area index [-]
   real(r8) :: chiv                       ! leaf angle distribution factor
   real(r8) :: ppc                        ! convective precipitation in time-step [mm]
   real(r8) :: ppl                        ! large-scale precipitation in time-step [mm]
   real(r8) :: p0                         ! precipitation in time-step [mm]
   real(r8) :: fpi                        ! coefficient of interception
   real(r8) :: fpi_rain                   ! coefficient of interception of rain
   real(r8) :: fpi_snow                   ! coefficient of interception of snow
   real(r8) :: alpha_rain                 ! coefficient of interception of rain
   real(r8) :: alpha_snow                 ! coefficient of interception of snow
   real(r8) :: pinf                       ! interception of precipitation in time step [mm]
   real(r8) :: tti_rain                   ! direct rain throughfall in time step [mm]
   real(r8) :: tti_snow                   ! direct snow throughfall in time step [mm]
   real(r8) :: tex_rain                   ! canopy rain drainage in time step [mm]
   real(r8) :: tex_snow                   ! canopy snow drainage in time step [mm]
   real(r8) :: vegt                       ! sigf*lsai
   real(r8) :: xs                         ! proportion of the grid area where the intercepted rainfall
                                          ! plus the preexisting canopy water storage
   real(r8)  :: unl_snow_temp,U10,unl_snow_wind,unl_snow
   real(r8)  :: ap, cp, aa1, bb1, exrain, arg, w
   real(r8)  :: thru_rain, thru_snow
   real(r8)  :: xsc_rain, xsc_snow

   real(r8)  :: fvegc          ! vegetation fraction
   real(r8)  :: FT             ! the temperature factor for snow unloading
   real(r8)  :: FV             ! the wind factor for snow unloading
   real(r8)  :: ICEDRIP        ! snow unloading

   real(r8)  :: ldew_smelt
   real(r8)  :: ldew_frzc
   real(r8)  :: FP
   real(r8)  :: int_rain
   real(r8)  :: int_snow

   real(r8) :: qflx_irrig_drip
   real(r8) :: qflx_irrig_sprinkler
   real(r8) :: qflx_irrig_flood
   real(r8) :: qflx_irrig_paddy

CONTAINS

   SUBROUTINE LEAF_interception_CoLM2014 (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,&
                                          prc_rain,prc_snow,prl_rain,prl_snow,&
                                          ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Calculation of  interception and drainage of precipitation
   ! the treatment are based on Sellers et al. (1996)

!Original Author:
!-------------------
   !canopy interception scheme modified by Yongjiu Dai based on Sellers et al. (1996)

!References:
!-------------------
   !---Dai, Y., Zeng, X., Dickinson, R.E., Baker, I., Bonan, G.B., BosiloVICh, M.G., Denning, A.S.,
   !   Dirmeyer, P.A., Houser, P.R., Niu, G. and Oleson, K.W., 2003.
   !   The common land model. Bulletin of the American Meteorological Society, 84(8), pp.1013-1024.

   !---Lawrence, D.M., Thornton, P.E., Oleson, K.W. and Bonan, G.B., 2007.
   !   The partitioning of evapotranspiration into transpiration, soil evaporation,
   !   and canopy evaporation in a GCM: Impacts on land–atmosphere interaction. Journal of Hydrometeorology, 8(4), pp.862-880.

   !---Oleson, K., Dai, Y., Bonan, B., BosiloVIChm, M., Dickinson, R., Dirmeyer, P., Hoffman,
   !   F., Houser, P., Levis, S., Niu, G.Y. and Thornton, P., 2004.
   !   Technical description of the community land model (CLM).

   !---Sellers, P.J., Randall, D.A., Collatz, G.J., Berry, J.A., Field, C.B., Dazlich, D.A., Zhang, C.,
   !   Collelo, G.D. and Bounoua, L., 1996. A revised land surface parameterization (SiB2) for atmospheric GCMs.
   !   Part I: Model formulation. Journal of climate, 9(4), pp.676-705.

   !---Sellers, P.J., Tucker, C.J., Collatz, G.J., Los, S.O., Justice, C.O., Dazlich, D.A. and Randall, D.A., 1996.
   !   A revised land surface parameterization (SiB2) for atmospheric GCMs. Part II:
   !   The generation of global fields of terrestrial biophysical parameters from satellite data.
   !   Journal of climate, 9(4), pp.706-737.


!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   !---2023.02.21  Zhongwang Wei @ SYSU : Snow and rain interception
   !---2021.12.08  Zhongwang Wei @ SYSU
   !---2019.06     Hua Yuan: remove sigf and USE lai+sai for judgement.
   !---2014.04     Yongjiu Dai
   !---2002.08.31  Yongjiu Dai
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in) :: deltim         ! seconds in a time step [second]
   real(r8), intent(in) :: dewmx          ! maximum dew [mm]
   real(r8), intent(in) :: forc_us        ! wind speed
   real(r8), intent(in) :: forc_vs        ! wind speed
   real(r8), intent(in) :: chil           ! leaf angle distribution factor
   real(r8), intent(in) :: prc_rain       ! convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow       ! convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain       ! large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow       ! large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf           ! fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai            ! leaf area index [-]
   real(r8), intent(in) :: sai            ! stem area index [-]
   real(r8), intent(in) :: tair           ! air temperature [K]
   real(r8), intent(in) :: tleaf          ! sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew        ! depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   ! depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   ! depth of water on foliage [mm]
   real(r8), intent(in)    :: z0m         ! roughness length
   real(r8), intent(in)    :: hu          ! forcing height of U

   real(r8), intent(out) :: pg_rain       ! rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow       ! snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr         ! interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain    ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow    ! snowfall interception (mm h2o/s)

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         satcap = dewmx*vegt

         p0  = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+qflx_irrig_sprinkler)*deltim

         w = ldew+p0
         IF (tleaf > tfrz) THEN
            xsc_rain = max(0., ldew-satcap)
            xsc_snow = 0.
         ELSE
            xsc_rain = 0.
            xsc_snow = max(0., ldew-satcap)
         ENDIF
         ldew = ldew - (xsc_rain + xsc_snow)

         ap = pcoefs(2,1)
         cp = pcoefs(2,2)

         IF (p0 > 1.e-8) THEN
            ap = ppc/p0 * pcoefs(1,1) + ppl/p0 * pcoefs(2,1)
            cp = ppc/p0 * pcoefs(1,2) + ppl/p0 * pcoefs(2,2)

            !----------------------------------------------------------------------
            !      proportional saturated area (xs) and leaf drainage(tex)
            !-----------------------------------------------------------------------
            chiv = chil
            IF ( abs(chiv) .le. 0.01 ) chiv = 0.01
            aa1 = 0.5 - 0.633 * chiv - 0.33 * chiv * chiv
            bb1 = 0.877 * ( 1. - 2. * aa1 )
            exrain = aa1 + bb1

            ! coefficient of interception
            ! set fraction of potential interception to max 0.25 (Lawrence et al. 2007)
            ! assume alpha_rain = alpha_snow
            alpha_rain = 0.25
            fpi = alpha_rain * ( 1.-exp(-exrain*lsai) )
            tti_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fpi )
            tti_snow = (prc_snow+prl_snow)*deltim * ( 1.-fpi )

            xs = 1.
            IF (p0*fpi>1.e-9) THEN
               arg = (satcap-ldew)/(p0*fpi*ap) - cp/ap
               IF (arg>1.e-9) THEN
                  xs = -1./bp * log( arg )
                  xs = min( xs, 1. )
                  xs = max( xs, 0. )
               ENDIF
            ENDIF

            ! assume no fall down of the intercepted snowfall in a time step
            ! drainage
            tex_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * fpi * (ap/bp*(1.-exp(-bp*xs))+cp*xs) &
                     - (satcap-ldew) * xs
            tex_rain = max( tex_rain, 0. )
            tex_snow = 0.

#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif

         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim

#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim, satcap
            CALL abort
         ENDIF
#endif

      ELSE
         ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.

      ENDIF

   END SUBROUTINE LEAF_interception_CoLM2014

   SUBROUTINE LEAF_interception_CoLM202x (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,&
                                          prc_rain,prc_snow,prl_rain,prl_snow,&
                                          ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,&
                                          qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Calculation of interception and drainage of precipitation (under development)
   ! the scheme developed by Zhongwang wei @ SYSU (not finished yet)

!Original Author:
!-------------------
   !---Zhongwang Wei @ SYSU

!References:
!-------------------
   !---Zhong, F., Jiang, S., van Dijk, A.I., Ren, L., Schellekens, J. and Miralles, D.G., 2022.
   !   Revisiting large-scale interception patterns constrained by a synthesis of global experimental
   !   data. Hydrology and Earth System Sciences, 26(21), pp.5647-5667.
   !---

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   !---2023.04.30  Zhongwang Wei @ SYSU : Snow and rain interception
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in) :: deltim         ! seconds in a time step [second]
   real(r8), intent(in) :: dewmx          ! maximum dew [mm]
   real(r8), intent(in) :: forc_us        ! wind speed
   real(r8), intent(in) :: forc_vs        ! wind speed
   real(r8), intent(in) :: chil           ! leaf angle distribution factor
   real(r8), intent(in) :: prc_rain       ! convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow       ! convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain       ! large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow       ! large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf           ! fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai            ! leaf area index [-]
   real(r8), intent(in) :: sai            ! stem area index [-]
   real(r8), intent(in) :: tair           ! air temperature [K]
   real(r8), intent(in) :: tleaf          ! sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew        ! depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   ! depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   ! depth of water on foliage [mm]
   real(r8), intent(in)    :: z0m         ! roughness length
   real(r8), intent(in)    :: hu          ! forcing height of U

   real(r8), intent(out) :: pg_rain       ! rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow       ! snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr         ! interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain    ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow    ! snowfall interception (mm h2o/s)

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         satcap = dewmx*vegt

         p0  = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+qflx_irrig_sprinkler)*deltim

         w = ldew+p0

         IF (tleaf > tfrz) THEN
            xsc_rain = max(0., ldew-satcap)
            xsc_snow = 0.
         ELSE
            xsc_rain = 0.
            xsc_snow = max(0., ldew-satcap)
         ENDIF
         ldew = ldew - (xsc_rain + xsc_snow)

         ap = pcoefs(2,1)
         cp = pcoefs(2,2)

         IF (p0 > 1.e-8) THEN
            ap = ppc/p0 * pcoefs(1,1) + ppl/p0 * pcoefs(2,1)
            cp = ppc/p0 * pcoefs(1,2) + ppl/p0 * pcoefs(2,2)
            !----------------------------------------------------------------------
            !      proportional saturated area (xs) and leaf drainage(tex)
            !-----------------------------------------------------------------------
            chiv = chil
            IF ( abs(chiv) .le. 0.01 ) chiv = 0.01
            aa1 = 0.5 - 0.633 * chiv - 0.33 * chiv * chiv
            bb1 = 0.877 * ( 1. - 2. * aa1 )
            exrain = aa1 + bb1

            ! coefficient of interception
            ! set fraction of potential interception to max 0.25 (Lawrence et al. 2007)
            alpha_rain = 0.25
            fpi = alpha_rain * ( 1.-exp(-exrain*lsai) )
            tti_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fpi )
            tti_snow = (prc_snow+prl_snow)*deltim * ( 1.-fpi )

            xs = 1.
            IF (p0*fpi>1.e-9) THEN
               arg = (satcap-ldew)/(p0*fpi*ap) - cp/ap
               IF (arg>1.e-9) THEN
                  xs = -1./bp * log( arg )
                  xs = min( xs, 1. )
                  xs = max( xs, 0. )
               ENDIF
            ENDIF

            ! assume no fall down of the intercepted snowfall in a time step drainage
            tex_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * fpi * (ap/bp*(1.-exp(-bp*xs))+cp*xs) - (satcap-ldew) * xs

            !       tex_rain = (prc_rain+prl_rain)*deltim * fpi * (ap/bp*(1.-exp(-bp*xs))+cp*xs) &
            !                - (satcap-ldew) * xs
            tex_rain = max( tex_rain, 0. )
            tex_snow = 0.

#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif

         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim


#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim, satcap
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF
   END SUBROUTINE LEAF_interception_CoLM202x

   SUBROUTINE LEAF_interception_CLM4 (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,&
                                       prc_rain,prc_snow,prl_rain,prl_snow,&
                                       ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                       pg_snow,qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Interception and drainage of precipitation
   ! the treatment are modified from CLM4.5

!Original Author:
!-------------------
   !Lawrence, D.M.

!References:
!-------------------
   !---Lawrence, D.M., Thornton, P.E., Oleson, K.W. and Bonan, G.B., 2007.
   !   The partitioning of evapotranspiration into transpiration, soil evaporation,
   !   and canopy evaporation in a GCM: Impacts on land–atmosphere interaction. Journal of Hydrometeorology, 8(4), pp.862-880.

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   ! 2023.02.21  Zhongwang Wei @ SYSU : Snow and rain interception
   ! 2021.12.08  Zhongwang Wei @ SYSU
   ! 2014.04     Yongjiu Dai
   ! 2002.08.31  Yongjiu Dai
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in) :: deltim    !seconds in a time step [second]
   real(r8), intent(in) :: dewmx     !maximum dew [mm]
   real(r8), intent(in) :: forc_us   !wind speed
   real(r8), intent(in) :: forc_vs   !wind speed
   real(r8), intent(in) :: chil      !leaf angle distribution factor
   real(r8), intent(in) :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai       !leaf area index [-]
   real(r8), intent(in) :: sai       !stem area index [-]
   real(r8), intent(in) :: tair     !air temperature [K]
   real(r8), intent(in) :: tleaf     !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of water on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of U

   real(r8), intent(out) :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow ! snowfall interception (mm h2o/s)

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         satcap = dewmx*vegt

         p0  = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+qflx_irrig_sprinkler)*deltim

         w = ldew+p0

         IF (tleaf > tfrz) THEN
            xsc_rain = max(0., ldew-satcap)
            xsc_snow = 0.
         ELSE
            xsc_rain = 0.
            xsc_snow = max(0., ldew-satcap)
         ENDIF

         ldew = ldew - (xsc_rain + xsc_snow)

         IF (p0 > 1.e-8) THEN
            exrain =0.5
            ! coefficient of interception
            ! set fraction of potential interception to max 0.25 (Lawrence et al. 2007)
            alpha_rain = 0.25
            fpi = alpha_rain * ( 1.-exp(-exrain*lsai) )
            tti_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fpi )
            tti_snow = (prc_snow+prl_snow)*deltim * ( 1.-fpi )

            ! assume no fall down of the intercepted snowfall in a time step
            ! drainage
            tex_rain = (prc_rain+prl_rain)*deltim * fpi + ldew - satcap
            tex_rain = max(tex_rain, 0. )
            tex_snow = 0.

#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif

         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------
         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim


#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim, satcap
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF

   END SUBROUTINE LEAF_interception_CLM4

   SUBROUTINE LEAF_interception_CLM5 (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf,&
                                    prc_rain,prc_snow,prl_rain,prl_snow,&
                                    ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,&
                                    qintr,qintr_rain,qintr_snow)

!DESCRIPTION
!===========
   ! Interception and drainage of precipitation
   ! the treatment are modified from CLM5.0

!Original Author:
!-------------------
   !---Lawrence, D.M.

!References:
!-------------------
   !---Lawrence, D.M., Thornton, P.E., Oleson, K.W. and Bonan, G.B., 2007.
   !   The partitioning of evapotranspiration into transpiration, soil evaporation,
   !   and canopy evaporation in a GCM: Impacts on land–atmosphere interaction. Journal of Hydrometeorology, 8(4), pp.862-880.
   !---Lawrence, D.M., Fisher, R.A., Koven, C.D., Oleson, K.W., Swenson, S.C., Bonan, G., Collier, N., Ghimire, B.,
   !   van Kampenhout, L., Kennedy, D. and Kluzek, E., 2019. The Community Land Model version 5:
   !   Description of new features, benchmarking, and impact of forcing uncertainty.
   !   Journal of Advances in Modeling Earth Systems, 11(12), pp.4245-4287.
   !---Fan, Y., Meijide, A., Lawrence, D.M., Roupsard, O., Carlson, K.M., Chen, H.Y.,
   !   Röll, A., Niu, F. and Knohl, A., 2019. Reconciling canopy interception parameterization
   !   and rainfall forcing frequency in the Community Land Model for simulating evapotranspiration
   !   of rainforests and oil palm plantations in Indonesia. Journal of Advances in Modeling Earth Systems, 11(3), pp.732-751.


!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   ! 2023.02.21  Zhongwang Wei @ SYSU
   ! 2021.12.08  Zhongwang Wei @ SYSU
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in) :: deltim    !seconds in a time step [second]
   real(r8), intent(in) :: dewmx     !maximum dew [mm]
   real(r8), intent(in) :: forc_us   !wind speed
   real(r8), intent(in) :: forc_vs   !wind speed
   real(r8), intent(in) :: chil      !leaf angle distribution factor
   real(r8), intent(in) :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai       !leaf area index [-]
   real(r8), intent(in) :: sai       !stem area index [-]
   real(r8), intent(in) :: tair      !air temperature [K]
   real(r8), intent(in) :: tleaf     !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew           !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain      !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_snow      !depth of water on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of U

   real(r8), intent(out) :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow ! snowfall interception (mm h2o/s)

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         p0  = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+qflx_irrig_sprinkler)*deltim
         w = ldew+p0
         satcap_rain = dewmx*vegt
         satcap_snow = satcap_rain*60.0

         xsc_rain      = max(0., ldew_rain-satcap_rain)
         xsc_snow      = max(0., ldew_snow-satcap_snow)

         ldew_rain     = ldew_rain-xsc_rain
         ldew_snow     = ldew_snow-xsc_snow

         !unload due to wind and temperature
         !U10= sqrt(forc_us*forc_us+forc_vs*forc_vs)*log(10.0/z0m)/log(hu/z0m)
         IF(ldew_snow > 1.e-8) THEN
            U10           =  sqrt(forc_us*forc_us+forc_vs*forc_vs)
            unl_snow_temp =  ldew_snow*(tleaf-270.0)/(1.87*1.e5)
            unl_snow_temp =  max(unl_snow_temp,0.0)
            unl_snow_wind =  U10*ldew_snow/(1.56*1.e5)
            unl_snow_temp =  max(unl_snow_wind,0.0)
            unl_snow      =  unl_snow_temp+unl_snow_wind
            unl_snow      =  min(unl_snow,ldew_snow)

            xsc_snow      =  xsc_snow+unl_snow
            ldew_snow     = ldew_snow - unl_snow
         ENDIF

         ldew          = ldew - (xsc_rain + xsc_snow)

         IF(p0 > 1.e-8) THEN
            alpha_rain = 1.0
            alpha_snow = 1.0
            fpi_rain   = alpha_rain * tanh(lsai)
            fpi_snow   = alpha_snow * ( 1.-exp(-0.5*lsai) )
            tti_rain   = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fpi_rain )
            tti_snow   = (prc_snow+prl_snow)*deltim * ( 1.-fpi_snow )
            tex_rain   = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * fpi_rain -satcap_rain         !*(prc_rain+prl_rain)/p0 !(satcap-ldew) * xs
            tex_snow   = (prc_snow+prl_snow)*deltim * fpi_snow -satcap_snow         ! (ap/bp*(1.-exp(-bp*xs))+cp*xs) - (satcap-ldew) * xs
            tex_rain   = max( tex_rain, 0. )
            tex_snow   = max( tex_snow, 0. )

#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif
         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf      = p0 - (thru_rain + thru_snow)
         ldew_rain = ldew_rain+ (prc_rain + prl_rain + qflx_irrig_sprinkler)*deltim - thru_rain
         ldew_snow = ldew_snow+ (prc_snow + prl_snow)*deltim  - thru_snow

         ldew      = ldew_rain+ldew_snow !+ pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim
         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim

#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) 'w, ldew, (pg_rain+pg_snow)*deltim, satcap_rain,satcap_snow:',w, ldew, (pg_rain+pg_snow)*deltim, satcap_rain,satcap_snow
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF

   END SUBROUTINE LEAF_interception_CLM5

   SUBROUTINE LEAF_interception_NOAHMP(deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                       prc_rain,prc_snow,prl_rain,prl_snow,&
                                       ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Interception and drainage of precipitation
   ! the treatment are modified from Noah-MP 5.0

!Original Author:
!-------------------
   !---Guo-Yue Niu

!References:
!-------------------
   !---Yang, M., Zuo, R., Li, X. and Wang, L., 2019. Improvement test for the canopy interception parameterization scheme
   !   in the community land model. Sola, 15, pp.166-171.
   !---Niu, G.Y., Yang, Z.L., Mitchell, K.E., Chen, F., Ek, M.B., Barlage, M., Kumar, A.,
   !   Manning, K., Niyogi, D., Rosero, E. and Tewari, M., 2011. The community Noah land
   !   surface model with multiparameterization options (Noah‐MP): 1. Model description and evaluation
   !   with local‐scale measurements. Journal of Geophysical Research: Atmospheres, 116(D12).
   !---He, C., Valayamkunnath, P., Barlage, M., Chen, F., Gochis, D., Cabell, R., Schneider, T.,
   !   Rasmussen, R., Niu, G.Y., Yang, Z.L. and Niyogi, D., 2023. Modernizing the open-source
   !   community Noah-MP land surface model (version 5.0) with enhanced modularity,
   !   interoperability, and applicability. EGUsphere, 2023, pp.1-31.

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   ! 2023.02.21  Zhongwang Wei @ SYSU
   ! 2021.12.08  Zhongwang Wei @ SYSU
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in)    :: deltim    !seconds in a time step [second]
   real(r8), intent(in)    :: dewmx     !maximum dew [mm]
   real(r8), intent(in)    :: forc_us   !wind speed
   real(r8), intent(in)    :: forc_vs   !wind speed
   real(r8), intent(in)    :: chil      !leaf angle distribution factor
   real(r8), intent(in)    :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in)    :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in)    :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in)    :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in)    :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in)    :: lai       !leaf area index [-]
   real(r8), intent(in)    :: sai       !stem area index [-]
   real(r8), intent(in)    :: tair     !air temperature [K]
   real(r8), intent(inout) :: tleaf   !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of liquid on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of liquid on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of U

   real(r8), intent(out)   :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out)   :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out)   :: qintr_snow ! snowfall interception (mm h2o/s)
   real(r8)                :: BDFALL
      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         satcap_rain = dewmx*vegt
         BDFALL      = 67.92+51.25*EXP(MIN(2.5,(tleaf-273.15))/2.59)
         satcap_snow = 6.6*(0.27+46./BDFALL) * lsai
         satcap_snow = max(0.0,satcap_snow)
         fvegc=max(0.05,1.0-exp(-0.52*lsai))

         p0  = (prc_rain + prc_snow + prl_rain + prl_snow+qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+qflx_irrig_sprinkler)*deltim

         w   = ldew+p0

         xsc_rain    = max(0., ldew_rain-satcap_rain)
         xsc_snow    = max(0., ldew_snow-satcap_snow)

         ldew_rain   = ldew_rain-xsc_rain
         ldew_snow   = ldew_snow-xsc_snow

         !snow unloading
         IF (ldew_snow>1.e-8) THEN
            FT = MAX(0.0,(tair - 270.15) / 1.87E5)
            FV = SQRT(forc_us*forc_us + forc_vs*forc_vs) / 1.56E5
            ICEDRIP = MAX(0.,ldew_snow) * (FV+FT)    !MB: removed /DT
            ICEDRIP = MIN(ICEDRIP,ldew_snow)
            xsc_snow      =  xsc_snow+ICEDRIP
            ldew_snow     =  ldew_snow - ICEDRIP
         ENDIF

         ! phase change and excess !
         IF (tleaf > tfrz) THEN
            IF (ldew_snow>1.e-8) THEN
               ldew_smelt = MIN(ldew_snow,(tleaf-tfrz)*CICE*ldew_snow/DENICE/(HFUS))
               ldew_smelt = MAX(ldew_smelt,0.0)
               ldew_snow  = ldew_snow-ldew_smelt
               ldew_rain  = ldew_rain+ldew_smelt
               xsc_rain   = xsc_rain + MAX(0., ldew_rain-satcap_rain)
               ldew_rain  = ldew_rain - MAX(0., ldew_rain-satcap_rain)
            ENDIF
            ! tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ELSE
            IF (ldew_rain>1.e-8) THEN
               ldew_frzc  = MIN(ldew_rain,(tfrz-tleaf)*CWAT*ldew_rain/DENH2O/(HFUS))
               ldew_frzc  = MAX(ldew_frzc,0.0)
               ldew_snow  = ldew_snow+ldew_frzc
               ldew_rain  = ldew_rain-ldew_frzc
               xsc_snow   = xsc_snow + MAX(0., ldew_snow-satcap_snow)
               ldew_snow     = ldew_snow - MAX(0., ldew_snow-satcap_snow)
            ENDIF
            !tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ENDIF
         ldew          = ldew - (xsc_rain + xsc_snow)

         IF (p0 > 1.e-8) THEN

            tti_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fvegc )
            tti_snow = (prc_snow+prl_snow)*deltim * ( 1.-fvegc )

            FP=p0/(10.*ppc+ppl)

            int_rain=min(fvegc*FP*(prc_rain+prl_rain+qflx_irrig_sprinkler),(satcap_rain-ldew_rain)/deltim* &
            (1.0-exp(-(prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim/satcap_rain)))
            int_snow=min(fvegc*FP*(prc_snow + prl_snow),(satcap_snow-ldew_snow)/deltim* &
            (1.0-exp(-(prc_snow+prl_snow)*deltim/satcap_snow)))
            int_rain=max(0.,int_rain)*deltim
            int_snow=max(0.,int_snow)*deltim

            tex_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*fvegc*deltim  - int_rain
            tex_snow = (prc_snow+prl_snow)*fvegc*deltim - int_snow
#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif
         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !BDFALL = 67.92+51.25*EXP(MIN(2.5,(SFCTMP-TFRZ))/2.59)

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim


         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim


#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim  !, satcap
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.

      ENDIF

   END SUBROUTINE LEAF_interception_NOAHMP


   SUBROUTINE LEAF_interception_MATSIRO (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                         prc_rain,prc_snow,prl_rain,prl_snow,&
                                         ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,&
                                         qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Interception and drainage of precipitation
   ! the treatment are modified from MATSIRO 6 (under development)

!Original Author:
!-------------------
   !---MATSIRO6 document writing team∗

!References:
!-------------------
   !---Tatebe, H., Ogura, T., Nitta, T., Komuro, Y., Ogochi, K., Takemura, T., Sudo, K., Sekiguchi, M.,
   !   Abe, M., Saito, F. and Chikira, M., 2019. Description and basic evaluation of simulated mean state,
   !   internal variability, and climate sensitivity in MIROC6. Geoscientific Model Development, 12(7), pp.2727-2765. 116(D12).
   !---Takata, K., Emori, S. and Watanabe, T., 2003. Development of the minimal advanced treatments of surface interaction and
   !   runoff. Global and planetary Change, 38(1-2), pp.209-222.
   !---Guo, Q., Kino, K., Li, S., Nitta, T., Takeshima, A., Suzuki, K.T., Yoshida, N. and Yoshimura, K., 2021.
   !   Description of MATSIRO6.

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   ! 2023.02.21  Zhongwang Wei @ SYSU
   ! 2021.12.08  Zhongwang Wei @ SYSU
!=======================================================================

   IMPLICIT NONE

   real(r8), intent(in) :: deltim    !seconds in a time step [second]
   real(r8), intent(in) :: dewmx     !maximum dew [mm]
   real(r8), intent(in) :: forc_us   !wind speed
   real(r8), intent(in) :: forc_vs   !wind speed
   real(r8), intent(in) :: chil      !leaf angle distribution factor
   real(r8), intent(in) :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai       !leaf area index [-]
   real(r8), intent(in) :: sai       !stem area index [-]
   real(r8), intent(in) :: tair     !air temperature [K]
   real(r8), intent(inout) :: tleaf   !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of liquid on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of liquid on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of  U


   real(r8), intent(out) :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow ! snowfall interception (mm h2o/s)
   !local
   real(r8) :: fint, Ac, dewmx_MATSIRO,ldew_rain_s, ldew_snow_s,ldew_rain_n, ldew_snow_n
   real(r8) :: tex_rain_n,tex_rain_s,tex_snow_n,tex_snow_s,tti_rain_n,tti_rain_s,tti_snow_n,tti_snow_s

      !the canopy water capacity per leaf area index is set to 0.2mm
      dewmx_MATSIRO = 0.2
      !the fracrtion of the convective precipitation area is assumed to be uniform (0.1)
      Ac            = 0.1

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         p0  = (prc_rain + prc_snow + prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow + qflx_irrig_sprinkler)*deltim

         satcap_rain = dewmx_MATSIRO*vegt
         satcap_snow = dewmx_MATSIRO*vegt

         w = ldew+p0

         xsc_rain = max(0., ldew_rain-satcap_rain)
         xsc_snow = max(0., ldew_snow-satcap_snow)

         ldew_rain     = ldew_rain-xsc_rain
         ldew_snow     = ldew_snow-xsc_snow
         ! phase change and excess !
         IF (tleaf > tfrz) THEN
            IF (ldew_snow>1.e-8) THEN
               ldew_smelt = MIN(ldew_snow,(tleaf-tfrz)*CICE*ldew_snow/DENICE/(HFUS))
               ldew_smelt = MAX(ldew_smelt,0.0)
               ldew_snow  = ldew_snow-ldew_smelt
               ldew_rain  = ldew_rain+ldew_smelt
               xsc_rain   = xsc_rain + MAX(0., ldew_rain-satcap_rain)
               ldew_rain  = ldew_rain - MAX(0., ldew_rain-satcap_rain)
            ENDIF
            ! tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ELSE
            IF (ldew_rain>1.e-8) THEN
               ldew_frzc  = MIN(ldew_rain,(tfrz-tleaf)*CWAT*ldew_rain/DENH2O/(HFUS))
               ldew_frzc  = MAX(ldew_frzc,0.0)
               ldew_snow  = ldew_snow+ldew_frzc
               ldew_rain  = ldew_rain-ldew_frzc
               xsc_snow   = xsc_snow  + MAX(0., ldew_snow-satcap_snow)
               ldew_snow  = ldew_snow - MAX(0., ldew_snow-satcap_snow)
            ENDIF
            !tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ENDIF
         ldew          = ldew - (xsc_rain + xsc_snow)

         IF (p0 > 1.e-8) THEN
            ! interception efficiency
            fpi_rain  = min(1.0,lai+sai)
            fpi_snow  = min(1.0,lai+sai)

            !-----------------------------------------------------------------------
            ! Storm area
            !-----------------------------------------------------------------------
            ldew_rain_s = ldew_rain + ((prl_rain+qflx_irrig_sprinkler) * fpi_rain + prc_rain * fpi_rain / Ac)  * deltim
            ldew_snow_s = ldew_snow + (prl_snow * fpi_snow + prc_snow * fpi_snow / Ac)  * deltim
            !
            tti_rain_s  = (prl_rain+qflx_irrig_sprinkler + prc_rain/Ac) * (1.d0-fpi_rain) * deltim
            tti_snow_s  = (prl_snow + prc_snow/Ac) * (1.d0-fpi_snow) * deltim

            tex_rain_s  = max(ldew_rain_s - satcap_rain, 0.d0) + (1.14d-11)*1000.*deltim* &
            exp(min(ldew_rain_s,satcap_rain)/1000.* 3.7d3 ) !cwb_adrp1 = 1.14d-11   ! dripping coefficient, [m/sec] rutter et.al.(1975)
            tex_rain_s  = min(tex_rain_s, ldew_rain_s)
            ldew_rain_s = ldew_rain_s - tex_rain_s

            !
            tex_snow_s  = max(ldew_snow_s - satcap_snow, 0.d0) + (1.14d-11)*1000.*deltim* &
            exp(min(ldew_snow_s,satcap_snow)/1000.0* 3.7d3 ) !cwb_adrp2 = 3.7d3      ! dripping coefficient, [/m] rutter et.al.(1975)
            tex_snow_s  = min(tex_snow_s, ldew_snow_s)
            ldew_snow_s = ldew_snow_s - tex_snow_s

            !-------------------------------------------------------------------------
            ! Non-storm area
            !-------------------------------------------------------------------------
            ldew_rain_n = ldew_rain + (prl_rain+qflx_irrig_sprinkler) * fpi_rain  * deltim
            ldew_snow_n = ldew_snow + prl_snow * fpi_snow  * deltim

            !
            tti_rain_n  = (prl_rain+qflx_irrig_sprinkler) * (1.d0-fpi_rain) * deltim
            tti_snow_n  = (prl_snow) * (1.d0-fpi_snow) * deltim


            tex_rain_n  = max(ldew_rain_n  - satcap_rain, 0.d0) + (1.14d-11)*1000.*deltim* &
            exp(min(ldew_rain_n,satcap_rain)/1000.* 3.7d3)
            tex_rain_n  = min(tex_rain_n, ldew_rain_n)
            ldew_rain_n = ldew_rain_n - tex_rain_n

            !
            tex_snow_n  =  max(ldew_snow_n - satcap_snow, 0.d0) + (1.14d-11)*1000.*deltim* &
            exp(min(ldew_snow_n,satcap_snow)/1000.* 3.7d3 )
            tex_snow_n  =  min(tex_snow_n, ldew_snow_n)
            ldew_snow_n =  ldew_snow_n - tex_snow_n
            !-------------------------------------------------------------------------
            !-------------------------------------------------------------------------
            ! Average
            !-------------------------------------------------------------------------
            ldew_rain = ldew_rain_n + (ldew_rain_s - ldew_rain_n) * Ac
            ldew_snow = ldew_snow_n + (ldew_snow_s - ldew_snow_n) * Ac
            ldew_rain = max(0.0,ldew_rain)
            ldew_snow = max(0.0,ldew_snow)

            tti_rain  = tti_rain_n*(1-Ac)+tti_rain_s*Ac
            tti_snow  = tti_snow_n+(tti_snow_s-tti_snow_n) * Ac
            tti_rain  = max(0.0,tti_rain)
            tti_snow  = max(0.0,tti_snow)

            tex_rain  = tex_rain_n+(tex_rain_s-tex_rain_n)*Ac
            tex_snow  = tex_snow_n+(tex_snow_s-tex_snow_n)*Ac
            tex_rain  = max(0.0,tex_rain)
            tex_snow  = max(0.0,tex_snow)
            !-------------------------------------------------------------------------


#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif

         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF

         !BDFALL = 67.92+51.25*EXP(MIN(2.5,(SFCTMP-TFRZ))/2.59)

         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf
         ldew_rain= ldew_rain+(prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim- thru_rain
         ldew_snow= ldew_snow+(prc_snow+prl_snow)*deltim- thru_snow

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim
#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim !, satcap
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF
   END SUBROUTINE LEAF_interception_MATSIRO

   SUBROUTINE LEAF_interception_VIC (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                       prc_rain,prc_snow,prl_rain,prl_snow,&
                                       ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                       pg_snow,qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   ! Calculation of  interception and drainage of precipitation
   ! the treatment are based on VIC 5.0 (under development)

!Original Author:
!-------------------
   !---Hamman, J.J. AND Liang X.

!References:
!-------------------
   !---Hamman, J.J., Nijssen, B., Bohn, T.J., Gergel, D.R. and Mao, Y., 2018.
   !   The Variable Infiltration Capacity model version 5 (VIC-5): Infrastructure
   !   improvements for new applications and reproducibility. Geoscientific Model Development,
   !   11(8), pp.3481-3496.
   !---Liang, X., Lettenmaier, D.P., Wood, E.F. and Burges, S.J., 1994.
   !   A simple hydrologically based model of land surface water and energy fluxes
   !   for general circulation models. Journal of Geophysical Research: Atmospheres, 99(D7),
   !   pp.14415-14428.

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!REVISION HISTORY
!----------------
   ! 2023.02.21  Zhongwang Wei @ SYSU
   ! 2021.12.08  Zhongwang Wei @ SYSU
!=======================================================================


   IMPLICIT NONE

   real(r8), intent(in) :: deltim    !seconds in a time step [second]
   real(r8), intent(in) :: dewmx     !maximum dew [mm]
   real(r8), intent(in) :: forc_us   !wind speed
   real(r8), intent(in) :: forc_vs   !wind speed
   real(r8), intent(in) :: chil      !leaf angle distribution factor
   real(r8), intent(in) :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in) :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in) :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in) :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in) :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in) :: lai       !leaf area index [-]
   real(r8), intent(in) :: sai       !stem area index [-]
   real(r8), intent(in) :: tair     !air temperature [K]
   real(r8), intent(inout) :: tleaf   !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of liquid on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of liquid on foliage [mm]
   real(r8), intent(in) :: z0m            !roughness length
   real(r8), intent(in) :: hu             !forcing height of U


   real(r8), intent(out) :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out) :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out) :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out) :: qintr_snow ! snowfall interception (mm h2o/s)

   real(r8) :: Imax1,Lr,ldew_max_snow,Snow,Rain,DeltaSnowInt,Wind,BlownSnow,SnowThroughFall
   real(r8) :: MaxInt,MaxWaterInt,RainThroughFall,Overload,IntRainFract,IntSnowFract,ldew_smelt
   real(r8) :: drip

      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         !the maximum bearing  capacity of the tree regardless of air temp (Imax1)
         Imax1=4.0*lsai*0.0005 *1000.0 ! in mm
         MaxInt=0.1*lsai
         IF (tair>-272.15) THEN
            Lr=4.0
         ELSE IF (tair<=-272.15 .and. tair>=-270.15) THEN
            Lr=1.5*(tair-273.15)+5.5
         ELSE
            Lr=1.0
         ENDIF

         satcap_snow=0.0005 *Lr *lsai * 1000.0  ! in mm !!!
         !/* Calculate amount of snow intercepted on branches and stored in  intercepted snow. */
         satcap_rain= 0.035 * (ldew_snow) + MaxInt !

         p0  = (prc_rain + prc_snow + prl_rain + prl_snow+ qflx_irrig_sprinkler)*deltim
         ppc = (prc_rain+prc_snow)*deltim
         ppl = (prl_rain+prl_snow+ qflx_irrig_sprinkler)*deltim
         w = ldew+p0

         xsc_rain   = max(0., ldew_rain-satcap_rain)
         xsc_snow   = max(0., ldew_snow-satcap_snow)

         ldew_rain  = ldew_rain-xsc_rain
         ldew_snow  = ldew_snow-xsc_snow
         ! phase change and excess !
         IF (tleaf > tfrz) THEN
            IF (ldew_snow>1.e-8) THEN
               ldew_smelt = MIN(ldew_snow,(tleaf-tfrz)*CICE*ldew_snow/DENICE/(HFUS))
               ldew_smelt = MAX(ldew_smelt,0.0)
               ldew_snow  = ldew_snow-ldew_smelt
               ldew_rain  = ldew_rain+ldew_smelt
               xsc_rain   = xsc_rain  + MAX(0., ldew_rain-satcap_rain)
               ldew_rain  = ldew_rain - MAX(0., ldew_rain-satcap_rain)
            ENDIF
            ! tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ELSE
            IF (ldew_rain>1.e-8) THEN
               ldew_frzc  = MIN(ldew_rain,(tfrz-tleaf)*CWAT*ldew_rain/DENH2O/(HFUS))
               ldew_frzc  = MAX(ldew_frzc,0.0)
               ldew_snow  = ldew_snow+ldew_frzc
               ldew_rain  = ldew_rain-ldew_frzc
               xsc_snow   = xsc_snow  + MAX(0., ldew_snow-satcap_snow)
               ldew_snow  = ldew_snow - MAX(0., ldew_snow-satcap_snow)
            ENDIF
            !tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ENDIF

         ldew          = ldew -(xsc_rain+xsc_snow)

         IF (p0 > 1.e-8) THEN
            ! interception efficiency
            fpi_rain  = min(1.0,lai+sai)
            fpi_snow  = min(1.0,lai+sai)

            tti_rain    = (prc_rain+prl_rain+ qflx_irrig_sprinkler)*deltim * ( 1. - fpi_rain )
            tti_snow    = (prc_snow+prl_snow)*deltim * ( 1. - fpi_snow )

            ldew_rain   = ldew_rain + (prc_rain+prl_rain+ qflx_irrig_sprinkler)*deltim * fpi_rain
            ldew_snow   = ldew_snow + (prc_snow+prl_snow)*deltim  * fpi_snow

            tex_rain    = max(0.0,ldew_rain-satcap_rain)
            tex_snow    = max(0.0,ldew_snow-satcap_snow)

            ldew_rain   = ldew_rain - tex_rain
            ldew_snow   = ldew_snow - tex_snow

            !unload of snow
            !* Reduce the amount of intercepted snow if windy and cold.
            !Ringyo Shikenjo Tokyo, #54, 1952.
            !Bulletin of the Govt. Forest Exp. Station,
            !Govt. Forest Exp. Station, Meguro, Tokyo, Japan.
            !FORSTX 634.9072 R475r #54.
            !Page 146, Figure 10.

            !Reduce the amount of intercepted snow if snowing, windy, and
            !cold (< -3 to -5 C).
            !Schmidt and Troendle 1992 western snow conference paper. */
            Wind= SQRT(forc_us*forc_us + forc_vs*forc_vs)
            IF (tleaf-273.15<-3.0 .and. Wind> 1.0) THEN
               BlownSnow=(0.2*Wind -0.2)* ldew_snow
               BlownSnow = min(ldew_snow,BlownSnow)
               tex_snow    =  tex_snow  + BlownSnow
               ldew_snow   =  ldew_snow - BlownSnow
            ENDIF
            !/* at this point we have calculated the amount of snowfall intercepted and
            !/* the amount of rainfall intercepted.  These values have been
            !/* appropriately subtracted from SnowFall and RainFall to determine
            !/* SnowThroughfall and RainThroughfall.  However, we can end up with the
            !/* condition that the total intercepted rain plus intercepted snow is
            !/* greater than the maximum bearing capacity of the tree regardless of air
            !/* temp (Imax1).  The following routine will adjust ldew_rain and ldew_snow
            !/* by triggering mass release due to overloading.  Of course since ldew_rain
            !/* and ldew_snow are mixed, we need to slough them of as fixed fractions  */
            IF (ldew_rain + ldew_snow > Imax1) THEN
               ! /*THEN trigger structural unloading*/
               Overload = (ldew_snow + ldew_rain) - Imax1
               IntRainFract = ldew_rain / (ldew_rain + ldew_snow)
               IntSnowFract = 1.0 - IntRainFract
               ldew_rain = ldew_rain - Overload * IntRainFract
               ldew_snow = ldew_snow - Overload * IntSnowFract
               tex_rain  = tex_rain  + Overload*IntRainFract
               tex_snow  = tex_snow  + Overload*IntSnowFract
            ENDIF

#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif

         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF


         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow

         ldew_rain= ldew_rain+(prc_rain+prl_rain+ qflx_irrig_sprinkler)*deltim- thru_rain
         ldew_snow= ldew_snow+(prc_snow+prl_snow)*deltim- thru_snow

         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf


         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim
#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim !, satcap
            CALL abort
         ENDIF
#endif

      ELSE
       ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF
   END SUBROUTINE LEAF_interception_VIC

   SUBROUTINE LEAF_interception_JULES(deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                       prc_rain,prc_snow,prl_rain,prl_snow,&
                                       ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,qintr_rain,qintr_snow)
   !DESCRIPTION
   !===========
      ! Interception and drainage of precipitation
      ! the treatment are modified from JULES

   !Original Author:
   !-------------------
      !---JULES development and research community

   !References:
   !-------------------
      !---Best et al. (2011): The Joint UK Land Environment Simulator (JULES), model description –
      !   Part 1: Energy and water fluxes. Geosci. Model Dev. 4:677–699.
      !---Clark et al. (2011): The Joint UK Land Environment Simulator (JULES), model description –
      !   Part 2: Carbon fluxes and vegetation dynamics. Geosci. Model Dev. 4:701–722.

   !ANCILLARY FUNCTIONS AND SUBROUTINES
   !-------------------

   !REVISION HISTORY
   !----------------
      ! 2023.02.21  Zhongwang Wei @ SYSU
      ! 2021.12.08  Zhongwang Wei @ SYSU
   !=======================================================================

      IMPLICIT NONE

   real(r8), intent(in)    :: deltim    !seconds in a time step [second]
   real(r8), intent(in)    :: dewmx     !maximum dew [mm]
   real(r8), intent(in)    :: forc_us   !wind speed
   real(r8), intent(in)    :: forc_vs   !wind speed
   real(r8), intent(in)    :: chil      !leaf angle distribution factor
   real(r8), intent(in)    :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in)    :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in)    :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in)    :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in)    :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in)    :: lai       !leaf area index [-]
   real(r8), intent(in)    :: sai       !stem area index [-]
   real(r8), intent(in)    :: tair     !air temperature [K]
   real(r8), intent(inout) :: tleaf   !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of liquid on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of liquid on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of U

   real(r8), intent(out)   :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out)   :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out)   :: qintr_snow ! snowfall interception (mm h2o/s)
   real(r8)                :: snowinterceptfact,unload_rate_cnst,unload_rate_u,Wind
      IF (lai+sai > 1e-6) THEN
         lsai   = lai + sai
         vegt   = lsai
         !---------------------------------------------------
         !TODO: these variable should be based on vegetation type
         unload_rate_cnst= 0.001 !a constant term (kg m-2 s-1) that represents unloading processes like sublimation, wind erosion etc.
         unload_rate_u   = 0.001 !wind speed dependent term (s-1*1000) that causes additional unloading proportional to wind speed.
         !---------------------------------------------------
         ! Constant in relationship between mass of intercepted snow and snowfall rate
         snowinterceptfact = 0.6
         satcap_snow       = 4.4 *lsai
         satcap_rain       = 0.1 *lsai

         ! Caution here: JULES is PFT based, fvegc is not exxisitng
         fvegc       = max(0.05,1.0-exp(-0.52*lsai))

         p0          = (prc_rain + prc_snow + prl_rain + prl_snow+qflx_irrig_sprinkler)*deltim
         ppc         = (prc_rain + prc_snow)*deltim
         ppl         = (prl_rain + prl_snow + qflx_irrig_sprinkler)*deltim

         w           = ldew+p0

         xsc_rain    = max(0., ldew_rain-satcap_rain)
         xsc_snow    = max(0., ldew_snow-satcap_snow)

         ldew_rain   = ldew_rain-xsc_rain
         ldew_snow   = ldew_snow-xsc_snow

         !snow unloading
         !something wrong with this part in JULES, need to be checked
!         IF (ldew_snow>1.e-8) THEN
!            Wind= SQRT(forc_us*forc_us + forc_vs*forc_vs)
!            IF (Wind > 1.0) THEN
!               ICEDRIP       =  unload_rate_cnst + unload_rate_u * Wind
!               ICEDRIP       =  MIN(ICEDRIP,ldew_snow)
!               xsc_snow      =  xsc_snow+ICEDRIP
!               ldew_snow     =  ldew_snow - ICEDRIP
!            ENDIF
!         ENDIF

         ! phase change and excess !
         IF (tleaf > tfrz) THEN
            IF (ldew_snow>1.e-8) THEN
               ldew_smelt = MIN(ldew_snow,(tleaf-tfrz)*CICE*ldew_snow/DENICE/(HFUS))
               ldew_smelt = MAX(ldew_smelt,0.0)
               ldew_snow  = ldew_snow-ldew_smelt
               ldew_rain  = ldew_rain+ldew_smelt
               xsc_rain   = xsc_rain + MAX(0., ldew_rain-satcap_rain)
               ldew_rain  = ldew_rain - MAX(0., ldew_rain-satcap_rain)
            ENDIF
            ! tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ELSE
            IF (ldew_rain>1.e-8) THEN
               ldew_frzc  = MIN(ldew_rain,(tfrz-tleaf)*CWAT*ldew_rain/DENH2O/(HFUS))
               ldew_frzc  = MAX(ldew_frzc,0.0)
               ldew_snow  = ldew_snow+ldew_frzc
               ldew_rain  = ldew_rain-ldew_frzc
               xsc_snow   = xsc_snow + MAX(0., ldew_snow-satcap_snow)
               ldew_snow  = ldew_snow - MAX(0., ldew_snow-satcap_snow)
            ENDIF
            !tleaf      = fvegc*tfrz+ (1.0-fwet)*tleaf
         ENDIF
         ldew          = ldew - (xsc_rain + xsc_snow)

         IF (p0 > 1.e-8) THEN

            tti_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim * ( 1.-fvegc )
            tti_snow = (prc_snow+prl_snow)*deltim * ( 1.-fvegc )
            int_rain = min(fvegc*(prc_rain+prl_rain+qflx_irrig_sprinkler),snowinterceptfact*(satcap_rain-ldew_rain)/deltim* &
            (1.0-exp(-(prc_rain+prl_rain+qflx_irrig_sprinkler)*deltim/satcap_rain)))
            int_snow = min(fvegc*(prc_snow + prl_snow),snowinterceptfact*(satcap_snow-ldew_snow)/deltim* &
            (1.0-exp(-(prc_snow+prl_snow)*deltim/satcap_snow)))
            int_rain = max(0.,int_rain)*deltim
            int_snow = max(0.,int_snow)*deltim

            tex_rain = (prc_rain+prl_rain+qflx_irrig_sprinkler)*fvegc*deltim  - int_rain
            tex_snow = (prc_snow+prl_snow)*fvegc*deltim - int_snow
#if(defined CoLMDEBUG)
            IF (tex_rain+tex_snow+tti_rain+tti_snow-p0 > 1.e-10) THEN
               write(6,*) 'tex_ + tti_ > p0 in interception code : '
            ENDIF
#endif
         ELSE
            ! all intercepted by canopy leves for very small precipitation
            tti_rain = 0.
            tti_snow = 0.
            tex_rain = 0.
            tex_snow = 0.
         ENDIF
         !----------------------------------------------------------------------
         !   total throughfall (thru) and store augmentation
         !----------------------------------------------------------------------

         thru_rain = tti_rain + tex_rain
         thru_snow = tti_snow + tex_snow
         pinf = p0 - (thru_rain + thru_snow)
         ldew = ldew + pinf

         pg_rain = (xsc_rain + thru_rain) / deltim
         pg_snow = (xsc_snow + thru_snow) / deltim
         qintr   = pinf / deltim

         qintr_rain = prc_rain + prl_rain + qflx_irrig_sprinkler - thru_rain / deltim
         qintr_snow = prc_snow + prl_snow - thru_snow / deltim
#if(defined CoLMDEBUG)
         w = w - ldew - (pg_rain+pg_snow)*deltim
         IF (abs(w) > 1.e-6) THEN
            write(6,*) 'something wrong in interception code : '
            write(6,*) w, ldew, (pg_rain+pg_snow)*deltim !, satcap
            CALL abort
         ENDIF
#endif
      ELSE
         ! 07/15/2023, yuan: #bug found for ldew value reset.
         !NOTE: this bug should exist in other interception schemes @Zhongwang.
         IF (ldew > 0.) THEN
            IF (tleaf > tfrz) THEN
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler + ldew/deltim
               pg_snow = prc_snow + prl_snow
            ELSE
               pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
               pg_snow = prc_snow + prl_snow + ldew/deltim
            ENDIF
         ELSE
            pg_rain = prc_rain + prl_rain + qflx_irrig_sprinkler
            pg_snow = prc_snow + prl_snow
         ENDIF

         ldew  = 0.
         qintr = 0.
         qintr_rain = 0.
         qintr_snow = 0.
      ENDIF
   END SUBROUTINE LEAF_interception_JULES

   SUBROUTINE LEAF_interception_wrap(deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                                            prc_rain,prc_snow,prl_rain,prl_snow,&
                                                         ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                                pg_snow,qintr,qintr_rain,qintr_snow)
!DESCRIPTION
!===========
   !wrapper for calculation of canopy interception using USGS or IGBP land cover classification

!ANCILLARY FUNCTIONS AND SUBROUTINES
!-------------------

!Original Author:
!-------------------
   !---Shupeng Zhang

!References:


!REVISION HISTORY
!----------------

   IMPLICIT NONE

   real(r8), intent(in)    :: deltim    !seconds in a time step [second]
   real(r8), intent(in)    :: dewmx     !maximum dew [mm]
   real(r8), intent(in)    :: forc_us   !wind speed
   real(r8), intent(in)    :: forc_vs   !wind speed
   real(r8), intent(in)    :: chil      !leaf angle distribution factor
   real(r8), intent(in)    :: prc_rain  !convective ranfall [mm/s]
   real(r8), intent(in)    :: prc_snow  !convective snowfall [mm/s]
   real(r8), intent(in)    :: prl_rain  !large-scale rainfall [mm/s]
   real(r8), intent(in)    :: prl_snow  !large-scale snowfall [mm/s]
   real(r8), intent(in)    :: sigf      !fraction of veg cover, excluding snow-covered veg [-]
   real(r8), intent(in)    :: lai       !leaf area index [-]
   real(r8), intent(in)    :: sai       !stem area index [-]
   real(r8), intent(in)    :: tair     !air temperature [K]
   real(r8), intent(inout) :: tleaf   !sunlit canopy leaf temperature [K]

   real(r8), intent(inout) :: ldew   !depth of water on foliage [mm]
   real(r8), intent(inout) :: ldew_rain   !depth of liquid on foliage [mm]
   real(r8), intent(inout) :: ldew_snow   !depth of liquid on foliage [mm]
   real(r8), intent(in)    :: z0m            !roughness length
   real(r8), intent(in)    :: hu             !forcing height of U


   real(r8), intent(out)   :: pg_rain  !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: pg_snow  !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: qintr    !interception [kg/(m2 s)]
   real(r8), intent(out)   :: qintr_rain ! rainfall interception (mm h2o/s)
   real(r8), intent(out)   :: qintr_snow ! snowfall interception (mm h2o/s)

      IF (DEF_Interception_scheme==1) THEN
         CALL LEAF_interception_CoLM2014 (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)
      ELSEIF (DEF_Interception_scheme==2) THEN
         CALL LEAF_interception_CLM4 (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)
      ELSEIF (DEF_Interception_scheme==3) THEN
         CALL LEAF_interception_CLM5(deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)
      ELSEIF (DEF_Interception_scheme==4) THEN
         CALL LEAF_interception_NoahMP (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)
      ELSEIF  (DEF_Interception_scheme==5) THEN
         CALL LEAF_interception_matsiro (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)

      ELSEIF  (DEF_Interception_scheme==6) THEN
         CALL LEAF_interception_vic (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)

      ELSEIF  (DEF_Interception_scheme==7) THEN
         CALL LEAF_interception_JULES (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)

      ELSEIF  (DEF_Interception_scheme==8) THEN
         CALL LEAF_interception_colm202x (deltim,dewmx,forc_us,forc_vs,chil,sigf,lai,sai,tair,tleaf, &
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,&
                                             pg_snow,qintr,qintr_rain,qintr_snow)
      ENDIF

   END SUBROUTINE LEAF_interception_wrap

#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   SUBROUTINE LEAF_interception_pftwrap (ipatch,deltim,dewmx,forc_us,forc_vs,forc_t,&
                               prc_rain,prc_snow,prl_rain,prl_snow,&
                               ldew,ldew_rain,ldew_snow,z0m,hu,pg_rain,pg_snow,qintr,qintr_rain,qintr_snow)

! -----------------------------------------------------------------
! !DESCRIPTION:
! wrapper for calculation of canopy interception for PFTs within a land cover type.
!
! Created by Hua Yuan, 06/2019
!
! !REVISION HISTORY:
! 2023.02.21 Zhongwang Wei @ SYSU: add different options of canopy interception for PFTs
!
! -----------------------------------------------------------------

   USE MOD_Precision
   USE MOD_LandPFT
   USE MOD_Const_Physical, only: tfrz
   USE MOD_Vars_PFTimeInvariants
   USE MOD_Vars_PFTimeVariables
   USE MOD_Vars_1DPFTFluxes
   USE MOD_Const_PFT
   IMPLICIT NONE

   integer,  intent(in)    :: ipatch      !patch index
   real(r8), intent(in)    :: deltim      !seconds in a time step [second]
   real(r8), intent(in)    :: dewmx       !maximum dew [mm]
   real(r8), intent(in)    :: forc_us     !wind speed
   real(r8), intent(in)    :: forc_vs     !wind speed
   real(r8), intent(in)    :: forc_t      !air temperature
   real(r8), intent(in)    :: z0m         !roughness length
   real(r8), intent(in)    :: hu          !forcing height of U
   real(r8), intent(in)    :: ldew_rain   !depth of water on foliage [mm]
   real(r8), intent(in)    :: ldew_snow   !depth of water on foliage [mm]
   real(r8), intent(in)    :: prc_rain    !convective ranfall [mm/s]
   real(r8), intent(in)    :: prc_snow    !convective snowfall [mm/s]
   real(r8), intent(in)    :: prl_rain    !large-scale rainfall [mm/s]
   real(r8), intent(in)    :: prl_snow    !large-scale snowfall [mm/s]

   real(r8), intent(inout) :: ldew     !depth of water on foliage [mm]
   real(r8), intent(out)   :: pg_rain    !rainfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: pg_snow    !snowfall onto ground including canopy runoff [kg/(m2 s)]
   real(r8), intent(out)   :: qintr      !interception [kg/(m2 s)]
   real(r8), intent(out)   :: qintr_rain !rainfall interception (mm h2o/s)
   real(r8), intent(out)   :: qintr_snow !snowfall interception (mm h2o/s)

   integer i, p, ps, pe
#ifdef CROP
   integer  :: irrig_flag  ! 1 if sprinker, 2 if others
#endif
   real(r8) pg_rain_tmp, pg_snow_tmp

      pg_rain_tmp = 0.
      pg_snow_tmp = 0.

      ps = patch_pft_s(ipatch)
      pe = patch_pft_e(ipatch)

      IF(.not. DEF_USE_IRRIGATION) qflx_irrig_sprinkler = 0._r8

#ifdef CROP
      IF(DEF_USE_IRRIGATION)THEN
         CALL CalIrrigationApplicationFluxes(ipatch,ps,pe,deltim,qflx_irrig_drip,qflx_irrig_sprinkler,qflx_irrig_flood,qflx_irrig_paddy,irrig_flag=1)
      ENDIF
#endif

      IF (DEF_Interception_scheme==1) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_CoLM2014 (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                                prc_rain,prc_snow,prl_rain,prl_snow,&
                                                ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==2) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_clm4 (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==3) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_clm5 (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==4) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_clm5 (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==5) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_MATSIRO (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==6) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_VIC (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==7) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_JULES (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ELSE IF (DEF_Interception_scheme==8) THEN
         DO i = ps, pe
            p = pftclass(i)
            CALL LEAF_interception_CoLM202x (deltim,dewmx,forc_us,forc_vs,chil_p(p),sigf_p(i),lai_p(i),sai_p(i),forc_t,tleaf_p(i),&
                                             prc_rain,prc_snow,prl_rain,prl_snow,&
                                             ldew_p(i),ldew_p(i),ldew_p(i),z0m_p(i),hu,pg_rain,pg_snow,qintr_p(i),qintr_rain_p(i),qintr_snow_p(i))
            pg_rain_tmp = pg_rain_tmp + pg_rain*pftfrac(i)
            pg_snow_tmp = pg_snow_tmp + pg_snow*pftfrac(i)
         ENDDO
      ENDIF

     pg_rain = pg_rain_tmp
     pg_snow = pg_snow_tmp
     ldew  = sum(ldew_p(ps:pe) * pftfrac(ps:pe))
     qintr = sum(qintr_p(ps:pe) * pftfrac(ps:pe))
     qintr_rain = sum(qintr_rain_p(ps:pe) * pftfrac(ps:pe))
     qintr_snow = sum(qintr_snow_p(ps:pe) * pftfrac(ps:pe))
   END SUBROUTINE LEAF_interception_pftwrap
#endif

END MODULE MOD_LeafInterception
