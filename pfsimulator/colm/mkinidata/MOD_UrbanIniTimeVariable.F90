#include <define.h>

#ifdef URBAN_MODEL
MODULE MOD_UrbanIniTimeVariable

!=======================================================================
! Created by Hua Yuan, 09/16/2021
!
!=======================================================================

   USE MOD_Precision
   IMPLICIT NONE
   SAVE

   PUBLIC :: UrbanIniTimeVar

CONTAINS

   SUBROUTINE UrbanIniTimeVar(ipatch,froof,fgper,flake,hwr,hroof,&
                    alb_roof,alb_wall,alb_gimp,alb_gper,&
                    rho,tau,fveg,htop,hbot,lai,sai,coszen,&
                    fsno_roof,fsno_gimp,fsno_gper,fsno_lake,&
                    scv_roof,scv_gimp,scv_gper,scv_lake,&
                    sag_roof,sag_gimp,sag_gper,sag_lake,tlake,fwsun,dfwsun,&
                    extkd,alb,ssun,ssha,sroof,swsun,swsha,sgimp,sgper,slake)

   USE MOD_Precision
   USE MOD_Vars_Global
   USE MOD_Urban_Albedo

   IMPLICIT NONE

   integer, intent(in) :: &
         ipatch          ! patch index

   real(r8), intent(in) :: &
         froof,         &! roof fraction
         fgper,         &! impervious ground weight fraction
         flake,         &! lake fraction
         hwr,           &! average building height to their distance
         hroof           ! average building height

   real(r8), intent(in) :: &
         alb_roof(2,2), &! roof albedo (iband,direct/diffuse)
         alb_wall(2,2), &! wall albedo (iband,direct/diffuse)
         alb_gimp(2,2), &! impervious albedo (iband,direct/diffuse)
         alb_gper(2,2)   ! pervious albedo (iband,direct/diffuse)

   real(r8), intent(in) :: &
         fveg,          &! fraction of vegetation cover
         lai,           &! leaf area index
         sai,           &! stem area index
         htop,          &! canopy crown top height
         hbot,          &! canopy crown bottom height
         coszen,        &! cosine of solar zenith angle
         rho(2,2),      &! leaf reflectance (iw=iband, il=life and dead)
         tau(2,2)        ! leaf transmittance (iw=iband, il=life and dead)

   real(r8), intent(out) :: &
         fsno_roof,     &! fraction of soil covered by snow [-]
         fsno_gimp,     &! fraction of soil covered by snow [-]
         fsno_gper,     &! fraction of soil covered by snow [-]
         fsno_lake,     &! fraction of soil covered by snow [-]
         scv_roof,      &! snow cover, water equivalent [mm]
         scv_gimp,      &! snow cover, water equivalent [mm]
         scv_gper,      &! snow cover, water equivalent [mm]
         scv_lake,      &! snow cover, water equivalent [mm]
         sag_roof,      &! non dimensional snow age [-]
         sag_gimp,      &! non dimensional snow age [-]
         sag_gper,      &! non dimensional snow age [-]
         sag_lake,      &! non dimensional snow age [-]
         tlake           ! lake temperature

   real(r8), intent(out) :: &
         fwsun,         &! sunlit wall fraction [-]
         dfwsun,        &! change of fwsun
         extkd,         &! diffuse and scattered diffuse PAR extinction coefficient
         alb (2,2),     &! averaged albedo [-]
         ssun(2,2),     &! sunlit canopy absorption for solar radiation
         ssha(2,2),     &! shaded canopy absorption for solar radiation
         sroof(2,2),    &! roof absorption for solar radiation,
         swsun(2,2),    &! sunlit wall absorption for solar radiation,
         swsha(2,2),    &! shaded wall absorption for solar radiation,
         sgimp(2,2),    &! impervious ground absorption for solar radiation,
         sgper(2,2),    &! pervious ground absorption for solar radiation,
         slake(2,2)      ! lake absorption for solar radiation,

   !-----------------------------------------------------------------------
   real(r8) :: hveg    !height of crown central hight

      fsno_roof   = 0.   !fraction of ground covered by snow
      fsno_gimp   = 0.   !fraction of ground covered by snow
      fsno_gper   = 0.   !fraction of ground covered by snow
      fsno_lake   = 0.   !fraction of soil covered by snow [-]
      scv_roof    = 0.   !snow cover, water equivalent [mm, kg/m2]
      scv_gimp    = 0.   !snow cover, water equivalent [mm, kg/m2]
      scv_gper    = 0.   !snow cover, water equivalent [mm, kg/m2]
      scv_lake    = 0.   !snow cover, water equivalent [mm]
      sag_roof    = 0.   !roof snow age [-]
      sag_gimp    = 0.   !impervious ground snow age [-]
      sag_gper    = 0.   !pervious ground snow age [-]
      sag_lake    = 0.   !urban lake snow age [-]

      fwsun       = 0.5  !Fraction of sunlit wall [-]
      dfwsun      = 0.   !change of fwsun

      hveg        = min(hroof, (htop+hbot)/2.)

      ! urban surface albedo
      CALL alburban (ipatch,froof,fgper,flake,hwr,hroof,&
                     alb_roof,alb_wall,alb_gimp,alb_gper,&
                     rho,tau,fveg,hveg,lai,sai,max(0.01,coszen),fwsun,tlake,&
                     fsno_roof,fsno_gimp,fsno_gper,fsno_lake,&
                     scv_roof,scv_gimp,scv_gper,scv_lake,&
                     sag_roof,sag_gimp,sag_gper,sag_lake,&
                     dfwsun,extkd,alb,ssun,ssha,sroof,swsun,swsha,sgimp,sgper,slake)

   END SUBROUTINE UrbanIniTimeVar

END MODULE MOD_UrbanIniTimeVariable
!-----------------------------------------------------------------------
! EOP
#endif
