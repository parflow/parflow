!#include <misc.h>

subroutine pf_getforce (nx,ny,sw_pf,lw_pf,prcp_pf,tas_pf,u_pf,v_pf, &
  patm_pf,qatm_pf,lai_pf,sai_pf,z0m_pf,displa_pf,clm_forc_veg, &
  numpatch,planar_mask,idate)

!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! DESCRIPTION:
!  Access meteorological data - this current version reads 1D forcing
!  and distributes it to the clm domain (spatially constant).  This routine
!  must be modified to allow for spatially variable forcing, or coupling to
!  a GCM.
!
!  The user may likely want to modify this subroutine significantly,
!  to include such things as space/time intrpolation of forcing to the
!  CLM grid, reading of spatially variable binary data, etc.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: drv_getforce.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  !use precision
  !use drv_module          ! 1-D Land Model Driver variables
  !use drv_tilemodule      ! Tile-space variables
  !use clmtype             ! 1-D CLM variables
  !use clm_varcon, only : tfrz, tcrit
  USE MOD_Vars_1DForcing
  USE MOD_OrbCoszen
  USE MOD_Vars_TimeInvariants, only: patchlonr, patchlatr
  USE MOD_Const_Physical, only: rgas
  USE MOD_TimeManager
  USE MOD_MonthlyinSituCO2MaunaLoa, only: get_monthly_co2_mlo
  implicit none

!=== Arguments ===========================================================

  !type (drvdec) ,intent(inout) :: drv              
  !type (tiledec),intent(inout) :: tile(drv%nch)
  !type (clm1d)  ,intent(inout) :: clm (drv%nch)
  integer,intent(in)  :: nx,ny,numpatch,planar_mask(3,nx*ny)
  integer,intent(in)  :: clm_forc_veg                       ! BH: whether vegetation (LAI, SAI, z0m, displa) is being forced 0=no, 1=yes
  real(r8),intent(in) :: sw_pf((nx+2)*(ny+2)*3)             ! SW rad, passed from PF
  real(r8),intent(in) :: lw_pf((nx+2)*(ny+2)*3)             ! LW rad, passed from PF
  real(r8),intent(in) :: prcp_pf((nx+2)*(ny+2)*3)           ! Precip, passed from PF
  real(r8),intent(in) :: tas_pf((nx+2)*(ny+2)*3)            ! Air temp, passed from PF
  real(r8),intent(in) :: u_pf((nx+2)*(ny+2)*3)              ! u-wind, passed from PF
  real(r8),intent(in) :: v_pf((nx+2)*(ny+2)*3)              ! v-wind, passed from PF
  real(r8),intent(in) :: patm_pf((nx+2)*(ny+2)*3)           ! air pressure, passed from PF
  real(r8),intent(in) :: qatm_pf((nx+2)*(ny+2)*3)           ! air specific humidity, passed from PF

  real(r8),intent(in) :: lai_pf((nx+2)*(ny+2)*3)            ! lai, passed from PF !BH
  real(r8),intent(in) :: sai_pf((nx+2)*(ny+2)*3)            ! sai, passed from PF !BH
  real(r8),intent(in) :: z0m_pf((nx+2)*(ny+2)*3)            ! z0m, passed from PF !BH
  real(r8),intent(in) :: displa_pf((nx+2)*(ny+2)*3)         ! displacement height, passed from PF !BH
  !real(r8),intent(in) :: slope_x_pf((nx+2)*(ny+2)*3)        ! slope in x direction, passed from PF !IJB
  !real(r8),intent(in) :: slope_y_pf((nx+2)*(ny+2)*3)        ! slope in y direction, passed from PF !IJB
  integer, intent(in) :: idate(3)


!=== Local Variables =====================================================

  !real(r8) solar     ! incident solar radiation [w/m2]
  real(r8) prcp      ! precipitation [mm/s]
  integer t,i,j,k,l  ! Looping indices
! integer nx,ny      ! Array sizes
  real(r8) :: a, calday                                             ! Julian cal day (1.xx to 365.xx)
  real(r8) :: sunang, cloud, difrat, vnrat
  integer :: year, month, mday
  real(r8) :: pco2m
  
!=== End Variable List ===================================================

!=== Increment Time Step Counter
! clm%istep=clm%istep+1 
! clm%istep=istep_pf
  forc_hgt_u = 50.d0
  forc_hgt_t = 40.d0
  forc_hgt_q = 40.d0

! Valdai - 1D Met data

  ! IMF: modified for 2D
  ! Loop over tile space (convert from pf-to-clm)
    do t = 1, numpatch

        i = planar_mask(1,t)
        j = planar_mask(2,t)
        l = (1+i) + (nx+2)*(j) + (nx+2)*(ny+2)
        forc_swrad(t)          = sw_pf(l)
        forc_frl(t)            = lw_pf(l)
        prcp                   = prcp_pf(l)
        forc_t(t)              = tas_pf(l)
        forc_us(t)             = u_pf(l)
        forc_vs(t)             = v_pf(l)
        forc_psrf(t)           = patm_pf(l)
        forc_q(t)              = qatm_pf(l)
        !clm(t)%slope_x         = slope_x_pf(l)
        !clm(t)%slope_y         = slope_y_pf(l)
        ! BH: added the option for forcing or not the vegetation
        if  (clm_forc_veg == 1) then 
          !clm(t)%elai	        = lai_pf(l)
          !clm(t)%esai	        = sai_pf(l)	
          !clm(t)%z0m	        = z0m_pf(l) 
          !clm(t)%displa	    = displa_pf(l)     
        endif
      
        !Treat air density
        !forc_rhoair(t)  = forc_psrf(t)/(forc_t(t)*2.8704e2)

        IF(forc_t(t) < 180.) forc_t(t) = 180.
        ! the highest air temp was found in Kuwait 326 K, Sulaibya 2012-07-31;
        ! Pakistan, Sindh 2010-05-26; Iraq, Nasiriyah 2011-08-03
        IF(forc_t(t) > 326.) forc_t(t) = 326.

        forc_rhoair(t) = (forc_psrf(t) &
           - 0.378*forc_q(t)*forc_psrf(t)/(0.622+0.378*forc_q(t)))&
           / (rgas*forc_t(t))

        !======================================
        !!Treat solar (SW)
        !forc_sols(t)    = forc_swrad(t)*35./100.   !forc_sols
        !forc_soll(t)    = forc_swrad(t)*35./100.   !forc_soll
        !forc_solsd(t)   = forc_swrad(t)*15./100.   !forc_solsd
        !forc_solld(t)   = forc_swrad(t)*15./100.   !forc_solad
        a = forc_swrad(t)
        IF (isnan(a)) a = 0
        calday = calendarday(idate)
        sunang = orb_coszen (calday, patchlonr(t), patchlatr(t))
        IF (sunang.eq.0) THEN
           cloud = 0.
        ELSE
           cloud = (1160.*sunang-a)/(963.*sunang)
        ENDIF
        cloud = max(cloud,0.0001)
        cloud = min(cloud,1.)
        cloud = max(0.58,cloud)

        difrat = 0.0604/(sunang-0.0223)+0.0683
        IF(difrat.lt.0.) difrat = 0.
        IF(difrat.gt.1.) difrat = 1.

        difrat = difrat+(1.0-difrat)*cloud
        vnrat = (580.-cloud*464.)/((580.-cloud*499.)+(580.-cloud*464.))

        forc_sols(t)  = a*(1.0-difrat)*vnrat
        forc_soll(t)  = a*(1.0-difrat)*(1.0-vnrat)
        forc_solsd(t) = a*difrat*vnrat
        forc_solld(t) = a*difrat*(1.0-vnrat)
        !======================================

        forc_prc(t)     = prcp/3.d0
        forc_prl(t)     = prcp*2.d0/3.d0


        ! [GET ATMOSPHERE CO2 CONCENTRATION DATA]
        year  = idate(1)
        CALL julian2monthday (idate(1), idate(2), month, mday)
        pco2m = get_monthly_co2_mlo(year, month)*1.e-6

        forc_pco2m(t)   = pco2m*forc_psrf(t)
        forc_po2m(t)    = 0.209_r8*forc_psrf(t)
        
        !!Treat precip
        !!(Set upper limit of air temperature for snowfall at 275.65K.
        !! This cut-off was selected based on Fig. 1, Plate 3-1, of Snow
        !! Hydrology (1956)).
        !if (prcp > 0.) then
        !    if(clm(t)%forc_t > (tfrz + tcrit))then
        !      clm(t)%itypprc   = 1
        !      clm(t)%forc_rain = prcp
        !      clm(t)%forc_snow = 0.
        !    else
        !      clm(t)%itypprc   = 2
        !      clm(t)%forc_rain = 0.
        !      clm(t)%forc_snow = prcp
        !    endif
        !else
        !    clm(t)%itypprc      = 0
        !    clm(t)%forc_rain    = 0.
        !    clm(t)%forc_snow    = 0
        !endif
    enddo

end subroutine pf_getforce
