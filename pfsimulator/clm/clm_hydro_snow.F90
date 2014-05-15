!#include <misc.h>

subroutine clm_hydro_snow(clm)

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
!  Evaluate the change of snow mass and the snow water onto soil.
!  Water flow within snow is computed by an explicit and non-physical 
!  based scheme, which permits a part of liquid water over the holding 
!  capacity (a tentative value is used, i.e. equal to 0.033*porosity) to
!  percolate into the underlying layer.  Except for cases where the 
!  porosity of one of the two neighboring layers is less than 0.05, zero 
!  flow is assumed. The water flow out of the bottom of the snow pack will 
!  participate as the input of the soil water and runoff. 
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!  15 November 2000: Mariana Vertenstein
!=========================================================================
! $Id: clm_hydro_snow.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : denh2o, denice
  implicit none

!=== Arguments ===========================================================  

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

!=== Local Variables =====================================================

  integer  j                         ! do loop/array indices

  real(r8) qin,                    & ! water flow into the elmement (mm/s)
       qout,                       & ! water flow out of the elmement (mm/s)
       wgdif,                      & ! ice mass after minus sublimation
       vol_liq(clm%snl+1 : 0),     & ! partial volume of liquid water in layer
       vol_ice(clm%snl+1 : 0),     & ! partial volume of ice lens in layer
       eff_porosity(clm%snl+1: 0), & ! effective porosity = porosity - vol_ice
       qout_snowb                    ! rate of water out of snow bottom (mm/s)

!=== End Variable List ===================================================

  qout = 0.

  if (clm%snl+1 >=1) then

     clm%qflx_top_soil = clm%qflx_rain_grnd + clm%qflx_snomelt

  else

     ! Renew the mass of ice lens (h2osoi_ice) and liquid (h2osoi_liq) in the
     ! surface snow layer, resulting from sublimation (frost) / evaporation (condense)

     wgdif = clm%h2osoi_ice(clm%snl+1) + (clm%qflx_dew_snow - clm%qflx_sub_snow)*clm%dtime
     clm%h2osoi_ice(clm%snl+1) = wgdif
     if (wgdif < 0.) then
        clm%h2osoi_ice(clm%snl+1) = 0.
        clm%h2osoi_liq(clm%snl+1) = clm%h2osoi_liq(clm%snl+1) + wgdif
     endif
     clm%h2osoi_liq(clm%snl+1) = clm%h2osoi_liq(clm%snl+1) +  &
          (clm%qflx_rain_grnd + clm%qflx_dew_grnd - clm%qflx_evap_grnd)*clm%dtime
     clm%h2osoi_liq(clm%snl+1) = max(dble(0.), clm%h2osoi_liq(clm%snl+1))
     
     ! Porosity and partial volume
     
     do j = clm%snl+1, 0
        vol_ice(j) = min(dble(1.), clm%h2osoi_ice(j)/(clm%dz(j)*denice))
        eff_porosity(j) = 1. - vol_ice(j)
        vol_liq(j) = min(eff_porosity(j),clm%h2osoi_liq(j)/(clm%dz(j)*denh2o))
     enddo
     
     ! Capillary forces within snow are usually two or more orders of magnitude
     ! less than those of gravity. Only gravity terms are considered. 
     ! the genernal expression for water flow is "K * ss**3", however, 
     ! no effective parameterization for "K".  Thus, a very simple consideration 
     ! (not physically based) is introduced: 
     ! when the liquid water of layer exceeds the layer's holding 
     ! capacity, the excess meltwater adds to the underlying neighbor layer.
     
     qin = 0.
     do j= clm%snl+1, 0
        clm%h2osoi_liq(j) = clm%h2osoi_liq(j) + qin
        if (j <= -1) then
           ! No runoff over snow surface, just ponding on surface
           if (eff_porosity(j)<clm%wimp .OR. eff_porosity(j+1)<clm%wimp) then
              qout = 0.
           else
              qout = max(dble(0.),(vol_liq(j)-clm%ssi*eff_porosity(j))*clm%dz(j))
              qout = min(qout,(1.-vol_ice(j+1)-vol_liq(j+1))*clm%dz(j+1))
           endif
        else
           qout = max(dble(0.),(vol_liq(j)-clm%ssi*eff_porosity(j))*clm%dz(j))
        endif
        qout = qout*1000.
        clm%h2osoi_liq(j) = clm%h2osoi_liq(j) - qout
        qin = qout
     enddo
     
     qout_snowb = qout/clm%dtime
     clm%qflx_top_soil = qout_snowb
     
  endif

end subroutine clm_hydro_snow
