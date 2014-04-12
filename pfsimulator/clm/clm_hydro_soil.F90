!#include <misc.h>

subroutine clm_hydro_soil (clm) 

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
  !  This is the main subroutine used to execute the calculation of water
  !  processes over soil
  !
  !  (1) Water flow within soil (see clm_soilwater.f90)
  !
  !  (2) Runoff 
  !      The original code was provide by Robert E. Dickinson based on 
  !      following clues:  exponential decrease of Ksat, a water table 
  !      level determination level including highland and lowland levels 
  !      and fractional area of wetland (water table above the surface). 
  !      Runoff is parameterized from the lowlands in terms of precip 
  !      incident on wet areas and a base flow, where these are estimated 
  !      using ideas from TOPMODEL.
  !
  !  The original scheme was modified by Z.-L. Yang and G.-Y. Niu,
  !  *  using a new method to determine water table depth and
  !     the fractional wet area (fcov)
  !  *  computing runoff (surface and subsurface) from this
  !     fraction and the remaining fraction (i.e. 1-fcov)
  !  *  for the 1-fcov part, using BATS1e method to compute
  !     surface and subsurface runoff.
  !
  !   The original code on soil moisture and runoff were provided by 
  !   R. E. Dickinson in July 1996.
  !
  ! REVISION HISTORY:
  !  15 September 1999: Yongjiu Dai; Initial code
  !  12 November 1999:  Z.-L. Yang and G.-Y. Niu
  !  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
  !=========================================================================
  ! $Id: clm_hydro_soil.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision
  use clmtype
  use clm_varcon, only : denice, denh2o, hvap, tfrz
  use clm_varpar, only : nlevsoi
  implicit none

  ! When multiplied by suction*bsw/(water*4.71047e4*t_grnd) gives derivative of
  ! evaporation with respect to water in top soil layer
  ! (= d evap/d rel hum  *d rel hum / d soil suction * d sol suction / d water
  ! and using d evap / d rel humidity = rho * qsat * aerodynamic resistance,
  ! d rel hum / d soil suction = (rel hum)/(4.71047e4*t_grnd)
  ! and d soil suction / d soil water = bsw * (soil suction)/water)

  !=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

  !=== Local Variables =====================================================

  integer i                      ! loop counter

  real(r8) hk(1:nlevsoi),       & ! hydraulic conductivity (mm h2o/s)
       dhkdw(1:nlevsoi),        & ! d(hk)/d(vol_liq)
                                !@dwat(1:nlevsoi),         & ! change in soil water
       fcov,                    & ! fractional area with water table at surface
       s(1:nlevsoi),            & ! wetness of soil (including ice)
       vol_liq(1:nlevsoi),      & ! partial volume of liquid water in layer
       vol_ice(1:nlevsoi),      & ! partial volume of ice lens in layer
       xs,                      & ! excess soil water above saturation
       zwice,                   & ! the sum of ice mass of soil (kg/m2)
       zwt,                     & ! water table depth
       zmm (1:nlevsoi),         & ! layer depth (mm)
       dzmm(1:nlevsoi),         & ! layer thickness (mm)
       watmin                  !&  minimum soil moisture
  !@hksum                      ! summation of hydraulic cond for layers 6->9
  ! Stefan's modification:

  real(r8) &
       s1,                      & ! "s" at interface of layer
       s2                         ! k*s**(2b+2)
  ! For Z.-L. Yang & G.-Y. Niu's modification

  real(r8) zmean               ! The surface soil layers contributing to runoff
  real(r8) wmean               ! The averaged soil wetness in surface soil layers
  real(r8) fz                  ! coefficient for water table depth
  real(r8) qflx_drain_wet      ! subsurface runoff from "wet" part (mm h2o/s)
  real(r8) qflx_drain_dry      ! subsurface runoff from "dry" part (mm h2o/s)

  !=== End Variable List ===================================================

  !=========================================================================
  ! [1] Surface runoff
  !=========================================================================

  ! Porosity of soil, partial volume of ice and liquid

  zwice = 0.
  do i = 1,nlevsoi 
     zwice = zwice + clm%h2osoi_ice(i)
     vol_ice(i) = min(clm%watsat(i), clm%h2osoi_ice(i)/(clm%dz(i)*denice))
     clm%eff_porosity(i) = clm%watsat(i)-vol_ice(i)
     vol_liq(i) = min(clm%eff_porosity(i), clm%h2osoi_liq(i)/(clm%dz(i)*denh2o))
  enddo

  ! Calculate wetness of soil
  do i = 1,nlevsoi
     s(i) = min(dble(1.),(vol_ice(i)+vol_liq(i))/clm%watsat(i))
  end do

  !write(20,*) s(:nlevsoi)

  ! Determine water table 

  wmean = 0.                                                  
  fz    = 1.0                                                
  do i  = 1, nlevsoi                                        
     wmean = wmean + s(i)*clm%dz(i)                          
  enddo
  zwt = fz * (clm%zi(nlevsoi) - wmean)                   

  ! Saturation fraction

  fcov = clm%wtfact*min(dble(1.),exp(-zwt))

  ! Currently no overland flow parameterization in code is considered
  ! qflx_surf = 0.   Zong-Liang Yang & G.-Y. Niu
  !*modified surface runoff according to the concept of TOPMODEL 

  wmean = 0.                                               
  zmean = 0.                                              
  do i = 1, 3                                          
     zmean = zmean + clm%dz(i)                          
     wmean = wmean + s(i) * clm%dz(i)                 
  enddo
  wmean = wmean / zmean
  clm%qflx_surf = 0.0d0
  clm%qflx_infl = clm%qflx_top_soil - clm%qflx_evap_grnd


  ! Add in hillslope runoff
  ! qflx_surf = qflx_surf+ max(0.,clm%qflx_top_soil*fcov)                            
  ! Infiltration into surface soil layer (minus the evaporation)
  ! and the derivative of evaporation with respect to water in top soil layer

  !  sdamp = 0.
  !  if (clm%snl+1 >= 1) then
  !     clm%qflx_infl = clm%qflx_top_soil - clm%qflx_surf - clm%qflx_evap_grnd
  !     wat1 = clm%h2osoi_liq(1)             ! equals theta*clm%dz(nlevlak)*1000. (mm)
  !     if (clm%qflx_evap_grnd > 0. .AND. wat1 >= 1.e-3) then
  !        sdampmax = clm%sfactmax/clm%dz(1)/1000.
  !        s_node = max(vol_liq(1)/clm%watsat(1),0.01)
  !        smp_node = max(clm%smpmin, -clm%sucsat(1)*s_node**(-clm%bsw(1)))
  !        sdamp = min(-clm%sfact*smp_node/wat1, sdampmax)
  !     endif
  !  else
  !     clm%qflx_infl = clm%qflx_top_soil - clm%qflx_surf
  !  endif

  !=========================================================================
  ! [2] Set up r, a, b, and c vectors for tridiagonal solution and renew bl
  !=========================================================================

  ! Following length units are all in millimeters

  do i = 1,nlevsoi 
     zmm(i) = clm%z(i)*1.e3
     dzmm(i) = clm%dz(i)*1.e3
  enddo

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Instead of calling the subroutine to calc soilwater distribution
  ! we introduced the Parflow couple, which is called after clm_main.
  ! That way clm_soilwater becomes obsolete
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  ! LEGACY 
  ! call clm_soilwater (vol_liq, clm%eff_porosity, clm%qflx_infl, sdamp, &
  !                     dwat   , hk              , dhkdw        , clm)

  ! Set zero to hydraulic conductivity if effective porosity 5% in any of 
  ! two neighbor layers or liquid content (theta) less than 0.001

  do i = 1, nlevsoi
     if (      (clm%eff_porosity(i) < clm%wimp) &
          .OR. (clm%eff_porosity(min(nlevsoi,i+1)) < clm%wimp) &
          .OR. (clm%pf_vol_liq(i) <= 1.e-3))then
        hk(i) = 0.
        dhkdw(i) = 0.
     else
        s1 = 0.5*(vol_liq(i)+vol_liq(min(nlevsoi,i+1))) / &
             (0.5*(clm%watsat(i)+clm%watsat(min(nlevsoi,i+1))))
        s2 = clm%hksat(i)*s1**(2.*clm%bsw(i)+2.)
        hk(i) = s1*s2  
        dhkdw(i) = (2.*clm%bsw(i)+3.)*s2*0.5/clm%watsat(i)
        if(i == nlevsoi) dhkdw(i) = dhkdw(i) * 2.
     endif
  enddo


  ! Renew the mass of liquid water
  !  do i= 1,nlevsoi 
  !     clm%h2osoi_liq(i) = clm%h2osoi_liq(i) + dwat(i)*dzmm(i)
  !	 clm%h2osoi_liq(i) = vol_liq(i)*clm%dz(i)*denh2o
  !  enddo

  !=========================================================================
  ! [3] Streamflow and total runoff
  !=========================================================================

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  ! Instead of using parameterized subsurface runoff (drainage), all
  ! subsurface flow is calculated by ParFlow. The code below is kept as 
  ! legacy only -- drainage fluxes are set to zero. 
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  clm%qflx_drain = 0.                      ! subsurface runoff
  qflx_drain_wet = 0.                      ! subsurface runoff
  qflx_drain_dry = 0.                      ! subsurface runoff

  !=== LEGACY
  ! The amount of streamflow is assumed maintained by flow from the 
  ! lowland water table with different levels contributing according to 
  ! their thickness and saturated hydraulic conductivity, i.e. a given 
  ! layer below the water table interface loses water at a rate per unit 
  ! depth given by qflx_drain*hk/(sum over all layers below this water table 
  ! of hk*dz). Because this is a slow smooth process, and not strongly 
  ! coupled to water in any one layer, it should remain stable for 
  ! explicit time differencing. Hence, for simplicity it is removed
  ! explicitly prior to the main soil water calculation.
  ! Another assumption: no subsurface runoff for ice mixed soil 
  ! Zong-Liang Yang & G.-Y. Niu                                         
  ! 
  ! hksum = 0.
  ! do i = 6,nlevsoi-1                  
  !    hksum = hksum + hk(i)
  ! enddo
  ! if (zwice <= 0. .AND. hksum > 0.) then
  !    zsat = 0.                                                    
  !    wsat = 0.                                                    
  !    dzksum = 0.                                                  
  !    do i = 6,nlevsoi-1                  
  !       zsat = zsat + clm%dz(i)*hk(i)                               
  !       wsat = wsat + s(i)*clm%dz(i)*hk(i)                         
  !       dzksum  = dzksum   + hk(i)*clm%dz(i)                       
  !    enddo
  !
  !    wsat = wsat / zsat                                         
  !
  !    qflx_drain_dry = (1.-fcov)*4.e-2* wsat ** (2.*clm%bsw(1)+3.)  ! mm/s
  !    qflx_drain_wet = fcov * 1.e-5 * exp(-zwt)                     ! mm/s
  !    clm%qflx_drain = qflx_drain_dry + qflx_drain_wet
  !
  !    do i = 6, nlevsoi-1                 
  !       clm%h2osoi_liq(i) = clm%h2osoi_liq(i) &
  !                           - clm%dtime*clm%qflx_drain*clm%dz(i)*hk(i)/dzksum                                  
  !    enddo
  ! endif

  ! --------------------------------------------------------------------
  ! Limit h2osoi_liq to be greater than or equal to watmin. 
  ! Get water needed to bring h2osoi_liq equal watmin from lower layer. 
  ! --------------------------------------------------------------------

  watmin = 0.0
  ! do i = 1, nlevsoi-1
  !     if (clm%h2osoi_liq(i) < 0.) then
  !        xs = watmin-clm%h2osoi_liq(i)
  !     else
  xs = 0.
  !     end if
  !    clm%h2osoi_liq(i  ) = clm%h2osoi_liq(i  ) + xs
  !    clm%h2osoi_liq(i+1) = clm%h2osoi_liq(i+1) - xs
  ! end do
  ! i = nlevsoi
  ! if (clm%h2osoi_liq(i) < watmin) then
  !    xs = watmin-clm%h2osoi_liq(i)
  !  else
  xs = 0.
  ! end if
  ! clm%h2osoi_liq(i) = clm%h2osoi_liq(i) + xs
  ! clm%qflx_drain = clm%qflx_drain - xs/clm%dtime

  ! Determine water in excess of saturation

  xs = max(dble(0.), clm%h2osoi_liq(1)-(clm%pondmx+clm%eff_porosity(1)*dzmm(1)))
  !@ I implement a warning here because "pondmx" is a empirical factor we don't really know/use 
  !@  if (xs > 0.) then
  !@   write(20,*)"TROUBLE: Ponding in individual cell"
  !@   clm%h2osoi_liq(1) = clm%pondmx+clm%eff_porosity(1)*dzmm(1)
  !@  endif   

  !do i = 2,nlevsoi 
  !  xs = xs + max(clm%h2osoi_liq(i)-clm%eff_porosity(i)*dzmm(i), 0.)     ! [mm]
  !  clm%h2osoi_liq(i) = min(clm%eff_porosity(i)*dzmm(i), clm%h2osoi_liq(i))
  !enddo

  ! Sub-surface runoff and drainage 

  clm%qflx_drain = 0.0d0 !clm%qflx_drain + xs/clm%dtime  &
  !+ hk(nlevsoi) + dhkdw(nlevsoi)*dwat(nlevsoi) ! [mm/s]

  ! set imbalance for glacier, lake and wetland to 0. 

  clm%qflx_qrgwl = 0.   !only set for lakes, wetlands and glaciers 

  ! for now set implicit evaporation to zero

  clm%eflx_impsoil = 0.

  ! Renew the ice and liquid mass due to condensation

  if (clm%snl+1 >= 1) then
     !     clm%h2osoi_liq(1) = clm%h2osoi_liq(1) + clm%qflx_dew_grnd*clm%dtime
     clm%h2osoi_ice(1) = clm%h2osoi_ice(1) + (clm%qflx_dew_snow-clm%qflx_sub_snow)*clm%dtime
  endif

end subroutine clm_hydro_soil




