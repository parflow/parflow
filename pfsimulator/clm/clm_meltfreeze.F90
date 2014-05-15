!#include <misc.h>

subroutine clm_meltfreeze (fact,     brr,       hs,     dhsdT,  &   
                           tssbef,   xmf,       clm) 

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
!  Calculation of the phase change within snow and soil layers:
!
!  (1) Check the conditions for which the phase change may take place, 
!      i.e., the layer temperature is great than the freezing point 
!      and the ice mass is not equal to zero (i.e. melting), 
!      or the layer temperature is less than the freezing point 
!      and the liquid water mass is not equal to zero (i.e. freezing).
!  (2) Assess the rate of phase change from the energy excess (or deficit) 
!      after setting the layer temperature to freezing point.
!  (3) Re-adjust the ice and liquid mass, and the layer temperature
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_meltfreeze.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varcon, only : tfrz, hfus
  use clm_varpar, only : nlevsoi
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm   ! CLM 1-D Module

  real(r8), intent(in) ::     &
       tssbef(clm%snl+1 : nlevsoi),   & ! temperature at previous time step [K]
       brr   (clm%snl+1 : nlevsoi),   & ! 
       fact  (clm%snl+1 : nlevsoi),   & ! temporary variables
       hs,                            & ! net ground heat flux into the surface
       dhsdT                            ! temperature derivative of "hs"

  real(r8), intent(out) ::    &
       xmf                              ! total latent heat of phase change

!=== Local Variables =====================================================

  integer j
  real(r8)  hm(clm%snl+1 : nlevsoi),     & ! energy residual [W/m2]
       xm(clm%snl+1 : nlevsoi),       & ! melting or freezing within a time step [kg/m2]
       heatr,                         & ! energy residual or loss after melting or freezing
       temp1                            ! temporary variables [kg/m2]

  real(r8), dimension(clm%snl+1 : nlevsoi) :: wmass0, wice0, wliq0
  real(r8)  propor,tinc  

!=== End Variable List ===================================================

! Initial 

  clm%qflx_snomelt = 0.
  xmf = 0.
  do j = clm%snl+1, nlevsoi
     clm%imelt(j) = 0
     hm(j) = 0.
     xm(j) = 0.
     wice0(j) = clm%h2osoi_ice(j)
     wliq0(j) = clm%h2osoi_liq(j)
     wmass0(j) = clm%h2osoi_ice(j) + clm%h2osoi_liq(j)
  enddo

! Melting identification
! If ice exists above melt point, melt some to liquid.

  do j = clm%snl+1, nlevsoi
     if (clm%h2osoi_ice(j) > 0. .AND. clm%t_soisno(j) > tfrz) then
        clm%imelt(j) = 1
        clm%t_soisno(j) = tfrz
     endif

     ! Freezing identification
     ! If liquid exists below melt point, freeze some to ice.

     if (clm%h2osoi_liq(j) > 0. .AND. clm%t_soisno(j) < tfrz) then
        clm%imelt(j) = 2
        clm%t_soisno(j) = tfrz
     endif
  enddo

! If snow exists, but its thickness is less than the critical value (0.01 m)

  if (clm%snl+1 == 1 .AND. clm%h2osno > 0.) then
     if (clm%t_soisno(1) > tfrz) then
        clm%imelt(1) = 1
        clm%t_soisno(1) = tfrz
     endif
  endif

! Calculate the energy surplus and loss for melting and freezing

  do j = clm%snl+1, nlevsoi
     if (clm%imelt(j) > 0) then
        tinc = clm%t_soisno(j)-tssbef(j)
        if (j > clm%snl+1) then
           hm(j) = brr(j) - tinc/fact(j) 
        else
           hm(j) = hs + dhsdT*tinc + brr(j) - tinc/fact(j) 
        endif
     endif
  enddo

! These two errors were checked carefully.  They result from the the computed error
! of "Tridiagonal-Matrix" in subroutine "thermal".

  do j = clm%snl+1, nlevsoi
     if (clm%imelt(j) == 1 .AND. hm(j) < 0.) then
        hm(j) = 0.
        clm%imelt(j) = 0
     endif

     if (clm%imelt(j) == 2 .AND. hm(j) > 0.) then
        hm(j) = 0.
        clm%imelt(j) = 0
     endif
  enddo

! The rate of melting and freezing

  do j = clm%snl+1, nlevsoi

     if (clm%imelt(j) > 0 .and. abs(hm(j)) > .0) then
        xm(j) = hm(j)*clm%dtime/hfus                        ! kg/m2

        ! If snow exists, but its thickness is less than the critical value (1 cm)
        ! Note: more work is needed to determine how to tune the snow depth for this case

        if (j == 1) then
           if ((clm%snl+1 == 1) .AND. (clm%h2osno > 0.) .AND. (xm(j) > 0.)) then
              temp1 = clm%h2osno                                        ! kg/m2
              clm%h2osno = max(dble(0.),temp1-xm(j))
              propor = clm%h2osno/temp1
              clm%snowdp = propor * clm%snowdp
              heatr = hm(j) - hfus*(temp1-clm%h2osno)/clm%dtime         ! W/m2
              if (heatr > 0.) then
                 xm(j) = heatr*clm%dtime/hfus                           ! kg/m2
                 hm(j) = heatr                                          ! W/m2
              else
                 xm(j) = 0.
                 hm(j) = 0.
              endif
              clm%qflx_snomelt = max(dble(0.),(temp1-clm%h2osno))/clm%dtime   ! kg/(m2 s)
              xmf = hfus*clm%qflx_snomelt
           endif
        endif

        heatr = 0.
        if (xm(j) > 0.) then
           clm%h2osoi_ice(j) = max(dble(0.), wice0(j)-xm(j))
           heatr = hm(j) - hfus*(wice0(j)-clm%h2osoi_ice(j))/clm%dtime
        else if (xm(j) < 0.) then
           clm%h2osoi_ice(j) = min(wmass0(j), wice0(j)-xm(j))
           heatr = hm(j) - hfus*(wice0(j)-clm%h2osoi_ice(j))/clm%dtime  
        endif

        clm%h2osoi_liq(j) = max(dble(0.),wmass0(j)-clm%h2osoi_ice(j))

        if (abs(heatr) > 0.) then
           if (j > clm%snl+1) then
              clm%t_soisno(j) = clm%t_soisno(j) + fact(j)*heatr
           else
              clm%t_soisno(j) = clm%t_soisno(j) + fact(j)*heatr/(1.-fact(j)*dhsdT)
           endif
           if (clm%h2osoi_liq(j)*clm%h2osoi_ice(j)>0.) clm%t_soisno(j) = tfrz
        endif

        xmf = xmf + hfus * (wice0(j)-clm%h2osoi_ice(j))/clm%dtime

        if (clm%imelt(j) == 1 .AND. j < 1) then
           clm%qflx_snomelt = clm%qflx_snomelt + max(dble(0.),(wice0(j)-clm%h2osoi_ice(j)))/clm%dtime  
        endif

     endif

  enddo

! needed for output to lsm history file

  clm%eflx_snomelt = clm%qflx_snomelt * hfus  

end subroutine clm_meltfreeze
