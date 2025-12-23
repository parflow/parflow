!#include <misc.h>

subroutine clm_subdiv (clm)

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
!  Subdivides snow layers if they exceed their 
!  prescribed maximum thickness.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_subdiv.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm !CLM 1-D Module

!=== Local Variables =====================================================

  real(r8) drr,             & ! thickness of the combined [m] 
       dzsno(5),            & ! Snow layer thickness [m] 
       swice(5),            & ! Partial volume of ice [m3/m3]
       swliq(5),            & ! Partial volume of liquid water [m3/m3]
       tsno(5)                ! Nodal temperature [K]

  integer k,                & ! number of do looping
       msno                   ! number of snow layer 1 (top) to msno (bottom)

  real(r8) zwice,zwliq,propor

!=== End Variable List ===================================================

  msno = abs(clm%snl)
  do k = 1, msno
     dzsno(k) = clm%dz  (k + clm%snl)
     swice(k) = clm%h2osoi_ice(k + clm%snl)
     swliq(k) = clm%h2osoi_liq(k + clm%snl)
     tsno(k)  = clm%t_soisno (k + clm%snl)
  enddo

  if (msno == 1) then
     if (dzsno(1) > 0.03) then
        msno = 2

! Specified a new snow layer
        dzsno(1) = dzsno(1)/2.
        swice(1) = swice(1)/2.
        swliq(1) = swliq(1)/2.
        dzsno(2) = dzsno(1)
        swice(2) = swice(1)
        swliq(2) = swliq(1)
        tsno(2)  = tsno(1)
!       write(6,*)'Subdivided Top Node into two layer (1/2)'
     endif
  endif

  if (msno > 1) then
     if (dzsno(1) > 0.02) then
        drr = dzsno(1) - 0.02
        propor = drr/dzsno(1)
        zwice = propor*swice(1)
        zwliq = propor*swliq(1)
        propor = 0.02/dzsno(1)
        swice(1) = propor*swice(1)
        swliq(1) = propor*swliq(1)
        dzsno(1) = 0.02

        call clm_combo(dzsno(2),swliq(2),swice(2),tsno(2), &
             drr,zwliq,zwice,tsno(1))

        if (msno <= 2 .AND. dzsno(2) > 0.07) then
! Subdivided a new layer
           msno = 3
           dzsno(2) = dzsno(2)/2.
           swice(2) = swice(2)/2.
           swliq(2) = swliq(2)/2.
           dzsno(3) = dzsno(2)
           swice(3) = swice(2)
           swliq(3) = swliq(2)
           tsno(3)  = tsno(2)
        endif
     endif
  endif

  if (msno > 2) then
     if (dzsno(2) > 0.05) then
        drr = dzsno(2) - 0.05
        propor = drr/dzsno(2)
        zwice = propor*swice(2)
        zwliq = propor*swliq(2)
        propor = 0.05/dzsno(2)
        swice(2) = propor*swice(2)
        swliq(2) = propor*swliq(2)
        dzsno(2) = 0.05

        call clm_combo(dzsno(3),swliq(3),swice(3),tsno(3), &
             drr,zwliq,zwice,tsno(2))

        if (msno <= 3 .AND. dzsno(3) > 0.18) then

! Subdivided a new layer
           msno =  4
           dzsno(3) = dzsno(3)/2.
           swice(3) = swice(3)/2.
           swliq(3) = swliq(3)/2.
           dzsno(4) = dzsno(3)
           swice(4) = swice(3)
           swliq(4) = swliq(3)
           tsno(4)  = tsno(3)
        endif
     endif
  endif

  if (msno > 3) then
     if (dzsno(3) > 0.11) then
        drr = dzsno(3) - 0.11
        propor = drr/dzsno(3)
        zwice = propor*swice(3)
        zwliq = propor*swliq(3)
        propor = 0.11/dzsno(3)
        swice(3) = propor*swice(3)
        swliq(3) = propor*swliq(3)
        dzsno(3) = 0.11

        call clm_combo(dzsno(4),swliq(4),swice(4),tsno(4), &
             drr,zwliq,zwice,tsno(3))

        if (msno <= 4 .AND. dzsno(4) > 0.41) then

! Subdivided a new layer
           msno = 5
           dzsno(4) = dzsno(4)/2.
           swice(4) = swice(4)/2.
           swliq(4) = swliq(4)/2.
           dzsno(5) = dzsno(4)
           swice(5) = swice(4)
           swliq(5) = swliq(4)
           tsno(5)  = tsno(4)
        endif
     endif
  endif

  if (msno > 4) then
     if (dzsno(4) > 0.23) then 
        drr = dzsno(4) - 0.23
        propor = drr/dzsno(4)
        zwice = propor*swice(4)
        zwliq = propor*swliq(4)
        propor = 0.23/dzsno(4)
        swice(4) = propor*swice(4)
        swliq(4) = propor*swliq(4)
        dzsno(4) = 0.23

        call clm_combo(dzsno(5),swliq(5),swice(5),tsno(5), &
             drr,zwliq,zwice,tsno(4))

     endif
  endif

  clm%snl = - msno

  do k = clm%snl+1, 0
     clm%dz(k)   = dzsno(k - clm%snl)
     clm%h2osoi_ice(k) = swice(k - clm%snl)
     clm%h2osoi_liq(k) = swliq(k - clm%snl)
     clm%t_soisno(k)  = tsno (k - clm%snl)
  enddo

  do k = 0, clm%snl+1, -1
     clm%z(k)    = clm%zi(k) - 0.5*clm%dz(k)
     clm%zi(k-1) = clm%zi(k) - clm%dz(k)
  enddo

end subroutine clm_subdiv
