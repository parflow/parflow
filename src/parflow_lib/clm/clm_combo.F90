!#include <misc.h>

subroutine clm_combo (dz,     wliq,     wice,     t,            &
                      dz2,    wliq2,    wice2,    t2            ) 

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
!  Combines two elements and returns the following combined
!  variables: dz, t, wliq, wice. 
!  The combined temperature is based on the equation:
!  the sum of the enthalpies of the two elements =  
!  that of the combined element.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_combo.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clm_varcon,  only : cpice, cpliq, tfrz, hfus         
  implicit none

!=== Arguments ===========================================================

  real(r8), intent(inout) :: &
       dz,                & ! nodal thickness of 1 elements being combined [m]
       wliq,              & ! liquid water of element 1
       wice,              & ! ice of element 1 [kg/m2]
       t                    ! nodel temperature of elment 1 [K]

  real(r8), intent(in) ::    &
       dz2,               & ! nodal thickness of 2 elements being combined [m]
       wliq2,             & ! liquid water of element 2 [kg/m2]
       wice2,             & ! ice of element 2 [kg/m2]
       t2                   ! nodal temperature of element 2 [K]

!=== Local Variables =====================================================

  real(r8) dzc,              & ! Total thickness of nodes 1 and 2 (dzc=dz+dz2).
       wliqc,             & ! Combined liquid water [kg/m2]
       wicec,             & ! Combined ice [kg/m2]
       tc,                & ! Combined node temperature [K]
       h,                 & ! enthalpy of element 1 [J/m2]
       h2,                & ! enthalpy of element 2 [J/m2]
       hc                   ! temporary

!=== End Variable List ===================================================

  dzc = dz+dz2
  wicec = (wice+wice2)
  wliqc = (wliq+wliq2)
  h =(cpice*wice+cpliq*wliq)* &
       (t-tfrz)+hfus*wliq
  h2=(cpice*wice2+cpliq*wliq2)* &
       (t2-tfrz)+hfus*wliq2

  hc = h + h2
  if(hc < 0.)then
     tc = tfrz + hc/(cpice* &
          wicec+cpliq*wliqc)
  else if(hc.le.hfus*wliqc)then
     tc = tfrz
  else
     tc = tfrz + (hc - hfus*wliqc)/ &
          (cpice*wicec+cpliq*wliqc)
  endif

  dz = dzc
  wice = wicec 
  wliq = wliqc
  t = tc

end subroutine clm_combo
