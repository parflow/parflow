!#include <misc.h>

function clm_psi(k,zeta)

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
!  Stability function for rib < 0.
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================

! Declare Modules and data structures

  use precision
  implicit none

!=== Local Variables =====================================================      

  integer k        !
  real(r8) zeta, & ! dimensionless height used in Monin-Obukhov theory
       clm_psi,  & ! stability function for unstable case
       chik        ! 

!=== End Variable List ===================================================

  chik = (1.-16.*zeta)**0.25
  if (k == 1) then
     clm_psi = 2.*log((1.+chik)*0.5) &
          + log((1.+chik*chik)*0.5)-2.*atan(chik)+2.*atan(1.)
  else
     clm_psi = 2.*log((1.+chik*chik)*0.5)
  endif

end function clm_psi
