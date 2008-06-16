!#include <misc.h>

subroutine clm_tridia (n, a, b, c, r, u )

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
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_tridia.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  implicit none

!=== Arguments ===========================================================

  integer , intent(in)  :: n
  real(r8), intent(in)  :: a(1:n),b(1:n),c(1:n),r(1:n)
  real(r8), intent(out) :: u(1:n)

!=== Local Variables =====================================================

  integer j
  real(r8) gam(1:n),bet

!=== End Variable List ===================================================

  bet = b(1)
  u(1) = r(1) / bet
  do j = 2, n
     gam(j) = c(j-1) / bet
     bet = b(j) - a(j) * gam(j)
     u(j) = (r(j) - a(j)*u(j-1)) / bet
  enddo

  do j = n-1, 1, -1
     u(j) = u(j) - gam(j+1) * u(j+1)
  enddo

end subroutine clm_tridia
