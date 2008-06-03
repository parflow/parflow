!#include <misc.h>

subroutine drv_astp(ierr)

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
!  Allocate error check
!
! REVISION HISTORY:
!  15 December 1999:  Paul Houser and Jon Radakovich; Initial Code 
!   3 March 2000:     Jon Radakovich; Revision for diagnostic output
!  22 November 2000: Mariana Vertenstein
!=========================================================================
! $Id: drv_astp.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  implicit none
  integer ierr
  if(ierr /= 0) then
     write(6,*) 'Allocation request denied'
  endif

end subroutine drv_astp






