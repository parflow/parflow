!#include <misc.h>

subroutine clm_snowage (clm)

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
!  Updates snow cover and snow age, based on BATS code.
!
! REVISION HISTORY:
!  Original Code:  Robert Dickinson
!  15 September 1999: Yongjiu Dai; Integration of code into CLM
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: clm_snowage.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

! Declare Modules and data structures

  use precision
  use clmtype
  use clm_varcon, only : tfrz
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm  !CLM 1-D Module

!=== Local Variables =====================================================

  real(r8)                 & !
       age1,               & ! snow aging factor due to crystal growth [-]
       age2,               & ! snow aging factor due to surface growth [-]
       age3,               & ! snow aging factor due to accum of other particles [-]
       arg,                & ! temporary variable used in snow age calculation [-]
       arg2,               & ! temporary variable used in snow age calculation [-]
       dela,               & ! temporary variable used in snow age calculation [-]
       dels,               & ! temporary variable used in snow age calculation [-]
       sge                   ! temporary variable used in snow age calculation [-]

!=== End Variable List ===================================================

  if (clm%h2osno <= 0.) then

     clm%snowage = 0.

! RMM removed arbitrary snow age for SWE >=800; we have places in the US / CO / CA that have 800 mm of SWE and this 
! cutoff caused issues when the SWE passed that amount
  !else if (clm%h2osno > 800.) then   ! Over Antarctica

  !   clm%snowage = 0.

  else                               !! RMM removed ! Away from Antarctica 

     age3  = 0.3
     arg   = 5.e3*(1./tfrz-1./clm%t_grnd)
     arg2  = min(dble(0.),dble(10.)*arg)
     age2  = exp(arg2)
     age1  = exp(arg)
     dela  = 1.e-6*clm%dtime*(age1+age2+age3)
     dels  = 0.1*max(dble(0.0), clm%h2osno-clm%h2osno_old)
     sge   = (clm%snowage+dela)*(1.0-dels)
     clm%snowage   = max(dble(0.0),sge)

  endif

end subroutine clm_snowage
