!#include <misc.h>

subroutine clm_hydro_irrig (clm)

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
! Irrigate crops to depth of 1 m and modify runoff accordingly
!
! REVISION HISTORY:
!  7 November 2000: Mariana Vertenstein; Initial code
!
!=========================================================================
! $Id: clm_hydro_irrig.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use clmtype
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : denh2o
  implicit none

!=== Arguments ===========================================================

  type (clm1d), intent(inout) :: clm	 !CLM 1-D Module

!=== Local Variables =====================================================

  integer i        !index
  real(r8) wtold   !temporary var
  real(r8) wtnew   !temporary var
  real(r8) temp    !temporary var 

!=========================================================================

! To saturate the soil, make the liquid water volume equal the minimum of 
! the effective porosity and watopt. Irrigate soil to a depth of 30 cm.

  clm%qflx_qirr = 0.
  if (clm%irrig .and. (clm%elai+clm%esai)>0) then
     wtold = 0.
     wtnew = 0.
     do i = 1, nlevsoi
        if (clm%zi(i) <= 0.3) then
           wtold = wtold + clm%h2osoi_liq(i)
           temp = min(clm%watopt(i),clm%eff_porosity(i))
           clm%h2osoi_liq(i) = max(temp*clm%dz(i)*denh2o,clm%h2osoi_liq(i))
           wtnew = wtnew + clm%h2osoi_liq(i)
        end if
     end do
     clm%qflx_qirr = (wtnew - wtold) / clm%dtime
  end if

end subroutine clm_hydro_irrig





