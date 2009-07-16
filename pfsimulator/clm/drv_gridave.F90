!#include <misc.h>

function drv_gridave (nch, mask, fgr, var, drv)
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
  ! The following function simply averages tile spatial arrays.
  !
  ! REVISION HISTORY:
  !  6 May 1999: Paul Houser; initial code
  !  7 Dec 2000: Mariana Vertenstein
  !=========================================================================
  ! $Id: drv_gridave.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision           !@ Stefan
  use drv_module          ! 1-D Land Model Driver variables
  implicit none

  !=== Arguments =============================================================

  type (drvdec):: drv                !@ Stefan
  integer , intent(in) :: nch        ! number of tiles
  integer , intent(in) :: mask(nch)  ! mask (1=not used, 0=used)
  real(r8), intent(in) :: fgr(nch)   ! Fraction of veg class in grid
  real(r8), intent(in) :: var(nch)   ! CLM Variable to average

  !=== Local Variables =====================================================

  integer  :: t                ! tile index
  integer  :: counter          !@ counter of sctive cells
  real(r8) :: drv_gridave      ! Spatial average function  

  !=== End Variable List ===================================================

  drv_gridave = 0.0
  counter = 0
  do t = 1,nch
     if (mask(t) == 1) then
        counter = counter + 1
        drv_gridave = drv_gridave + var(t)*fgr(t)
     endif
  enddo

  !@ Stefan: Arithmetic mean of var(t) over entire area; output is var(t) per unit area of domain
  !@ Assumption: dx = dy !
  drv_gridave = drv_gridave/counter                           !@dfloat(drv%nc*drv%nr)
  if (abs(drv_gridave) < 1.0d-99) drv_gridave = 0.0e0  

end function drv_gridave

real*8 function variance(nch,var)
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
  ! The following function simply averages tile spatial arrays.
  !
  ! REVISION HISTORY:
  !  6 May 1999: Paul Houser; initial code
  !  7 Dec 2000: Mariana Vertenstein
  !=========================================================================
  ! $Id: drv_gridave.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use precision           !@ Stefan
  use drv_module          ! 1-D Land Model Driver variables
  implicit none

  !=== Arguments =============================================================

  integer , intent(in) :: nch        ! number of tiles
  real(r8), intent(in) :: var(nch)   ! CLM Variable to average

  !=== Local Variables =====================================================

  !=== End Variable List ===================================================

  variance = (nch*sum(var**2)-(sum(var))**2)/(nch*(nch-1))


end function variance















