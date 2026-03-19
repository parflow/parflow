!#include <misc.h>
!
! DEPRECATED: This subroutine is never called in ParFlow-CLM.
! It writes ALMA standard output to Fortran unit 57, but no caller
! opens this unit or invokes drv_almaout. The subroutine is retained
! as an empty stub to avoid link errors from CMakeLists.txt.
!
! This was the only consumer of clm%smpmax in active computation
! (via the swetwilt/SoilWet calculation). With the body removed,
! smpmax is fully dead code.
!
! Original description: ALMA standard output (Radakovich, Dec 2000)
!

subroutine drv_almaout (drv,tile,clm)

! Declare Modules and data structures — kept for interface compatibility
  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables

  implicit none
  type (drvdec)       :: drv
  type (tiledec)      :: tile(drv%nch)
  type (clm1d)        :: clm(drv%nch)

  ! Body removed — subroutine is never called in ParFlow-CLM.
  ! See git history for original ALMA output implementation.

end subroutine drv_almaout
