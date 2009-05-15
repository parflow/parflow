!#include <misc.h>

subroutine drv_clm2g (drv,grid,tile,clm)

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
!  Transfer variables from CLM tile space to grid space. 
!  A few examples are given here - the user may need to add more
!  Note that tile to grid space transfers for written output can and
!  are simplified and independent from this subroutine.
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!=========================================================================
! $Id: drv_clm2g.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_module          ! 1-D Land Model Driver variables
  use drv_tilemodule      ! Tile-space variables
  use clmtype             ! 1-D CLM variables
  use drv_gridmodule      ! Grid-space variables
  implicit none

!=== Arguments ===========================================================

  type (drvdec)  :: drv              
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm (drv%nch)
  type (griddec) :: grid(drv%nc,drv%nr)   

!=== End Variable Definition =============================================

  call drv_t2gr(clm%qflx_evap_tot ,grid%qflx_evap_tot ,drv%nc,drv%nr,drv%nch,tile%fgrd,tile%col,tile%row)
  call drv_t2gr(clm%eflx_lh_tot   ,grid%eflx_lh_tot   ,drv%nc,drv%nr,drv%nch,tile%fgrd,tile%col,tile%row)
  call drv_t2gr(clm%eflx_lwrad_out,grid%eflx_lwrad_out,drv%nc,drv%nr,drv%nch,tile%fgrd,tile%col,tile%row)
  call drv_t2gr(clm%t_ref2m       ,grid%t_ref2m       ,drv%nc,drv%nr,drv%nch,tile%fgrd,tile%col,tile%row)
  call drv_t2gr(clm%t_rad         ,grid%t_rad         ,drv%nc,drv%nr,drv%nch,tile%fgrd,tile%col,tile%row)

  return
end subroutine drv_clm2g





