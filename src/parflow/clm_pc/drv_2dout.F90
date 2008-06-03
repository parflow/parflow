!#include <misc.h>

subroutine drv_2dout (casc, drv, tile, clm)

!=========================================================================
!
!  clm(t)clm(t)clm(t)clm(t)clm(t)clm(t)clm(t)clm(t)CL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  clm(t) WEB INFO: http://clm(t).gsfc.nasa.gov
!  LMclm(t)clm(t)clm(t)clm(t)clm(t)clm(t)clm(t)clm(t)  clm(t) ListServ/Mailing List: 
!
!=========================================================================
! DESCRIPTION:
!  Average clm(t) domain output and write 1-D clm(t) results. 
!
!  NOTE:Due to the complexity of various 2-D file formats, we have
!       excluded its treatment here, leaving it to the user's design.  
!
! REVISION HISTORY:
!  15 September 1999: Yongjiu Dai; Initial code
!  15 December 1999:  Paul Houser and Jon Radakovich; F90 Revision 
!=========================================================================
! $Id: drv_2dout.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  use drv_tilemodule      ! Tile-space variables
  use drv_module          ! 1-D Land Model Driver variables
  use clmtype             ! 1-D clm(t) variables
  use clm_varpar, only : nlevsoi, nlevsno
  use clm_varcon, only : hvap
  use casctype
  implicit none

!=== Arguments =============================================================

  type (drvdec)  :: drv              
  type (tiledec) :: tile(drv%nch)
  type (clm1d)   :: clm(drv%nch)
  type (casc2D)  :: casc(drv%nc,drv%nr)

!=== Local Variables =======================================================

  integer  :: n,j,t                        ! Temporary counters
  integer  :: r,c                          !@ Stefan: to pass spatial info to average function
  integer  :: mask(drv%nch)                ! Spatial averaging water mask
  real(r8) :: saturation(drv%nc,drv%nr,nlevsoi) ! 
  real     :: single_surf
!=== End Variable List ===================================================

!=== Open file for real-time plotting ====================================
!  open(3001,file="runoff_realt.pfb", form='binary',status='unknown')

  n=drv%nch
  c=drv%nc
  r=drv%nr

  !write(2006,*)  drv%time, drv%ss, drv%mn, drv%hr, drv%da, drv%mo, drv%yr, " [1]"  
  !write(2006,*) "2D field of flow depth for each layer"

!== Start of the loop to write 2D arrays for each layer for a number of variables ========

  do j=1, nlevsoi
  t = 0
   do r=1, drv%nr
    do c=1, drv%nc
	 t =t + 1
	 write(2000) clm(t)%h2osoi_liq(j)
	 
	 if (j==1) then
      write(1995)clm(t)%qflx_top_soil
      write(1996)clm(t)%qflx_infl
      write(1997)clm(t)%qflx_evap_grnd
      write(1998)clm(t)%eflx_soil_grnd
      write(1999)clm(t)%qflx_evap_veg
	  single_surf=clm(t)%qflx_surf
	  write(2001)clm(t)%qflx_surf 
!	  write(3001)single_surf
	  write(2002) clm(t)%qflx_evap_tot
	  write(2003) clm(t)%t_grnd
	  write(2004) clm(t)%qflx_evap_soi
	  write(2005) clm(t)%qflx_tran_veg
     endif
     
	 enddo
    enddo
   enddo 

!close(3001)

!=== Call real-time plotting routines
!call plotting(casc, drv, tile, clm)

end subroutine drv_2dout







