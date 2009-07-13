!#include <misc.h>

subroutine drv_2dout (drv, grid, clm)

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
  use drv_module          ! 1-D Land Model Driver variables
  use drv_gridmodule
  use clmtype             ! 1-D clm(t) variables
  use clm_varpar, only : nlevsoi, nlevsno
  use clm_varcon, only : hvap
  implicit none

  !=== Arguments =============================================================

  type (drvdec)  :: drv              
  type (clm1d)   :: clm(drv%nch)
  type (griddec) :: grid(drv%nc,drv%nr)

  !=== Local Variables =======================================================

  integer  :: i,j,k                        ! Temporary counters
  !=== End Variable List ===================================================

  !== Start of the loop to write 2D arrays for each layer for a number of variables ========

  do k=1, nlevsoi

     do j=1, drv%nr
        do i=1,drv%nc
           if (k==1) then
              write(1995) clm(grid(i,j)%tilei)%qflx_top_soil
              write(1996) clm(grid(i,j)%tilei)%qflx_infl
              write(1997) clm(grid(i,j)%tilei)%qflx_evap_grnd
              write(1998) clm(grid(i,j)%tilei)%eflx_soil_grnd
              write(1999) clm(grid(i,j)%tilei)%qflx_evap_veg
              write(2000) clm(grid(i,j)%tilei)%eflx_sh_tot
              write(2001) clm(grid(i,j)%tilei)%eflx_lh_tot
              write(2002) clm(grid(i,j)%tilei)%qflx_evap_tot
              write(2003) clm(grid(i,j)%tilei)%t_grnd
              write(2004) clm(grid(i,j)%tilei)%qflx_evap_soi
              write(2005) clm(grid(i,j)%tilei)%qflx_tran_veg
              write(2006) clm(grid(i,j)%tilei)%eflx_lwrad_out
              write(2007) clm(grid(i,j)%tilei)%h2osno  !MHD/RMM
           endif
           write(2009) clm(grid(i,j)%tilei)%t_soisno(k)
        enddo
     enddo

  enddo

end subroutine drv_2dout
