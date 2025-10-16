!#include <misc.h>

!subroutine drv_readvegtf (drv,grid,tile,clm,nx, ny, ix, iy,gnx, gny, rank)
subroutine drv_readvegtf (grid, nx, ny, ix, iy, gnx, gny, rank)

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
  !  This primary goal of this routine is to determine tile space.
  !
  ! REVISION HISTORY:
  !  15 Jan 2000: Paul Houser; Initial code
  !=========================================================================      
  ! $Id: drv_readvegtf.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
  !=========================================================================

  use MOD_Precision
  !use drv_module          ! 1-D Land Model Driver variables
  !use drv_tilemodule      ! Tile-space variables
  !use clmtype             ! 1-D CLM variables
  use drv_gridmodule      ! Grid-space variables
  implicit none

  !=== Arguments ===========================================================

  !type (drvdec)  :: drv              
  !type (tiledec) :: tile(drv%nch)
  !type (clm1d)   :: clm (drv%nch)
  !type (griddec) :: grid(drv%nc,drv%nr)  
  type (griddec) :: grid(nx,ny) 

  !=== Local Variables =====================================================

  integer  :: c,r,t,i,j     !Loop counters
  !real(r8) :: rsum          !Temporary vegetation processing variable
  !real(r8) :: fvt(drv%nt)   !Temporary vegetation processing variable
  !real(r8) :: max           !Temporary vegetation processing variable
  !integer  :: nchp          !Number of tiles use for array size
  !real(r8) :: sand          !temporary value of input sand
  !real(r8) :: clay          !temporary value of input clay
  integer  :: ix, iy, nx, ny, gnx, gny   !global and local grid indicies from ParFlow
  integer  :: rank
  character*100 :: RI

  !=== End Variable Definition =============================================

  !nchp = drv%nch
  write(RI,*)rank

  !=== Read in Vegetation Data
  !open(2,file=trim(adjustl(drv%vegtf))//'.'//trim(adjustl(RI)),form='formatted',action='read')
  open(2,file=trim(adjustl('CoLM_readin.dat')),form='formatted',action='read')

  !read(2,*)  !skip header
  !read(2,*)  !skip header
!  print*, 
  ! do r=1,drv%nr     !rows
   ! do c=1,drv%nc  !columns
  do r = 1, gny  ! @RMM replaced local row/column with global grid
    do c = 1, gnx
      if (((c > ix).and.(c <= (ix+nx))).and.((r > iy).and.(r <= (iy+ny)))) then
        read(2,*) i,j,          &
                  grid(c-ix,r-iy)%patchclass,                               &
                  grid(c-ix,r-iy)%patchlonr,                                &       
                  grid(c-ix,r-iy)%patchlatr,                                &
                  !(grid(c-ix,r-iy)%vf_quartz(t),t=1,nl_soil),               &
                  (grid(c-ix,r-iy)%int_soil_grav_l(t),t=1,8),              &
                  (grid(c-ix,r-iy)%int_soil_sand_l(t),t=1,8),                   &
                  (grid(c-ix,r-iy)%int_soil_clay_l(t),t=1,8),                 &
                  (grid(c-ix,r-iy)%int_soil_oc_l(t),t=1,8),              &
                  (grid(c-ix,r-iy)%int_soil_bd_l(t),t=1,8)
                  !(grid(c-ix,r-iy)%psi0(t),t=1,nl_soil),                    &
                  !(grid(c-ix,r-iy)%bsw(t),t=1,nl_soil),                     &
                  !(grid(c-ix,r-iy)%theta_r(t),t=1,nl_soil),                 &
                  !(grid(c-ix,r-iy)%alpha_vgm(t),t=1,nl_soil),               &
                  !(grid(c-ix,r-iy)%n_vgm(t),t=1,nl_soil),                   &
                  !(grid(c-ix,r-iy)%L_vgm(t),t=1,nl_soil),                   &
                  !(grid(c-ix,r-iy)%hksati(t),t=1,nl_soil),                  &
                  !(grid(c-ix,r-iy)%csol(t),t=1,nl_soil),                    &
                  !(grid(c-ix,r-iy)%k_solids(t),t=1,nl_soil),                &
                  !(grid(c-ix,r-iy)%dksatu(t),t=1,nl_soil),                  &
                  !(grid(c-ix,r-iy)%dksatf(t),t=1,nl_soil),                  &
                  !(grid(c-ix,r-iy)%dkdry(t),t=1,nl_soil)
                  !(grid(c-ix,r-iy)%BA_alpha(t),t=1,nl_soil),                &
                  !(grid(c-ix,r-iy)%BA_beta(t),t=1,nl_soil)
                  !(grid(c-ix,r-iy)%OM_density(t),t=1,nl_soil),              &
                  !(grid(c-ix,r-iy)%BD_all(t),t=1,nl_soil)
                  !grid(c-ix,r-iy)%htoplc
        !grid(c-ix,r-iy)%sand(:) = sand
        !grid(c-ix,r-iy)%clay(:) = clay

        !rsum=0.0
        !do t=1,drv%nt
        !   rsum=rsum+grid(c-ix,r-iy)%fgrd(t)
        !enddo
        !if (rsum >= drv%mina) then
        !   grid(c-ix,r-iy)%mask=1
        !else
        !   grid(c-ix,r-iy)%mask=0
        !endif
          
      else
        read(2,*)
      end if

    enddo ! C 
  enddo ! R 

  ! write(*,*) 'Size of Tile-Space Dimension:',nchp
  ! write(*,*) 'Actual Number of Tiles:',drv%nch,drv%nt
  ! write(*,*)
  close(2)
  return

end subroutine drv_readvegtf
