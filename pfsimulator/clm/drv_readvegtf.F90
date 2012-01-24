!#include <misc.h>

subroutine drv_readvegtf (drv,grid,tile,clm,nx, ny, ix, iy,gnx, gny, rank)

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

  !=== Local Variables =====================================================

  integer  :: c,r,t,i,j     !Loop counters
  real(r8) :: rsum          !Temporary vegetation processing variable
  real(r8) :: fvt(drv%nt)   !Temporary vegetation processing variable
  real(r8) :: max           !Temporary vegetation processing variable
  integer  :: nchp          !Number of tiles use for array size
  real(r8) :: sand          !temporary value of input sand
  real(r8) :: clay          !temporary value of input clay
  integer  :: ix,iy,nx,ny, gnx, gny   !global and local grid indicies from ParFlow
  integer  :: rank
  character*100 :: RI

  !=== End Variable Definition =============================================

  nchp = drv%nch
  write(RI,*)rank

  !=== Read in Vegetation Data
  !open(2,file=trim(adjustl(drv%vegtf))//'.'//trim(adjustl(RI)),form='formatted',action='read')
  open(2,file=trim(adjustl(drv%vegtf)),form='formatted',action='read')

  read(2,*)  !skip header
  read(2,*)  !skip header
!  print*, 
  ! do r=1,drv%nr     !rows
   ! do c=1,drv%nc  !columns
  do r =1, gny  ! @RMM replaced local row/column with global grid
   do c = 1, gnx
    if (((c > ix).and.(c <= (ix+nx))).and.((r > iy).and.(r <= (iy+ny)))) then
       read(2,*) i,j,          &
                 grid(c-ix,r-iy)%latdeg,  &
                 grid(c-ix,r-iy)%londeg,  &
                 sand,              &
                 clay,              &
                 grid(c-ix,r-iy)%isoicol, &
                 (grid(c-ix,r-iy)%fgrd(t),t=1,drv%nt)
       grid(c-ix,r-iy)%sand(:) = sand
       grid(c-ix,r-iy)%clay(:) = clay

       rsum=0.0
       do t=1,drv%nt
          rsum=rsum+grid(c-ix,r-iy)%fgrd(t)
       enddo
       if (rsum >= drv%mina) then
          grid(c-ix,r-iy)%mask=1
       else
          grid(c-ix,r-iy)%mask=0
       endif
        
    else
       read(2,*)
    end if

   enddo ! C 
  enddo ! R 

  !=== Exclude tiles with MINA (minimum tile grid area),  
  !=== normalize remaining tiles to 100%
  do r=1,drv%nr  !rows
   do c=1,drv%nc  !columns         

        rsum=0.0
        do t=1,drv%nt
           if (grid(c,r)%fgrd(t).lt.drv%mina)grid(c,r)%fgrd(t)=0.0    ! impose area percent cutoff
           rsum=rsum+grid(c,r)%fgrd(t)
        enddo

        if (rsum.gt.0.0) then ! Renormalize veg fractions within a grid to 1  
           do t=1,drv%nt      ! Renormalize SUMT back to 1.0
              if (rsum > 0.0) grid(c,r)%fgrd(t) = grid(c,r)%fgrd(t)/rsum
           enddo
        endif

     enddo
  enddo

  !=== Exclude tiles with MAXT (Maximum Tiles per grid), 
  !=== normalize remaining tiles to 100%
  !=== Determine the grid predominance order of the tiles
  !=== PVEG(NT) will contain the predominance order of tiles

  do r=1,drv%nr  !rows
     do c=1,drv%nc  !columns

        do t=1,drv%nt
           fvt(t)=grid(c,r)%fgrd(t)  !fvt= temp fgrd working array
           grid(c,r)%pveg(t)=0
        enddo
        do i=1,drv%nt  !Loop through predominance level
           max=0.0
           t=0
           do j=1,drv%nt
              if (fvt(j) > max)then
                 if (grid(c,r)%fgrd(j) > 0) then
                    max=fvt(j)
                    t=j
                 endif
              endif
           enddo
           if (t > 0)then
              grid(c,r)%pveg(t)=i
              fvt(t)=-999.0       !eliminate chosen from next search 
           endif
        enddo
     enddo !IR
  enddo !IC 

  !=== Impose MAXT Cutoff

  do r=1,drv%nr  !rows
     do c=1,drv%nc  !columns         
        rsum=0.0
        do t=1,drv%nt
           if (grid(c,r)%pveg(t).lt.1) then
              grid(c,r)%fgrd(t)=0.0    
              grid(c,r)%pveg(t)=0  
           endif
           if (grid(c,r)%pveg(t)>drv%maxt) then
              grid(c,r)%fgrd(t)=0.0              ! impose maxt cutoff
              grid(c,r)%pveg(t)=0  
           endif
           rsum=rsum+grid(c,r)%fgrd(t)
        enddo

        if (rsum > 0.0) then   ! Renormalize veg fractions within a grid to 1
           do t=1,drv%nt       ! Renormalize SUMT back to 1.0
              if (rsum > 0.0)grid(c,r)%fgrd(t)=grid(c,r)%fgrd(t)/rsum
           enddo
        endif

     enddo
  enddo

  !=== Make Tile Space

  drv%nch=0
  do t=1,drv%nt                                              !loop through each tile type
     do r=1,drv%nr                                           !loop through rows
        do c=1,drv%nc                                        !loop through columns
           if (grid(c,r)%mask.eq.1) then                     !we have land 
              if (grid(c,r)%fgrd(t) > 0.0) then
                 drv%nch = drv%nch+1                         !Count the number of tiles
                 grid(c,r)%tilei       = drv%nch             !@ Index to convert tile to grid data in one sweep; works only of 1 l-cover per cell
                 tile(drv%nch)%row     = r                   !keep track of tile row
                 tile(drv%nch)%col     = c                   !keep track of tile column
                 tile(drv%nch)%vegt    = t                   !keep track of tile surface type
                 tile(drv%nch)%fgrd    = grid(c,r)%fgrd(t)   !keep track of tile fraction
                 tile(drv%nch)%pveg    = grid(c,r)%pveg(t)   !Predominance of vegetation class in grid
                 tile(drv%nch)%sand(:) = grid(c,r)%sand(:)   !Percent sand in soil
                 tile(drv%nch)%clay(:) = grid(c,r)%clay(:)   !Percent clay in soil
                 clm(drv%nch)%londeg   = grid(c,r)%londeg    !Longitude of tile (degrees)
                 clm(drv%nch)%latdeg   = grid(c,r)%latdeg    !Latitude of tile (degrees)
                 clm(drv%nch)%isoicol  = grid(c,r)%isoicol   !Soil color 
                 clm(drv%nch)%lat = clm(drv%nch)%latdeg*4.*atan(1.)/180. !tile latitude  (radians)
                 clm(drv%nch)%lon = clm(drv%nch)%londeg*4.*atan(1.)/180. !tile longitude (radians)
              endif
           endif
        enddo
     enddo
  enddo

  ! write(*,*) 'Size of Tile-Space Dimension:',nchp
  ! write(*,*) 'Actual Number of Tiles:',drv%nch,drv%nt
  ! write(*,*)
  close(2)
  return

end subroutine drv_readvegtf
