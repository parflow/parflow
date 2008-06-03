subroutine plotting(casc, drv, tile, clm)
  use dfport

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
 
 real :: apr1(drv%nc,drv%nr),apr2(2,10) !(drv%nc,drv%nr)
 integer imax
 integer iext,jext
 real xlen,ylen
 real,allocatable :: cval1(:),cval2(:)
 real,allocatable :: color1(:,:),color2(:,:)
 integer nval1,nval2
 logical ioffp
 real spval,maxapr
 real dimx,dimy
 integer tstep,ntstep,r,c,counter,i,t
 logical orientation

  open(3001,file="runoff_realt.pfb", form='binary',status='old')  ! @ 2D output file     
!   open(2002,file="evap_2d.pfb", form='binary',status='old')   ! @ 2D output file     
!    open(2003,file="Tsoil_2d.pfb", form='binary',status='old') ! @ 2D output file


 orientation = .TRUE.
 xlen = 4.0
 ylen = 4.0
 iext = 2  !drv%nc
 jext = 10 !drv%nr
 imax = 10 !drv%nc
 nval1 = 11
 nval2 = 11
 dimx = 1000.0d0 / xlen !drv%nc*casc(1,1)%W/xlen
 dimy = 2.0d0 / ylen    !drv%nr*casc(1,1)%W/ylen

 ntstep = 1

 allocate (cval1(nval1),cval2(nval2))
 allocate (color1(3,nval1),color2(3,nval2))

  do tstep = 1,ntstep
  t = 0
   do r=1, jext
    do c=1, iext
     t = t + 1
      !if (clm(t)%qflx_surf < 1.0e-10) then
      !apr1(c,r)=0.0e+1
 !     apr1(c,r)=clm(t)%watsat(1)
      !else
 !     apr1(c,r)=clm(t)%qflx_surf
 !     apr1(c,r)=clm(t)%watsat(1)
	  !endif
	  apr2(c,r) = clm(1)%pf_vol_liq(r)/clm(1)%watsat(1)
     enddo
    enddo
   enddo
           
!=== Determine contour interval for apr1====
 do counter = 1, nval1
  if (counter == 1) then
   cval1(counter) = minval(apr1)  
   color1(1,counter) = 0.95
   color1(2,counter) = 0.2
   color1(3,counter) = 0.05
  else
   cval1(counter) = cval1(counter-1) + (maxval(apr1)-minval(apr1))/float(nval1)
   color1(1,counter) = color1(1,counter-1) - 0.1
   color1(2,counter) = 0.2
   color1(3,counter) = color1(2,counter-1) + 0.1
  endif
 enddo
   
 !=== Determine contour interval for apr2====
 do counter = 1, nval1
  if (counter == 1) then
   cval2(counter) = 0.0
   color2(1,counter) = 0.95
   color2(2,counter) = 0.2
   color2(3,counter) = 0.05
  else
   cval2(counter) = cval2(counter-1) +  0.1
   color2(1,counter) = color1(1,counter-1) - 0.1
   color2(2,counter) = 0.2
   color2(3,counter) = color1(2,counter-1) + 0.1
  endif
 enddo
 
print *,"Start the plotting routines"

!=== PLOTTING OF CONTOURPLOTS ===========================
 call psinit(orientation)
!=== Lower plot ================ 
 !call plot(1.0,1.0,-3)
 !call setlw(0.01)
 !if (maxval(apr1) /= 0.0)  call concolr(apr1,imax,iext,jext,xlen,ylen,cval1,color1,nval1,0,0.0d0)
 !call border(xlen,ylen,1111,1111,iext,1,jext,1)
 !call axis(0.0,0.0,4hx(m) ,-4,xlen,0.,0.0,dimx) !x,y,title,# character in title, length, 0/90 for x/y,start value, increment
 !call axis(0.0,0.0,4hy(m) ,4,xlen,90.,0.0,dimy)
 
!=== Upper plot =================== 
 call plot(0.0,5.0,-3)
 call setlw(0.01)
 if (maxval(apr2) /= 0.0) call concolr(apr2,imax,iext,jext,xlen,ylen,cval2,color2,nval2,0,0.0d0)
 call border(xlen,ylen,1111,1111,iext,1,jext,1)
 call axis(0.0,0.0,4hx(m) ,-4,xlen,0.,0.0,dimx) !x,y,title,# character in title, length, 0/90 for x/y,start value, increment
 call axis(0.0,0.0,4hy(m) ,4,xlen,90.,0.0,dimy)
 call plotnd
 
 !=== PLOTTING OF LINEGRAPHS ===========================
! call newdev(outflow,7)
! call psinit(orientation)
  
 close(3001)
 end subroutine plotting


