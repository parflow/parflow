subroutine topomask(clm,drv)
  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use clm_varpar, only : nlevsoi,parfl_nlevsoi
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module

!@ Local variables
  real*8  dx2, dy2, dz2
  real*8 value
  real*8 ri, rj, rk1, rk2, headsum, rsum, junk,  &
         ksum, kavg,f, dx, dy, dz, x1, y1, z1								
  integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,			&
            ns,  rx, ry, rz,nx,ny,nz, nnx, nny, nnz,    &
			is,dummy
  integer*4 ijk, namelength, xtent,ytent,ztent
  integer t
  
  open(301,file="topomask.pfb",form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='old') !@ binary file of mask values in pfb format
   
! Start: reading of domain spatial information
  read(301) x1 !X
  read(301) y1 !Y 
  read(301) z1 !Z
!  print *,"1",x1,y1,z1

  read(301) nx !NX
  read(301) ny !NY
  read(301) nz !NZ
!  print *,"n",nx,ny,nz

  read(301) dx !DX
  read(301) dy !DY
  read(301) dz !DZ
!  print *,"d",dx,dy,dz

  drv%dx = dx
  drv%dy = dy
  drv%dz = dz
  read(301) ns !num_subgrids
!  print *,"ns",ns
! End: reading of domain spatial information

! Start: loop over number of sub grids
  do is = 0, (ns-1)

! Start: reading of sub-grid spatial information
   read(301) ix
   read(301) iy
   read(301) iz
!   print *,"i",ix,iy,iz
   
   read(301) nnx
   read(301) nny
   read(301) nnz
!   print *,"nn",nnx,nny,nnz

   read(301) rx
   read(301) ry
   read(301) rz
!   print *,"r",rx,ry,rz

! End: reading of sub-grid spatial information

! Start: read in saturation data from each individual subgrid
  do  k=iz, iz + nnz - 1
  t = 0
   do  j=iy, iy + nny - 1
    do  i=ix, ix + nnx - 1
     t = t + 1
     read(301) value
	 clm(t)%topo_mask(parfl_nlevsoi-k) = value
	 if (value == 1.0d0) clm(t)%planar_mask = 1 
    end do
   end do
  end do
! End: read in saturation data from each individual subgrid

  end do
! End: loop over number of sub grids

  close(301)
  end
