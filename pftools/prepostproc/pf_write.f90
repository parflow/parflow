  program pf_write
  implicit none
  real*8  dx2, dy2, dz2
  real*8,allocatable :: data(:,:,:)
  real*8 ri, rj, rk1, rk2, headsum, rsum, junk,  &
         ksum, kavg,f, dx, dy, dz, x1, y1, z1								
  integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,			&
            ns,  rx, ry, rz,nx,ny,nz, nnx, nny, nnz,    &
			is,dummy
  integer*4 ijk, namelength, xtent,ytent,ztent
  integer t
  character*100 ifname,ofname
  
  write(*,*)"Input file name:"
  read(*,'(a)')ifname
  write(*,*)"Output file name:"
  read(*,'(a)')ofname
  
  open(99,file=trim(adjustl(ifname)),status='old',action='read')
  read(99,*)nx,ny,nz
  allocate (data(nx,ny,nz))
  do k=1,nz
    do j=1,ny
      do i=1,nx
      read(99,*) data(i,j,k)
      enddo
    enddo
  enddo

  x1=0.0d0
  y1=0.0d0
  z1=256.0d0
  ns=1
  nnx=nx
  nny=ny
  nnz=nz
  dx = 1000.
  dy = 1000.
  dz = 0.5
  ix=0
  iy=0
  iz=0
  rx=1
  ry=1
  rz=1

  open(100,file=trim(adjustl(ofname)),form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN')         !binary outputfile of Parflow
   
  ! Write in header info

! Start: reading of domain spatial information
  write(100) x1 !X
  write(100) y1 !Y 
  write(100) z1 !Z
!  print *,"1",x1,y1,z1

  write(100) nx !NX
  write(100) ny !NY
  write(100) nz !NZ
!  print *,"n",nx,ny,nz

  write(100) dx !DX
  write(100) dy !DY
  write(100) dz !DZ
!  print *,"d",dx,dy,dz

  dx2 = dx
  dy2 = dy
  dz2 = dz
  write(100) ns !num_subgrids
!  print *,"ns",ns
! End: reading of domain spatial information

! Start: loop over number of sub grids
  do is = 0, (ns-1)

! Start: reading of sub-grid spatial information
   write(100) ix
   write(100) iy
   write(100) iz
!   print *,"i",ix,iy,iz
   
   write(100) nnx
   write(100) nny
   write(100) nnz
!   print *,"nn",nnx,nny,nnz

   write(100) rx
   write(100) ry
   write(100) rz
!   print *,"r",rx,ry,rz

! End: writing of sub-grid spatial information

! Start: write data for each individual subgrid
  do  k=iz +1 , iz + nnz
   do  j=iy +1 , iy + nny
    do  i=ix +1 , ix + nnx
     write(100) dabs(data(i,j,k))
    end do
   end do
  end do
! End: read in saturation data from each individual subgrid

  end do
! End: loop over number of sub grids

  
  close(99)
  close(100)
  end
