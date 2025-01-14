 subroutine pfb_write(value,fname,nx,ny,nz,x1,y1,z1,dx,dy,dz)
  implicit none
  real*8 value(nx,ny,nz)
  real*8 dx, dy, dz, x1, y1, z1                                                         
  integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,                   &
            ns,  rx, ry, rz,nx,ny,nz, nnx, nny, nnz,is
  character*100 fname
 
  open(100,file=trim(adjustl(fname)),form='unformatted',   &
                 recordtype='stream',convert='BIG_ENDIAN',status='unknown')
  nnx=nx;nny=ny;nnz=nz
  ns=1
  ix=0;iy=0;iz=0
  rx=0;ry=0;rz=0
 
! Start: writing of domain spatial information
  write(100) x1 !X
  write(100) y1 !Y 
  write(100) z1 !Z
 
  write(100) nx !NX
  write(100) ny !NY
  write(100) nz !NZ
 
  write(100) dx !DX
  write(100) dy !DY
  write(100) dz !DZ
 
  write(100) ns !num_subgrids
! End: writing of domain spatial information
 
! Start: loop over number of sub grids
  do is = 0, (ns-1)
 
! Start: writing of sub-grid spatial information
   write(100) ix
   write(100) iy
   write(100) iz
 
   write(100) nnx
   write(100) nny
   write(100) nnz
   write(100) rx
   write(100) ry
   write(100) rz
 
! End: writing of sub-grid spatial information
 
! Start: write in saturation data from each individual subgrid
  do  k=iz +1 , iz + nnz
   do  j=iy +1 , iy + nny
    do  i=ix +1 , ix + nnx
     write(100) value(i,j,k)
    end do
   end do
   end do
! End: write in saturation data from each individual subgrid
 
! End: write in saturation data from each individual subgrid
 
  end do
! End: loop over number of sub grids
 

 
  close(100)
  end subroutine

