subroutine pf_read(x,filename,nx,ny,nz)
real*8  :: x(nx,ny,nz)
character*100 :: filename
character*2 :: proc 
integer*4 :: nx
integer*4 :: ny
integer*4 :: nz
integer*4 :: islog 
real*8  :: dx2
real*8  :: dy2
real*8  :: dz2
real*8  :: x1
real*8  :: y1
real*8  :: z1
Real*8 value,  ri, rj, rk1, rk2, headsum, rsum, junk,  &
ksum, kavg,f, dx, dy, dz                                                                
Integer*4 i,j,k, nni, nnj, nnk, ix, iy, iz,                     &
ns,  rx, ry, rz, nnx, nny, nnz, is
integer*4 ijk, namelength, xtent,ytent,ztent,np
                      
!allocate(x(nx,ny,nz)) 
                      
                      
        print*, trim(filename) 
                    
!      Open File
 
open(15,file=trim(filename),form='unformatted',   &
recordtype='stream',convert='BIG_ENDIAN',status='old')
 
!      Read in header info 
 
        read(15) x1
        read(15) y1
        read(15) z1 
              
        read(15) nx 
        read(15) ny 
        read(15) nz 
 
        read(15) dx
        read(15) dy
        read(15) dz
 
        dx2 = dx 
        dy2 = dy
        dz2 = dz 
 
        read(15) ns 
 
print*, 'header for file:', filename
print*, 'X0, Y0, Z0: ', x1, y1, z1
print*, 'nx, ny, nz: ', nx, ny, nz
print*, 'dx, dy, dz: ', dx, dy, dz
print*, 'number of subgrids: ', ns
 
do is = 0, (ns-1)
if (is > 0) then
!close (15)
!write(proc,'(i2)') is
!if (is < 10) write(proc,'(i1)') is
!open(15,file=trim(filename)//'.pfb.'//trim(proc),form='unformatted',   &
!recordtype='stream',convert='BIG_ENDIAN',status='old')

end if
  read(15) ix
  read(15) iy
  read(15) iz
  read(15) nnx
  read(15) nny
  read(15) nnz

print*, ix,iy,iz, nnx, nny, nnz
 
  read(15) rx
  read(15) ry
  read(15) rz
 
  do  k=iz +1 , iz + nnz
    do  j=iy +1 , iy + nny
    do  i=ix +1 , ix + nnx
    read(15) value
!        if (islog == 1) then
!     x(i,j,k) = dlog10(value)
!       else
    x(i,j,k) = value
!      end if
        end do
    end do
  end do
 
end do
      close(15)
      print*, 'file read in'
        return
        end

