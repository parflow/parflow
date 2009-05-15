subroutine pf_read(x,filename,ixlim,iylim,izlim)

use parflow_config

Real*8 x(ixlim,iylim,izlim),  ri, rj, rk1, rk2, headsum,    &
rsum, junk,                                           &   
ksum, kavg,f, dx, dy, dz, x1, y1, z1
Integer*4 i,j,k, i1, j1, k1, ixlim, iylim, izlim, &
ns, ix, iy, iz, rx, ry, rz
character*40 filename

!
!	Open File
!
filename = trim(filename)
open(15,file=filename, access=ACCESS, form=FORM, convert='BIG_ENDIAN')

!
! Calc domain bounds
!
	ix = 0
	iy = 0
	iz = 0


	ns = 1

	rx = 1
	ry = 1
	rz = 1

	x1 = dble(ixlim)*dx
	y1 = dble(iylim)*dy
	z1 = dble(izlim)*dz
!
!	read header info
!

read(15) x1
read(15) y1
read(15) z1

read(15) ixlim
read(15) iylim
read(15) izlim

read(15) dx
read(15) dy
read(15) dz

read(15) ns

read(15) ix
read(15) iy
read(15) iz

read(15) ixlim
read(15) iylim
read(15) izlim

read(15) rx
read(15) ry
read(15) rz


do  k=1,izlim
do  j=1,iylim
do  i=1,ixlim
read(15) x(i,j,k)
end do
end do
end do

close(15)
return
end
