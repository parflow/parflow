program extract
! Extracts saturation and pressure values from pfb output
implicit none
integer i,j,k,l,m,t,st,et,it,na
integer nx,ny,nz,nl,nt,nlogged,nobs
integer,allocatable :: obsx(:),obsy(:)
integer,allocatable ::  counter(:,:)
real*8,allocatable :: mask(:,:,:),value(:,:,:),surf(:,:),avg(:,:)
real*8,allocatable :: obsv(:)
character*100 fname,pname,ISTEP,LOGG,mname
integer da(13)
data da/31,28,31,30,31,30,31,31,30,31,30,31,31/

!open(1,file='obs_pfb.dat',action='write')
open(1,file='obs_pfb.sat.dat',action='write')
!open(3,file='batch',status='unknown')
open(3,file='batch.sat',status='unknown')

print *,"READING BATCH FILE"
! NX Ny NZ
read(3,*)nx,ny,nz
! Project name:"
read(3,*)pname
! Start time:"
read(3,*) st
! End time:"
read(3,*) et
! Number of time steps in output package:"
read(3,*) nt
!Number of time steps logged:"
read(3,*)nlogged
! Number of observation locations
read(3,*)nobs
allocate (obsx(nobs),obsy(nobs),obsv(nobs))
do i=1,nobs
read(3,*) obsx(i),obsy(i)
enddo 

allocate(mask(nx,ny,nz),avg(nx,ny))
mask=0; value=0.0d0; avg=0.0d0; surf=0.0d0; counter=0
mname = 'washita.out.mask.00000.pfb'
call pf_read(mask,mname,nx,ny,nz)

it=0;na=0
do t=(st-1+nt),et,nt ! Start over tstep
write(ISTEP,*)t

 do l=1,nlogged      ! Start over nlogged
 it = it + 6
 na = na + 1
 write(LOGG,*)l
 fname=trim(adjustl(pname))//'.'//trim(adjustl(LOGG))//'.'//trim(adjustl(ISTEP))//'.pfb'
 print *,fname
 allocate(value(nx,ny,nz),surf(nx,ny),counter(nx,ny))
 value=0.0d0; surf=0.0d0; counter=0
 call pf_read(value,fname,nx,ny,nz)
 
  do k=nz,1,-1
   do j=1,ny
    do i=1,nx
    if (mask(i,j,k) == 1.0d0)  counter(i,j)=counter(i,j)+1
    do m=1,nobs
     if ((i==obsx(m).and.j==obsy(m).and.counter(i,j)==1)) obsv(m) = value(i,j,k)
    enddo
    if (counter(i,j)==1) then
     surf(i,j)=value(i,j,k)
     avg(i,j)=avg(i,j) + value(i,j,k)
    !if (i==10.and.j==10) print *,mask(i,j,k),value(i,j,k),avg(i,j)
    endif
    enddo
   enddo
  enddo
 
write(1,'(i4,<nobs>(e12.4))')it,(obsv(m),m=1,nobs)

 fname='press.'//trim(adjustl(LOGG))//'.'//trim(adjustl(ISTEP))
 fname='sat.'//trim(adjustl(LOGG))//'.'//trim(adjustl(ISTEP))
open(2,file=trim(fname)//'.dat',action='write')
! Calc and print monntly average
! if (na==4) then
 ! avg = avg/dfloat(na)
  !na = 0
write(2,*) nx, ny, 1
  do j=1,ny
   do i=1,nx
    write(2,*) surf(i,j)
   enddo
  enddo
 ! avg = 0.0d0
! endif
close(2)
 deallocate(value,surf,counter)
 enddo  ! End over nlogged

enddo   ! End over t

close (1)
close (3)
end program extract
