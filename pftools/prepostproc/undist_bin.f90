program ud_bin
implicit none
real*8,allocatable :: value(:,:,:), avg(:,:,:)
integer i,j,k,step,ix,iy,iz,nx,ny,nz,bt,et
integer gnx,gny,gnz,counter,c,days(12),mo,da,hr,imo
integer proc,nproc,fehler
integer lend,fend,ndt,t
integer obsx(1),obsy(1)
character*100 RI,ISTEP,fname,dummy,var
character*2 cda,cmo

  data days /31,28,31,30,31,30,31,31,30,31,30,31/

print *,"File name:"
read(*,*) fname
print *,"var:"
read(*,*) var
print *,"Beginning time step:"
read(*,*)bt
print *,"Ending time step:"
read(*,*)et
write(*,*)"NX:"
read(*,*)gnx
write(*,*)"NX:",gnx
write(*,*)"NY:"
read(*,*)gny
write(*,*)"NY:",gny
print *,"Number of processors:"
read(*,*)nproc

print *,"Number of processors:",nproc

allocate(value(gnx,gny,et),avg(3,gnx,gny))



do step=bt,et ! Start over tstep
write(ISTEP,*)step

  do proc=0,nproc-1 ! Start over processors
  write(RI,*)proc
  open(1,file=trim(adjustl(fname))//'.'//trim(adjustl(ISTEP))//'.bin.'//trim(adjustl(RI)),status='old',form='binary',iostat=fehler)
  !open(1,file=trim(adjustl(fname))//'.'//trim(adjustl(RI)),status='old',form='binary',iostat=fehler)
  !print *,trim(adjustl(fname))//'.'//trim(adjustl(RI))
  !print *,"File Status:",fehler

   print *,trim(adjustl(fname))//'.'//trim(adjustl(ISTEP))//'.bin.'//trim(adjustl(RI))
   read(1,iostat=fend)ix,iy,iz,nx,ny,nz
   print *,ix,iy,iz,nx,ny,nz
   t=step
!print*, t
    do k=iz,iz+nz-1
     do j=iy,iy+ny-1
      do i=ix,ix+nx-1
      read(1,iostat=fend)value(i,j,t) ! The case treated here is only for 2D arrays
      enddo ! End over i
     enddo ! End over j
    enddo ! End over k
    close(1)  
  enddo ! End over processors

enddo ! End over tstep
print*, t

print *,"Writing ASCII"
counter = 8761 
counter = bt 
avg = 0.0d0
!mo = 9
!must be one year of hourly timesteps or will crash
do mo =1, 12
do da = 1, days(mo)
do hr = 1, 24
 do j=1,gny
  do i=1,gnx
  avg(1,i,j) = avg(1,i,j) + value(i,j,counter)
  enddo ! i
 enddo ! j
 counter = counter + 1
end do ! hour
avg(1,:,:) = avg(1,:,:) / 24.d0
write(cmo,'(i2.2)') mo
write(cda,'(i2.2)') da
open(10,file=trim(var)//cmo//'.'//cda//'.2006.txt')
print*, var//cmo//'.'//cda//'.2006.txt'
write(10,*) gnx, gny, 1
 do j=1,gny
  do i=1,gnx
  write(10,*) avg(1,i,j)
   avg(2,i,j) = avg(2,i,j) + avg(1,i,j)
  enddo ! i
 enddo ! j
close(10)
avg(1,:,:) = 0.d0
end do ! days 
avg(2,:,:) = avg(2,:,:) / float(days(mo))
write(cmo,'(i2.2)') mo
write(cda,'(i2.2)') da
open(10,file=trim(var)//cmo//'.2006.txt')
write(10,*) gnx, gny, 1
 do j=1,gny
  do i=1,gnx
  write(10,*) avg(2,i,j)
   avg(3,i,j) = avg(3,i,j) + avg(2,i,j)
  enddo ! i
 enddo ! j
close(10)
avg(2,:,:) = 0.0d0
!mo = mo + 1
!if (mo == 13) mo = 1
end do ! month
avg(3,:,:) = avg(3,:,:) / 12.0d0
open(10,file=trim(var)//'.2006.txt') 
write(10,*) gnx, gny, 1  
 do j=1,gny
  do i=1,gnx
  write(10,*) avg(3,i,j)
  enddo ! i
 enddo ! j
close(10)


end
