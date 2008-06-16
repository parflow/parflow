subroutine vegfile_gen(clm,drv)
  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
  integer t

 open(1,file='val.drv_vegm.dat',form='formatted',status='unknown')

 write (1,'(a110)') "x  y  lat    lon    sand clay color  fractional coverage of grid by vegetation class (Must/Should Add to 1.0)"
 write (1,'(a125)') "      (Deg)	 (Deg)  (%/100)   index  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18"

 t = 0
 do i=1,nc
  do j=1,nr
   t = t + 1
   if (clm(t)%planar_mask == 1) write(1,'(2i3,2x,a119)') i,j,"57.6   33.1   0.16 0.265   2   0.0 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0"
  enddo
 enddo
 close (1)
end program vegfile_gen
