subroutine read_array(casc,drv,clm)

  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use casctype
  use clm_varpar, only : nlevsoi
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
  type (casc2D):: casc(drv%nc,drv%nr)

  integer i,j,k,t
  real(r8) hcond(drv%nc,drv%nr,nlevsoi)


  !open(1000,file='kfield.dat')
  !read(1000,*)
  open(1001,file='porosity.dat',status='old',form='formatted')
  read(1001,*)
  
  !open(6000,file="x_slopes.dat",status='old',form='formatted')
  !open(6001,file="y_slopes.dat",status='old',form='formatted')

  do k = 1, parfl_nlevsoi
   t = 0
   do j = 1, drv%nr
    do i = 1, drv%nc
	 t = t + 1
	 !read(1000,*) clm(t)%hksat(parfl_nlevsoi+1-k)
     !read(1001,*) clm(t)%watsat(parfl_nlevsoi+1-k)
     !print *,clm(t)%watsat(k)
     clm(t)%watsat(k) = 0.45d0
	enddo
   enddo
  enddo
  
  t = 0
  drv%max_sl = 0.0d0
  do j = 1, drv%nr
   do i = 1, drv%nc
    t = t + 1
    !if (clm(t)%planar_mask == 1) read(6000,*)casc(i,j)%x_sl
    !if (clm(t)%planar_mask == 1) read(6001,*)casc(i,j)%y_sl
   enddo
  enddo
  
  drv%max_sl = maxval(abs(casc%x_sl))
  if(maxval(abs(casc%y_sl)) > drv%max_sl) drv%max_sl = maxval(abs(casc%y_sl))  

  !close(1000)
  !close(1001)
  !close(6000)
  !close(6001)
end subroutine read_array

