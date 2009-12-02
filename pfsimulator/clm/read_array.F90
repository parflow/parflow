subroutine read_array(drv,clm,rank)

  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use clm_varpar, only : nlevsoi,parfl_nlevsoi
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)   !CLM 1-D Module

  integer i,j,k,t,rank
  real(r8) hcond(drv%nc,drv%nr,nlevsoi)
  real(r8) mask(drv%nc,drv%nr,parfl_nlevsoi)
  character*100 filename,RI

  write(RI,'(i5.5)') rank
  filename='washita.out.mask.00000.pfb.'//trim(adjustl(RI))

!  call pf_read(mask,filename,drv%nc,drv%nr,parfl_nlevsoi)

  do k = 1, parfl_nlevsoi ! CLM loop over z 
   t = 0
   do j = 1, drv%nr
    do i = 1, drv%nc
     t = t + 1
     clm(t)%watsat(k) = 0.45d0
!     clm(t)%topo_mask(k) = int(mask(i,j,parfl_nlevsoi-k+1))
!     if (mask(i,j,k) == 1.0d0 ) clm(t)%planar_mask = 1 
    enddo
   enddo
  enddo

end subroutine read_array

