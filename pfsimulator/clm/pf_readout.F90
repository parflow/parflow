subroutine pfreadout(clm,drv,tile,saturation,pressure,rank,ix,iy,nx,ny,nz, j_incr,k_incr,ip)

  use drv_module          ! 1-D Land Model Driver variables
  !use dfport
  use precision
  use drv_tilemodule      ! Tile-space variables
  use clmtype
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : denh2o
  implicit none

  type (drvdec) ,intent(inout) :: drv
  type (tiledec) :: tile(drv%nch)
  type (clm1d), intent(inout) :: clm(drv%nch)   !CLM 1-D Module
  integer nx, ny, nz
  real(r8) saturation((nx+2)*(ny+2)*(nz+2)),pressure((nx+2)*(ny+2)*(nz+2))
								
  integer i,j,k,rank,ix,iy, j_incr,k_incr,ip
  integer t, l
!  character*100 RI
!  character*100 ISTEP

!  flowd=0.0d0
!  wiltcounter = 0 
!  wilt = 0 

!  write(RI,*)rank
!  write(ISTEP,*)clm(1)%istep
  ! open(777,file='flowd.'//trim(adjustl(RI))//'.'//trim(adjustl(ISTEP)),status='unknown',form='unformatted')
  
!  sat_flag = 0    
  !print*, "+++++++++++++++ about to loop and copy sats back ino CLM ++++++++++++++"
  ! Start: assign saturation data from PF to tiles/layers of CLM


!j_incr = nx_f - nx
!k_incr = (nx_f * ny_f) - (ny * nx_f)
!print*, ' in readout'
!print*,  ip, j_incr, k_incr
do t=1,drv%nch
i=tile(t)%col
j=tile(t)%row
  do k = 1, nlevsoi
     l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))  ! updated indexing @RMM
!    l = 1+i + j_incr*(j-1) + k_incr*(clm(t)%topo_mask(1)-k)
     if(clm(t)%planar_mask == 1) then
      clm(t)%pf_vol_liq(k) = saturation(l) * clm(t)%watsat(k)
      clm(t)%pf_press(k) = pressure(l) * 1000.d0
!        clm(t)%h2osoi_liq(k) = saturation(l) * clm(t)%watsat(k)*clm(t)%dz(1)*denh2o
        clm(t)%h2osoi_liq(k) = clm(t)%pf_vol_liq(k)*clm(t)%dz(1)*denh2o
!		print*, t,i,j,k,clm(t)%h2osoi_liq(k),clm(t)%pf_vol_liq(k),clm(t)%pf_press(k)
!		print*,i,j,k,l
!		print*, clm(t)%pf_vol_liq(k),saturation(l),clm(t)%watsat(k)
!		print*,clm(t)%pf_press(k),pressure(l)
!		print*,clm(t)%h2osoi_liq(k)
      endif
  end do !k
  
end do !t


! Asign root fractions based on something like a wilting point
!  do t=1,drv%nch
!  i=tile(t)%col
!  j=tile(t)%row
!  if ((nlevsoi-wiltcounter(i,j))== 0) then 
!   frac = 0.0d0
!   !print *,"NO TRANSPIRATION AT:",i,j
!  else
!   frac = 1.0d0/dfloat(nlevsoi-wiltcounter(i,j))
!  endif
!   do k = 1, nlevsoi ! clm loop
!     clm(t)%rootfr(k) = dfloat(1-wilt(i,j,k))*frac
!  enddo
!  enddo

end subroutine pfreadout
