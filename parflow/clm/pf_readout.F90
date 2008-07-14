subroutine pfreadout(clm,drv,tile,saturation_data,pressure_data,temperature_data,rank,ix,iy)

  use drv_module          ! 1-D Land Model Driver variables
  !use dfport
  use precision
  use drv_tilemodule      ! Tile-space variables
  use clmtype
  use clm_varpar, only : nlevsoi,parfl_nlevsoi
  use clm_varcon, only : denh2o
  implicit none

  type (drvdec) ,intent(inout) :: drv
  type (tiledec) :: tile(drv%nch)
  type (clm1d), intent(inout) :: clm(drv%nch)	 !CLM 1-D Module
  real(r8) saturation_data(drv%nc,drv%nr,parfl_nlevsoi),pressure_data(drv%nc,drv%nr,parfl_nlevsoi)
  real(r8) temperature_data(drv%nc,drv%nr,parfl_nlevsoi)
  real(r8) flowd(drv%nc,drv%nr)
  real(r8) frac
								
  integer*4 i,j,k,rank,ix,iy
  integer dummy,wiltcounter(drv%nc,drv%nr),wilt(drv%nc,drv%nr,nlevsoi)
  integer t, counter(drv%nc,drv%nr),l,sat_flag
  character*100 RI,ISTEP

  flowd=0.0d0
  wiltcounter = 0 
  wilt = 0 

  write(RI,*)rank
  write(ISTEP,*)clm(1)%istep
  ! open(777,file='flowd.'//trim(adjustl(RI))//'.'//trim(adjustl(ISTEP)),status='unknown',form='unformatted')
  
  sat_flag = 0    
  !print*, "+++++++++++++++ about to loop and copy sats back ino CLM ++++++++++++++"
  ! Start: assign saturation data from PF to tiles/layers of CLM
  counter = 0 
  do k=1,parfl_nlevsoi  ! clm loop over z
    do t=1,drv%nch
    i=tile(t)%col
    j=tile(t)%row
    if(clm(t)%planar_mask == 1) then
      clm(t)%pf_vol_liq(k) = saturation_data(i,j,parfl_nlevsoi-k+1) * clm(t)%watsat(k)
      clm(t)%pf_press(k) = pressure_data(i,j,parfl_nlevsoi-k+1) * 1000.0d0 !/ (1000.0d0 * 9.81)
      if (clm(t)%topo_mask(k) >= 1) then
        counter(i,j)  = counter(i,j) + 1
        if (clm(t)%pf_press(k) < -100000.0d0) then 
          wiltcounter(i,j) = wiltcounter(i,j) + 1 ! counts how many layers are beyond the wilting point
          wilt(i,j,counter(i,j))=1                ! this layer wilts man
        endif 
        if (counter(i,j) == 1 .and. clm(t)%pf_press(k) > 0.0d0) sat_flag = 1
        if (counter(i,j) == 1) then 
           flowd(i,j) = clm(t)%pf_press(k)/1000.0d0
           flowd(i,j) = saturation_data(i,j,parfl_nlevsoi-k+1)
        endif
               
        clm(t)%h2osoi_liq(counter(i,j)) = clm(t)%pf_vol_liq(k)*clm(t)%dz(1)*denh2o
        clm(t)%t_soisno(counter(i,j)) = temperature_data(i,j,parfl_nlevsoi-k+1)
      endif
    endif 
!if (i == 10 .and. j == 10) write(*,'(3i,f,i,f)')i,j,k,(clm(t)%pf_press(k)/1000.0d0),clm(t)%topo_mask(k),clm(t)%pf_vol_liq(k)
    enddo
  enddo
    !if (sat_flag == 1) print *,"OVERLAND FLOW AT TSTEP:",clm(1)%istep
  
  dummy=1
  !write(777)ix,iy,dummy,drv%nc,drv%nr,dummy
  !do j=1,drv%nr
  !do i=1,drv%nc 
  !  write(777) flowd(i,j)
  !enddo
  !enddo
  !close(777)

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
