subroutine pf_couple(drv,clm,tile,evap_trans_data)

  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use drv_tilemodule      ! Tile-space variables
  use clm_varpar, only : nlevsoi,parfl_nlevsoi
  use clm_varcon, only : denh2o, denice, istwet, istice
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)	 !CLM 1-D Module
  type (tiledec) :: tile(drv%nch)

  integer i,j,k,c,r,l,t
  integer width_step,I_err
  integer r_len, c_len, j_len, flag
  integer counter(drv%nc,drv%nr)
  real(r8) begwatb,endwatb !@ beginning and ending water balance over ENTIRE domain
  real(r8) tot_infl_mm,tot_tran_veg_mm,tot_drain_mm !@ total mm of h2o from infiltration and transpiration
  real(r8) error !@ mass balance error over entire domain
  real(r8) evap_trans_data(drv%nc,drv%nr,parfl_nlevsoi),press
  
!@ Variable declarations: write *.pfb file
  real(r8) value
  
! End of variable declaration 

  Write(*,*)"========== start the loop over the flux ============="
 
! also: arbitrary cutoff value for evap rate (if rate is too small problems with Parflow solver)
  do t=1,drv%nch     !rows (y)
    do l = 1, nlevsoi
    if (l == 1) then
      clm(t)%pf_flux(l)=(-clm(t)%qflx_tran_veg*clm(t)%rootfr(l)) + clm(t)%qflx_infl
    else  
      clm(t)%pf_flux(l) = - clm(t)%qflx_tran_veg*clm(t)%rootfr(l)
    endif
    enddo
  enddo

  counter = 0
  do  k=1, parfl_nlevsoi ! PF loop over z
    do  t=1, drv%nch
    i=tile(t)%col
    j=tile(t)%row
    if (clm(t)%topo_mask(parfl_nlevsoi-k+1) == 0) then
      value = 0.0d0                              
      evap_trans_data(i,j,k) = value
    elseif(clm(t)%topo_mask(parfl_nlevsoi-k+1) >= 1) then 
      value = clm(t)%pf_flux(nlevsoi-counter(i,j))*3.6d0/drv%dz 
      evap_trans_data(i,j,k) = value
      counter(i,j) = counter(i,j) + 1
    endif 
    end do
  end do
  

!@ Start: Here we do the mass balance: We look at every tile/cell individually!
!@ Determine volumetric soil water
  begwatb = 0.0d0
  endwatb = 0.0d0
  tot_infl_mm = 0.0d0
  tot_tran_veg_mm = 0.0d0
  tot_drain_mm = 0.0d0
  
  do t=1,drv%nch  !@ Start: Loop over domain 
  i=tile(t)%col
  j=tile(t)%row
  if (clm(t)%planar_mask == 1) then !@ do only if we are in active domain   
    
    do l = 1, nlevsoi
    clm(t)%h2osoi_vol(l) = clm(t)%h2osoi_liq(l)/(clm(t)%dz(l)*denh2o) &
                           + clm(t)%h2osoi_ice(l)/(clm(t)%dz(l)*denice)
    enddo
      
    !@ Let's do it my way
    !@ Here we add the total water mass of the layers below CLM soil layers from Parflow to close water balance
    !@ We can use clm(1)%dz(1) because the grids are equidistant and congruent
     clm(t)%endwb=0.0d0 !@only interested in wb below surface
    
     do l = 1, parfl_nlevsoi ! CLM loop over z
       if (clm(t)%pf_press(l) > 0.0d0 .and. clm(t)%topo_mask(l)==1 ) then
         clm(t)%endwb = clm(t)%endwb + clm(t)%pf_press(l)
       endif
       clm(t)%endwb = clm(t)%endwb + clm(t)%pf_vol_liq(l) * clm(1)%dz(1) * 1000.0d0
       clm(t)%endwb = clm(t)%endwb + clm(t)%pf_vol_liq(l)/clm(t)%watsat(l) * 0.0001*clm(1)%dz(1) * clm(t)%pf_press(l)    
     enddo
      
    !@ Water balance over the entire domain
     begwatb = begwatb + clm(t)%begwb
     endwatb = endwatb + clm(t)%endwb
     tot_infl_mm = tot_infl_mm + clm(t)%qflx_infl_old * clm(1)%dtime
     tot_tran_veg_mm = tot_tran_veg_mm + clm(t)%qflx_tran_veg_old * clm(1)%dtime

   ! Determine wetland and land ice hydrology (must be placed here since need snow 
   ! updated from clm_combin) and ending water balance
   !@ Does my new way of doing the wb influence this?! 05/26/2004

     if (clm(t)%itypwat==istwet .or. clm(t)%itypwat==istice) call clm_hydro_wetice (clm(t))

! -----------------------------------------------------------------
! Energy AND Water balance for lake points
! -----------------------------------------------------------------
       
     if (clm(t)%lakpoi) then    
!      call clm_lake (clm)             @Stefan: This subroutine is still called from clm_main; why? 05/26/2004
      
       do l = 1, nlevsoi
       clm(t)%h2osoi_vol(l) = 1.0
       enddo  
     
     endif

! -----------------------------------------------------------------
! Update the snow age
! -----------------------------------------------------------------

!    call clm_snowage (clm)           @Stefan: This subroutine is still called from clm_main

! -----------------------------------------------------------------
! Check the energy and water balance
! -----------------------------------------------------------------
 
     call clm_balchk (clm(t), clm(t)%istep) !@ Stefan: in terms of wb, this call is obsolete;
                                               !@ energy balances are still calculated

  endif !@ mask statement
  enddo !@ End: Loop over domain  

  
  error = 0.0d0
  error = endwatb - begwatb - (tot_infl_mm - tot_tran_veg_mm) ! + tot_drain_mm
 
! SGS failed to compile with gfortran  
  write(199,'(1i,1x,f,1x,5e)') clm(1)%istep,drv%time,error,tot_infl_mm,tot_tran_veg_mm,begwatb,endwatb
  !print *,""
  !print *,"Error (%):",error
!@ End: mass balance  
  
!@ Pass sat_flag to sat_flag_o
! drv%sat_flag_o = drv%sat_flag

end subroutine pf_couple

