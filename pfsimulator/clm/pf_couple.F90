subroutine pf_couple(drv,clm,tile,evap_trans,saturation,pressure,porosity,nx,ny,nz,j_incr,k_incr,ip,istep_pf)

  use drv_module          ! 1-D Land Model Driver variables
  use precision
  use clmtype
  use drv_tilemodule      ! Tile-space variables
  use clm_varpar, only : nlevsoi
  use clm_varcon, only : denh2o, denice, istwet, istice
  implicit none

  type (drvdec):: drv
  type (clm1d) :: clm(drv%nch)     ! CLM 1-D Module
  type (tiledec) :: tile(drv%nch)
  integer,intent(in) :: istep_pf 

  integer i,j,k,l,t,ip
  integer nx,ny,nz, j_incr, k_incr
  ! real(r8) begwatb,endwatb !@ beginning and ending water balance over ENTIRE domain
  real(r8) tot_infl_mm,tot_tran_veg_mm,tot_drain_mm, total_soil_resistance !@ total mm of h2o from infiltration and transpiration
  real(r8) error !@ mass balance error over entire domain
  real(r8) evap_trans((nx+2)*(ny+2)*(nz+2))
  real(r8) saturation((nx+2)*(ny+2)*(nz+2)),pressure((nx+2)*(ny+2)*(nz+2))
  real(r8) porosity((nx+2)*(ny+2)*(nz+2))
  real(r8) abs_transpiration

  ! End of variable declaration 

  ! Write(*,*)"========== start the loop over the flux ============="

  ! @RMM Copy fluxes back into ParFlow
  ! print*, ' in pf_couple'
  ! print*,  ip, j_incr, k_incr
  ! evap_trans = 0.d0
  do t=1,drv%nch     
     i=tile(t)%col
     j=tile(t)%row
     if (clm(t)%planar_mask==1) then
        !! RZ water stress * T distribution
        if (clm(t)%rzwaterstress == 1) then
        total_soil_resistance = 0.0d0
        do k = 1, nlevsoi
        total_soil_resistance = total_soil_resistance + clm(t)%soil_resistance(k)*clm(t)%rootfr(k)
        end do  !k over soil column
        end if

        if (clm(t)%rzwaterstress == 0)  then  !! check what kind of water stress formulation we are using to distribute T
        do k = 1, nlevsoi
           l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))    ! updated indexing @RMM 4-12-09
           abs_transpiration = 0.0
           if (clm(t)%qflx_tran_veg >= 0.0) abs_transpiration = clm(t)%qflx_tran_veg
           if (k == 1) then
              clm(t)%pf_flux(k)=(-abs_transpiration*clm(t)%rootfr(k)) + clm(t)%qflx_infl + clm(t)%qflx_qirr_inst(k)
        !!print*, 'Beta:',(-clm(t)%qflx_tran_veg*clm(t)%rootfr(k)),clm(t)%qflx_infl,saturation(l),pressure(l)
           else  
              clm(t)%pf_flux(k)=(-abs_transpiration*clm(t)%rootfr(k)) + clm(t)%qflx_qirr_inst(k)
           endif
           ! copy back to pf, assumes timing for pf is hours and timing for clm is seconds
           ! IMF: replaced drv%dz with clm(t)%dz to allow variable DZ...
           evap_trans(l) = clm(t)%pf_flux(k) * 3.6d0 / clm(t)%dz(k)
        enddo  !! soil loop for uniform T
        else    !! weighted Transpiration over RZ by water stress
        !!  first check to see if total_soil_resistance is zero, set to 1.  This should only happen if T is turned off (and btran ==0)
        if(total_soil_resistance == 0.0) total_soil_resistance = 1.0d0

        do k = 1, nlevsoi
            l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))    ! updated indexing @RMM 4-12-09
        if (k == 1) then
            clm(t)%pf_flux(k)=(-clm(t)%qflx_tran_veg*clm(t)%rootfr(k)*clm(t)%soil_resistance(k))/ (total_soil_resistance)  &
            + clm(t)%qflx_infl + clm(t)%qflx_qirr_inst(k)
        else
            clm(t)%pf_flux(k)=(-clm(t)%qflx_tran_veg*clm(t)%rootfr(k)*clm(t)%soil_resistance(k))/ (total_soil_resistance)  &
            + clm(t)%qflx_qirr_inst(k)
        endif
        ! copy back to pf, assumes timing for pf is hours and timing for clm is seconds
        ! IMF: replaced drv%dz with clm(t)%dz to allow variable DZ...
        evap_trans(l) = clm(t)%pf_flux(k) * 3.6d0 / clm(t)%dz(k)
        enddo   !! end loop for non-uniform T
        end if   !! if for rzwaterstress

     ! else
     !    do k = 1, nlevsoi
     !       l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
     !       clm(t)%pf_flux(k) = 0.0
     !       evap_trans(l) = 0.0
     !    enddo
     endif
  enddo

  !@ Start: Here we do the mass balance: We look at every tile/cell individually!
  !@ Determine volumetric soil water
  !begwatb = 0.0d0
  drv%endwatb = 0.0d0
  tot_infl_mm = 0.0d0
  tot_tran_veg_mm = 0.0d0
  tot_drain_mm = 0.0d0

  do t=1,drv%nch   !@ Start: Loop over domain 
     i=tile(t)%col
     j=tile(t)%row
     if (clm(t)%planar_mask == 1) then !@ do only if we are in active domain   

        do l = 1, nlevsoi
           clm(t)%h2osoi_vol(l) = clm(t)%h2osoi_liq(l)/(clm(t)%dz(l)*denh2o) &
                                  + clm(t)%h2osoi_ice(l)/(clm(t)%dz(l)*denice)
        enddo

        ! @sjk Let's do it my way
        ! @sjk Here we add the total water mass of the layers below CLM soil layers from Parflow to close water balance
        ! @sjk We can use clm(1)%dz(1) because the grids are equidistant and congruent
        clm(t)%endwb=0.0d0 !@sjk only interested in wb below surface
        do k = clm(t)%topo_mask(3), clm(t)%topo_mask(1) ! CLM loop over z, starting at bottom of pf domains topo_mask(3)

           l = 1+i + j_incr*(j) + k_incr*(k)  ! updated indexing @RMM b/c we are looping from k3 to k1

           ! first we add direct amount of water: S*phi
           clm(t)%endwb = clm(t)%endwb + saturation(l)*porosity(l) * clm(1)%dz(1) * 1000.0d0

           ! then we add the compressible storage component, note the Ss is hard-wired here at 0.0001 should really be done in PF w/ real values
           clm(t)%endwb = clm(t)%endwb + saturation(l) * 0.0001*clm(1)%dz(1) * pressure(l) *1000.d0    

        enddo

        ! add height of ponded water at surface (ie pressure head at upper pf bddy if > 0) 	 
        l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1))
        if (pressure(l) > 0.d0 ) then
           clm(t)%endwb = clm(t)%endwb + pressure(l) * 1000.0d0
        endif

        !@ Water balance over the entire domain
        drv%endwatb = drv%endwatb + clm(t)%endwb
        tot_infl_mm = tot_infl_mm + clm(t)%qflx_infl_old * clm(1)%dtime
        tot_tran_veg_mm = tot_tran_veg_mm + clm(t)%qflx_tran_veg_old * clm(1)%dtime

        ! Determine wetland and land ice hydrology (must be placed here since need snow 
        ! updated from clm_combin) and ending water balance
        !@sjk Does my new way of doing the wb influence this?! 05/26/2004
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

        !call clm_balchk (clm(t), clm(t)%istep) !@ Stefan: in terms of wb, this call is obsolete;
        call clm_balchk (clm(t), istep_pf) !@ Stefan: in terms of wb, this call is obsolete;
        !@ energy balances are still calculated

     endif !@ mask statement
  enddo !@ End: Loop over domain, t



  error = 0.0d0
  error = drv%endwatb - drv%begwatb - (tot_infl_mm - tot_tran_veg_mm) ! + tot_drain_mm

  ! SGS according to standard "f" must have fw.d format, changed f -> f20.8, i -> i5 and e -> e10.2
  ! write(199,'(1i5,1x,f20.8,1x,5e13.5)') clm(1)%istep,drv%time,error,tot_infl_mm,tot_tran_veg_mm,drv%begwatb,drv%endwatb
  !write(199,'(1i5,1x,f20.8,1x,5e13.5)') istep_pf,drv%time,error,tot_infl_mm,tot_tran_veg_mm,drv%begwatb,drv%endwatb
  drv%begwatb =drv%endwatb
  !print *,"Error (%):",error
  !@ End: mass balance  

  !@ Pass sat_flag to sat_flag_o
  ! drv%sat_flag_o = drv%sat_flag

end subroutine pf_couple

