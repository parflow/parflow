!subroutine pf_couple(drv,clm,tile,evap_trans,saturation,pressure,porosity,nx,ny,nz,j_incr,k_incr,ip,istep_pf)
subroutine pf_couple(evap_trans,saturation,pressure,porosity,pf_dz_mult,pdz,nx,ny,nz,j_incr,k_incr,numpatch, &
   topo_mask,planar_mask,deltim,begwatb)

  !use drv_module          ! 1-D Land Model Driver variables
  !use precision
  !use clmtype
  !use drv_tilemodule      ! Tile-space variables
  !use clm_varpar, only : nlevsoi
  !use clm_varcon, only : denh2o, denice, istwet, istice
   USE MOD_Precision
   USE MOD_Vars_TimeInvariants, only: patchclass
   USE MOD_Vars_TimeVariables, only: rootr
   USE MOD_Vars_1DFluxes, only: pf_flux, qinfl, etr, qinfl_old, etr_old !, qseva
   USE MOD_Vars_Global, only: nl_soil, dz_soi
   implicit none

  !type (drvdec):: drv
  !type (clm1d) :: clm(drv%nch)     ! CLM 1-D Module
  !type (tiledec) :: tile(drv%nch)
  !integer,intent(in) :: istep_pf 

  integer i,j,k,l,t,m
  integer nx,ny,nz,j_incr,k_incr,numpatch
  !real(r8) begwatb,endwatb !@ beginning and ending water balance over ENTIRE domain
  real(r8) tot_infl_mm,tot_tran_veg_mm !@ total mm of h2o from infiltration and transpiration
  !real(r8) error !@ mass balance error over entire domain
  real(r8) evap_trans((nx+2)*(ny+2)*(nz+2))
  real(r8) saturation((nx+2)*(ny+2)*(nz+2)),pressure((nx+2)*(ny+2)*(nz+2))
  real(r8) porosity((nx+2)*(ny+2)*(nz+2)),pf_dz_mult((nx+2)*(ny+2)*(nz+2))
  real(r8) abs_transpiration, begwatb, endwatb, endwb, deltim, error, pdz
  integer:: topo_mask(3,nx*ny), planar_mask(3,nx*ny)

  ! End of variable declaration 

  ! Write(*,*)"========== start the loop over the flux ============="

  ! @RMM Copy fluxes back into ParFlow
  ! print*, ' in pf_couple'
  ! print*,  ip, j_incr, k_incr
  ! evap_trans = 0.d0
   do t = 1, numpatch     
         if (planar_mask(3,t) == 1) then
            i = planar_mask(1,t)
            j = planar_mask(2,t)
            !m = patchclass(t)
            do k = 1, nl_soil
               l = 1+i + j_incr*(j) + k_incr*(topo_mask(1,t)-(k-1))    ! updated indexing @RMM 4-12-09
               abs_transpiration = 0.0
               if (etr(t) >= 0.0) abs_transpiration = etr(t)
               if (k == 1) then
                  pf_flux(k,t)=(-abs_transpiration*rootr(k,t)) + qinfl(t) !- qseva(t)
            !!print*, 'Beta:',(-clm(t)%qflx_tran_veg*clm(t)%rootr(k)),clm(t)%qflx_infl,saturation(l),pressure(l)
               else  
                  pf_flux(k,t)=(-abs_transpiration*rootr(k,t))
               endif
               ! copy back to pf, assumes timing for pf is hours and timing for clm is seconds
               ! IMF: replaced drv%dz with clm(t)%dz to allow variable DZ...
               evap_trans(l) = pf_flux(k,t) * 3.6d0 / dz_soi(k)
            enddo
         ! else
         !    do k = 1, nlevsoi
         !       l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))
         !       clm(t)%pf_flux(k) = 0.0
         !       evap_trans(l) = 0.0
         !    enddo
         endif
   enddo

   endwatb = 0.0d0
   tot_infl_mm = 0.0d0
   tot_tran_veg_mm = 0.0d0

   do t = 1, numpatch
      if (planar_mask(3,t) == 1) then
         i = planar_mask(1,t)
         j = planar_mask(2,t)

         ! @sjk Let's do it my way
         ! @sjk Here we add the total water mass of the layers below CLM soil layers from Parflow to close water balance
         ! @sjk We can use clm(1)%dz(1) because the grids are equidistant and congruent
         endwb = 0.0d0 !@sjk only interested in wb below surface
         do k = topo_mask(3,t), topo_mask(1,t) ! CLM loop over z, starting at bottom of pf domains topo_mask(3)

            l = 1+i + j_incr*(j) + k_incr*(k)  ! updated indexing @RMM b/c we are looping from k3 to k1

            ! first we add direct amount of water: S*phi
            endwb = endwb + saturation(l) * porosity(l) * pdz*pf_dz_mult(l) * 1000.0d0

            ! then we add the compressible storage component, note the Ss is hard-wired here at 0.0001 should really be done in PF w/ real values
            endwb = endwb + saturation(l) * 0.0001 * pdz*pf_dz_mult(l) * pressure(l) * 1000.d0

         enddo

         ! add height of ponded water at surface (ie pressure head at upper pf bddy if > 0)
         l = 1+i + j_incr*(j) + k_incr*(topo_mask(1,t))
         if (pressure(l) > 0.d0 ) then
            endwb = endwb + pressure(l) * 1000.0d0
         endif

         !@ Water balance over the entire domain
         endwatb = endwatb + endwb
         tot_infl_mm = tot_infl_mm + qinfl_old(t) * deltim
         tot_tran_veg_mm = tot_tran_veg_mm + etr_old(t) * deltim

      endif
   enddo

   error = 0.0d0
   error = endwatb - begwatb - (tot_infl_mm - tot_tran_veg_mm) 

   ! SGS according to standard "f" must have fw.d format, changed f -> f20.8, i -> i5 and e -> e10.2
   ! write(199,'(1i5,1x,f20.8,1x,5e13.5)') clm(1)%istep,drv%time,error,tot_infl_mm,tot_tran_veg_mm,drv%begwatb,drv%endwatb
   !write(199,'(1i5,1x,f20.8,1x,5e13.5)') istep_pf,drv%time,error,tot_infl_mm,tot_tran_veg_mm,drv%begwatb,drv%endwatb
   begwatb = endwatb

   !print *,"Error (%):", error/begwatb

end subroutine pf_couple

