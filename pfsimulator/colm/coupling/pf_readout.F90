!subroutine pfreadout(clm,drv,tile,saturation,pressure,rank,ix,iy,nx,ny,nz, j_incr,k_incr,ip)
subroutine pfreadout(saturation,pressure,nx,ny,nz,j_incr,k_incr,numpatch,topo_mask,planar_mask)

  !use drv_module          ! 1-D Land Model Driver variables
  !!use dfport
  !use precision
  !use drv_tilemodule      ! Tile-space variables
  !use clmtype
  !use clm_varpar, only : nlevsoi
  !use clm_varcon, only : denh2o

  USE MOD_Precision
  USE MOD_Vars_TimeInvariants, only: porsl
  USE MOD_Vars_TimeVariables, only: wliq_soisno
  USE MOD_Vars_1DFluxes, only: pf_vol_liq, pf_press
  USE MOD_Const_Physical, only: denh2o
  USE MOD_Vars_Global, only: nl_soil, dz_soi

  implicit none

  !type (drvdec) ,intent(inout) :: drv
  !type (tiledec) :: tile(drv%nch)
  !type (clm1d), intent(inout) :: clm(drv%nch)   !CLM 1-D Module
  integer:: nx, ny, nz, j_incr, k_incr, numpatch
  real(r8):: saturation((nx+2)*(ny+2)*(nz+2)), pressure((nx+2)*(ny+2)*(nz+2))
  integer:: topo_mask(3,nx*ny), planar_mask(3,nx*ny)
  integer:: i, j, k
  integer:: t, l

    do t = 1, numpatch
        if(planar_mask(3,t) == 1) then
            i = planar_mask(1,t)
            j = planar_mask(2,t)
            do k = 1, nl_soil
                l = 1+i + j_incr*(j) + k_incr*(topo_mask(1,t)-(k-1))  ! updated indexing #RMM
                pf_vol_liq(k,t)  = saturation(l) * porsl(k,t)
                pf_press(k,t)    = pressure(l) * 1000.d0
                wliq_soisno(k,t) = pf_vol_liq(k,t) * dz_soi(k) * denh2o   ! fixed DZ #RMM
            end do !k
        endif
    end do !t

end subroutine pfreadout
