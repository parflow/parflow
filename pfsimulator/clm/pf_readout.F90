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

do t=1,drv%nch
i=tile(t)%col
j=tile(t)%row
  do k = 1, nlevsoi
     l = 1+i + j_incr*(j) + k_incr*(clm(t)%topo_mask(1)-(k-1))  ! updated indexing @RMM
     if(clm(t)%planar_mask == 1) then
      clm(t)%pf_vol_liq(k) = saturation(l) * clm(t)%watsat(k)
      clm(t)%pf_press(k)   = pressure(l) * 1000.d0
      clm(t)%h2osoi_liq(k) = clm(t)%pf_vol_liq(k)*clm(t)%dz(1)*denh2o
      endif
  end do !k
  
end do !t

end subroutine pfreadout
