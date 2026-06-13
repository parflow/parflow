!BHEADER**********************************************************************
!
!  Copyright (c) 1995-2009, Lawrence Livermore National Security,
!  LLC. Produced at the Lawrence Livermore National Laboratory. Written
!  by the Parflow Team (see the CONTRIBUTORS file)
!  <parflow@lists.llnl.gov> CODE-OCE!08-103. All rights reserved.
!
!  This file is part of Parflow. For details, see
!  http://www.llnl.gov/casc/parflow
!
!  Please read the COPYRIGHT file or Our Notice and the LICENSE file
!  for the GNU Lesser General Public License.
!
!  This program is free software; you can redistribute it and/or modify
!  it under the terms of the GNU General Publi!License (as published
!  by the Free Software Foundation) version 2.1 dated February 1999.
!
!  This program is distributed in the hope that it will be useful, but
!  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
!  and conditions of the GNU General Publi!License for more details.
!
!  You should have received a copy of the GNU Lesser General Public
!  License along with this program; if not, write to the Free Software
!  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
!  USA
!**********************************************************************EHEADER
! advect: this file contains 5 primary subroutines to simulate advective 
! transport, advect_upwind, advect_highorder, advect_transverse, advect_limit,
! and advect_computeconcen
!
! author: Joe Beisman
!-------------------------------------------------------------------------------
!    advect_upwind: computes first-order upwind fluxes, updates concentrations
!    can handle transient saturations
!    
!    the new low-order concentrations should always be positive and monotone 
!    in time, although small errors (~ 0.3%) may arise in the case of transient 
!    saturations 
!-------------------------------------------------------------------------------
  subroutine advect_upwind(s,sn,uedge,vedge,wedge,old_porsat, &
      new_porsat_inv,fx,fy,fz,dlo,dhi,hx,dims,dt)

  use, intrinsic :: iso_c_binding

  implicit none
  integer, parameter          :: dp = selected_real_kind(15, 307)
  integer(c_int), intent(in)  :: dlo(3), dhi(3), dims(3)
  real(c_double), intent(in)  :: s(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3) 
  real(c_double), intent(out) :: sn(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
  real(c_double), intent(in)  :: uedge(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2) 
  real(c_double), intent(in)  :: vedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+2) 
  real(c_double), intent(in)  :: wedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+3) 
  real(c_double), intent(in)  :: old_porsat(dlo(1)-1:dhi(1)+1,dlo(2)-1:dhi(2)+1,dlo(3)-1:dhi(3)+1)
  real(c_double), intent(in)  :: new_porsat_inv(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(c_double), intent(out) :: fx(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3) 
  real(c_double), intent(out) :: fy(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3) 
  real(c_double), intent(out) :: fz(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(in)  :: hx(3), dt

  integer  i,j,k
  integer  is,ie,js,je,ks,ke
  integer  ii,jj,kk
  real(dp) dt_dx,dt_dy,dt_dz

  is = dlo(1)
  ie = dhi(1)
  js = dlo(2)
  je = dhi(2)
  ks = dlo(3)
  ke = dhi(3)
  dt_dx = dt/hx(1)
  dt_dy = dt/hx(2)
  dt_dz = dt/hx(3)

  !x fluxes
  if (dims(1) == 1) then
    do k=ks,ke
      do j=js,je
        do i=is,ie+1

        if (uedge(i,j,k) >= 0.0)then
          ii =  i-1
        else
          ii =  i
        endif

        fx(i,j,k)=uedge(i,j,k)*s(ii,j,k)

        enddo
      enddo
    enddo
  else
    fx = 0.0_dp
  endif

  !y fluxes
  if (dims(2) == 1) then
    do k=ks,ke
      do j=js,je+1
        do i=is,ie

        if (vedge(i,j,k) >= 0.0)then
          jj =  j-1
        else
          jj =  j
        endif

        fy(i,j,k)=vedge(i,j,k)*s(i,jj,k)

        enddo
      enddo
    enddo
  else
    fy = 0.0_dp
  endif

  !z fluxes
  if (dims(3) == 1) then
    do k=ks,ke+1
      do j=js,je
        do i=is,ie

        if (wedge(i,j,k) >= 0.0)then
          kk =  k-1
        else
          kk =  k
        endif

         fz(i,j,k)=wedge(i,j,k)*s(i,j,kk)

         enddo
      enddo
    enddo
  else
    fz = 0.0_dp
  endif

  !these values give a first order scheme
  do k=ks,ke
    do j=js,je
      do i=is,ie

      sn(i,j,k) = (old_porsat(i,j,k)*s(i,j,k) + &
      dt_dx*(fx(i,j,k) - fx(i+1,j,k)) + dt_dy*(fy(i,j,k)-fy(i,j+1,k)) + &
      dt_dz*(fz(i,j,k)-fz(i,j,k+1))) * new_porsat_inv(i,j,k)

      enddo
    enddo
  enddo

  ! clear memory for reuse in other advect subroutines
  if (dims(1) == 1) fx = 0.0_dp
  if (dims(2) == 1) fy = 0.0_dp
  if (dims(3) == 1) fz = 0.0_dp

  return
  end subroutine advect_upwind


!-------------------------------------------------------------------------------
!    advect_highorder: computes high-order anti-diffusive flux corrections to 
!    be added to the low-order concentrations computed with advect_upwind
!    
!    employs the monotone-centered limiter to ensure montonicity in each 
!    primary dir
!
!    Lax-Wendroff style centered approximation with upwind-biased limiter
!-------------------------------------------------------------------------------
  subroutine advect_highorder(s,uedge,vedge,wedge, &
                             porsat_inv,sx,sy,sz,dlo,dhi,hx,dims,dt)

  use, intrinsic :: iso_c_binding

  implicit none
  integer, parameter             :: dp = selected_real_kind(15, 307)
  integer(c_int), intent (in)    :: dlo(3), dhi(3), dims(3)
  real(c_double), intent (in)    :: s(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
  real(c_double), intent (in)    :: uedge(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(c_double), intent (in)    :: vedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+2)
  real(c_double), intent (in)    :: wedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+3)
  real(c_double), intent (in)    :: porsat_inv(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(c_double), intent (inout) :: sx(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent (inout) :: sy(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent (inout) :: sz(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent (in)    :: hx(3), dt
  
  integer  i,j,k
  integer  is,ie,js,je,ks,ke
  integer  ii,jj,kk
  real(dp) dt_dx,dt_dy,dt_dz
  real(dp) half
  real(dp) rx,ry,rz
  real(dp) mclimit,thetax,thetay,thetaz
  real(dp) limx,limy,limz,abs_vel

  is = dlo(1)
  ie = dhi(1)
  js = dlo(2)
  je = dhi(2)
  ks = dlo(3)
  ke = dhi(3)
  dt_dx = dt/hx(1)
  dt_dy = dt/hx(2)
  dt_dz = dt/hx(3)
  half  = 0.5_dp

  !! compute second order corrections
  !x values
  if (dims(1) == 1) then
    do k=ks-1,ke+1
      do j=js-1,je+1
        do i=is-1,ie+2
 
          if (uedge(i,j,k) >= 0.0)then
            ii  = i-1
          else
            ii  = i+1
          endif

          rx=s(i,j,k)-s(i-1,j,k)

          if (rx /= 0.0)then
            thetax=(s(ii,j,k)-s(ii-1,j,k))/rx
            limx=mclimit(thetax)
            abs_vel = abs(uedge(i,j,k))
            sx(i,j,k) = half* abs_vel * &
            (1.0_dp-dt_dx*abs_vel * max(porsat_inv(i,j,k),porsat_inv(i-1,j,k)))*rx*limx
          endif

        enddo
      enddo
    enddo
  endif

  !y values
  if (dims(2) == 1) then
    do k=ks-1,ke+1
      do j=js-1,je+2
        do i=is-1,ie+1

          if (vedge(i,j,k) >= 0.0)then
            jj  = j-1
          else
            jj  = j+1
          endif

          ry=s(i,j,k)-s(i,j-1,k)

          if (ry /= 0.0_dp)then
            thetay=(s(i,jj,k)-s(i,jj-1,k))/ry
            limy=mclimit(thetay)
            abs_vel = abs(vedge(i,j,k))
            sy(i,j,k) = half* abs_vel * &
            (1.0_dp-dt_dy*abs_vel * max(porsat_inv(i,j,k),porsat_inv(i,j-1,k)))*ry*limy
          endif

        enddo
      enddo
    enddo
  endif

  !z values
  if (dims(3) == 1) then
    do k=ks-1,ke+2
      do j=js-1,je+1
        do i=is-1,ie+1

          if (wedge(i,j,k) >= 0.0)then
            kk  = k-1
          else
            kk  = k+1
          endif

          rz=s(i,j,k)-s(i,j,k-1)

          if (rz /= 0.0_dp)then
            thetaz=(s(i,j,kk)-s(i,j,kk-1))/rz
            limz=mclimit(thetaz)
            abs_vel = abs(wedge(i,j,k))
            sz(i,j,k) = half* abs_vel * &
            (1.0_dp- dt_dz*abs_vel * max(porsat_inv(i,j,k),porsat_inv(i,j,k-1)))*rz*limz
          endif

        enddo
      enddo
    enddo
  endif

  return
  end subroutine advect_highorder


!-------------------------------------------------------------------------------
!    advect_transverse: computes approximate solution to Riemann problems 
!    emanating from neighboring interfaces 
!    
!    computes transverse corrections to both the first-order increment waves
!    and second-order correction waves
!-------------------------------------------------------------------------------
  subroutine advect_transverse(s,uedge,vedge,wedge, &
      dlo,dhi,hx,dt,vx,wx,uy,wy,uz,vz, &
      sx,sy,sz)

  implicit none
  integer,  parameter     :: dp = selected_real_kind(15, 307)
  integer,  intent(in)    :: dlo(3), dhi(3)
  real(dp), intent(in)    :: hx(3), dt
  real(dp), intent(in)    :: s(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
  real(dp), intent(in)    :: uedge(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(dp), intent(in)    :: vedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+2)
  real(dp), intent(in)    :: wedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+3)
  real(dp), intent(out)   :: vx(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(out)   :: wx(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(out)   :: uy(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(out)   :: wy(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(out)   :: uz(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(out)   :: vz(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(dp), intent(inout) :: sx(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(dp), intent(inout) :: sy(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(dp), intent(inout) :: sz(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
 
  integer i,j,k
  integer is,ie,js,je,ks,ke
  integer ks1,ke1
  integer js1,je1
  integer is1,ie1
  integer ii2,jj2,kk2
  integer iiz,iiz2,jjx,jjx2,kky,kky2,iiy,jjz,kkx
  real(dp) dx_inv,dy_inv,dz_inv
  real(dp) half,third
  real(dp) rx,ry,rz
  real(dp) transvel

  is = dlo(1)
  ie = dhi(1)
  js = dlo(2)
  je = dhi(2)
  ks = dlo(3)
  ke = dhi(3)
  dx_inv = 1.0_dp/hx(1)
  dy_inv = 1.0_dp/hx(2)
  dz_inv = 1.0_dp/hx(3)
  half = 0.5_dp
  third = 1.0_dp/3.0_dp


  !! make smaller and faster for lower D problems
  if((ie-is) == 0) then
    is1 = is+1
    ie1 = ie-2
  else
    is1 = is
    ie1 = ie
  endif

  if((je-js) == 0) then
    js1 = js+1
    je1 = je-2
  else
    js1 = js
    je1 = je
  endif

  if((ke-ks) == 0) then
    ks1 = ks+1
    ke1 = ke-2
  else
    ks1 = ks
    ke1 = ke
  endif
  
  !! main loop
  !! compute increment fluxes and second order corrections
  !! compute transverse riemann problem, move fluxes appropriately
  do k=ks1-1,ke1+2
    do j=js1-1,je1+2
      do i=is1-1,ie1+2
  
      if (uedge(i,j,k) >= 0.0)then
        ii2 = i
      else
        ii2 = i-1
      endif
  
      if (vedge(i,j,k) >= 0.0)then
        jj2 = j
      else
        jj2 = j-1
      endif

      if (wedge(i,j,k) >= 0.0)then
        kk2 = k
      else
        kk2 = k-1
      endif

  
      !! transverse velocities - 0 if both don't have same sign
      vx(i,j,k) = (dt*dx_inv)*transvel(vedge(ii2,j,k),vedge(ii2,j+1,k)) 
      wx(i,j,k) = (dt*dx_inv)*transvel(wedge(ii2,j,k),wedge(ii2,j,k+1)) 
      uy(i,j,k) = (dt*dy_inv)*transvel(uedge(i,jj2,k),uedge(i+1,jj2,k)) 
      wy(i,j,k) = (dt*dy_inv)*transvel(wedge(i,jj2,k),wedge(i,jj2,k+1))
      uz(i,j,k) = (dt*dz_inv)*transvel(uedge(i,j,kk2),uedge(i+1,j,kk2))
      vz(i,j,k) = (dt*dz_inv)*transvel(vedge(i,j,kk2),vedge(i,j+1,kk2))
  
      !gradient
      rx=s(i,j,k)-s(i-1,j,k)
      ry=s(i,j,k)-s(i,j-1,k)
      rz=s(i,j,k)-s(i,j,k-1)
  
      
      if (vx(i,j,k) > 0.0)then
        jjx = j+1
        jjx2 = j+1
      else
        jjx = j
        jjx2 = j-1
      endif
    
      if (wx(i,j,k) > 0.0)then
        kkx = k+1
      else
        kkx = k
      endif
    
      if (uy(i,j,k) > 0.0)then
        iiy = i+1
      else
        iiy = i
      endif
    
      if (wy(i,j,k) > 0.0)then
        kky = k+1
        kky2 = k+1
      else
        kky = k
        kky2 = k-1
      endif
    
      if (uz(i,j,k) > 0.0)then
        iiz = i+1
        iiz2 = i+1
      else
        iiz = i
        iiz2 = i-1
      endif
    
      if (vz(i,j,k) > 0.0)then
        jjz = j+1
      else
        jjz = j
      endif
  
      ! these sweeps add transverse propogation information to the first order upwind method 
      !x sweep
      sy(ii2,jjx,k) = sy(ii2,jjx,k)-half*rx*uedge(i,j,k)*vx(i,j,k)
      sz(ii2,j,kkx)=sz(ii2,j,kkx) - half*rx*uedge(i,j,k)*wx(i,j,k) + &
      (third)*uedge(i,j,k)*abs(vx(i,j,k))*wx(i,j,k)*rx 
      sz(ii2,jjx2,kkx)=sz(ii2,jjx2,kkx) - (third)*uedge(i,j,k) * &
      abs(vx(i,j,k))*wx(i,j,k)*rx 
  
      !y sweep
      sz(i,jj2,kky) = sz(i,jj2,kky)-half*ry*vedge(i,j,k)*wy(i,j,k)
      sx(iiy,jj2,k)=sx(iiy,jj2,k) - half*ry*vedge(i,j,k)*uy(i,j,k) + &
      (third)*vedge(i,j,k)*abs(wy(i,j,k))*uy(i,j,k)*ry 
      sx(iiy,jj2,kky2)=sx(iiy,jj2,kky2) - (third)*vedge(i,j,k) * &
      abs(wy(i,j,k))*uy(i,j,k)*ry 
  
      !z sweep
      sx(iiz,j,kk2) = sx(iiz,j,kk2)-half*rz*wedge(i,j,k)*uz(i,j,k)
      sy(i,jjz,kk2)=sy(i,jjz,kk2) - half*rz*wedge(i,j,k)*vz(i,j,k) + &
      (third)*wedge(i,j,k)*abs(uz(i,j,k))*vz(i,j,k)*rz 
      sy(iiz2,jjz,kk2)=sy(iiz2,jjz,kk2) - (third)*wedge(i,j,k) * &
      abs(uz(i,j,k))*vz(i,j,k)*rz
  
      ! Add second order transverse propogation information
      !x sweep
      sy(i,jjx,k)      = sy(i,jjx,k) + vx(i,j,k)*sx(i,j,k)
      sy(i-1,jjx,k)    = sy(i-1,jjx,k) - vx(i,j,k)*sx(i,j,k)
      sz(i,j,kkx)      = sz(i,j,kkx) + (1.0_dp - abs(vx(i,j,k)))*wx(i,j,k)*sx(i,j,k)
      sz(i,jjx2,kkx)   = sz(i,jjx2,kkx) + abs(vx(i,j,k))*wx(i,j,k)*sx(i,j,k)
      sz(i-1,j,kkx)    = sz(i-1,j,kkx) - (1.0_dp - abs(vx(i,j,k)))*wx(i,j,k)*sx(i,j,k)
      sz(i-1,jjx2,kkx) = sz(i-1,jjx2,kkx) - abs(vx(i,j,k))*wx(i,j,k)*sx(i,j,k)
  
      !y sweep
      sz(i,j,kky)      = sz(i,j,kky) + wy(i,j,k)*sy(i,j,k)
      sz(i,j-1,kky)    = sz(i,j-1,kky) - wy(i,j,k)*sy(i,j,k)
      sx(iiy,j,k)      = sx(iiy,j,k) + (1.0_dp - abs(wy(i,j,k)))*uy(i,j,k)*sy(i,j,k)
      sx(iiy,j,kky2)   = sx(iiy,j,kky2) + abs(wy(i,j,k))*uy(i,j,k)*sy(i,j,k)
      sx(iiy,j-1,k)    = sx(iiy,j-1,k) - (1.0_dp - abs(wy(i,j,k)))*uy(i,j,k)*sy(i,j,k)
      sx(iiy,j-1,kky2) = sx(iiy,j-1,kky2) - abs(wy(i,j,k))*uy(i,j,k)*sy(i,j,k)
  
      !z sweep
      sx(iiz,j,k)      = sx(iiz,j,k) + uz(i,j,k)*sz(i,j,k)
      sx(iiz,j,k-1)    = sx(iiz,j,k-1) - uz(i,j,k)*sz(i,j,k)
      sy(i,jjz,k)      = sy(i,jjz,k) + (1.0_dp - abs(uz(i,j,k)))*vz(i,j,k)*sz(i,j,k)
      sy(iiz2,jjz,k)   = sy(iiz2,jjz,k) + abs(uz(i,j,k))*vz(i,j,k)*sz(i,j,k)
      sy(i,jjz,k-1)    = sy(i,jjz,k-1) - (1.0_dp - abs(uz(i,j,k)))*vz(i,j,k)*sz(i,j,k)
      sy(iiz2,jjz,k-1) = sy(iiz2,jjz,k-1) - abs(uz(i,j,k))*vz(i,j,k)*sz(i,j,k)
      
      enddo
    enddo
  enddo

  return
  end subroutine advect_transverse


!-------------------------------------------------------------------------------
!    advect_computeconcen: adds high-order corrections from advect_highorder
!    and transverse corrections from advect_transverse to low-order solution 
!    from advect_upwind 
!
!    Also accounts for transient saturations    
!-------------------------------------------------------------------------------
  subroutine advect_computeconcen(sn,sx,sy,sz, &
                                porsat_inv,dlo,dhi,hx,dt)

  use, intrinsic :: iso_c_binding

  implicit none
  integer, parameter            :: dp = selected_real_kind(15, 307)
  integer(c_int), intent(in)    :: dlo(3), dhi(3)
  real(c_double), intent(inout) :: sn(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
  real(c_double), intent(in)    :: sx(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(in)    :: sy(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(in)    :: sz(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(in)    :: porsat_inv(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(c_double), intent(in)    :: hx(3), dt
 
  integer i,j,k
  integer is,ie,js,je,ks,ke
  real(dp) dt_dx,dt_dy,dt_dz

  is = dlo(1)
  ie = dhi(1)
  js = dlo(2)
  je = dhi(2)
  ks = dlo(3)
  ke = dhi(3)
  dt_dx = dt/hx(1)
  dt_dy = dt/hx(2)
  dt_dz = dt/hx(3)

  do k=ks,ke
    do j=js,je
      do i=is,ie

       sn(i,j,k) = sn(i,j,k) + (dt_dx*(sx(i,j,k) - sx(i+1,j,k)) + &
       dt_dy*(sy(i,j,k)-sy(i,j+1,k)) + &
       dt_dz*(sz(i,j,k)-sz(i,j,k+1))) * porsat_inv(i,j,k)

      enddo
    enddo
  enddo

  return
  end subroutine advect_computeconcen


!-------------------------------------------------------------------------------
!    advect_limit: multi-dimensional limiter to ensure solution is TVD and that
!    min-max is not violated
!
!    modeled after Zalesak FCT limiter    
!-------------------------------------------------------------------------------
  subroutine advect_limit(sn,sx,sy,sz,dlo,dhi,hx,dims,dt,&
                   porsat_inv, p_plus,p_minus,q_plus,q_minus,r_plus,r_minus)
  use, intrinsic :: iso_c_binding

  implicit none
  integer,  parameter           :: dp = selected_real_kind(15, 307)
  integer(c_int), intent(in)    :: dlo(3), dhi(3), dims(3)
  real(c_double), intent(in)    :: dt,hx(3)
  real(c_double), intent(in)    :: sn(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
  real(c_double), intent(inout) :: sx(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3) 
  real(c_double), intent(inout) :: sy(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(inout) :: sz(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+3)
  real(c_double), intent(in)    :: porsat_inv(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2)
  real(c_double), intent(out)   :: p_plus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2) 
  real(c_double), intent(out)   :: p_minus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(c_double), intent(out)   :: q_plus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2) 
  real(c_double), intent(out)   :: q_minus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)
  real(c_double), intent(out)   :: r_plus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2) 
  real(c_double), intent(out)   :: r_minus(dlo(1)-1:dhi(1)+2,dlo(2)-1:dhi(2)+2,dlo(3)-1:dhi(3)+2)

  integer  is,ie,js,je,ks,ke,i,j,k 
  real(dp) dt_dx,dt_dy,dt_dz

  dt_dx = dt/hx(1)
  dt_dy = dt/hx(2)
  dt_dz = dt/hx(3)

  is = dlo(1)
  ie = dhi(1)
  js = dlo(2)
  je = dhi(2)
  ks = dlo(3)
  ke = dhi(3)
         
  do k=ks-1,ke+1
    do j=js-1,je+1
      do i=is-1,ie+1

        p_plus(i,j,k) = dt_dx*(max(0.0_dp,sx(i,j,k)) - min(0.0_dp,sx(i+1,j,k))) + &
        dt_dy*(max(0.0_dp,sy(i,j,k)) - min(0.0_dp,sy(i,j+1,k))) + &
        dt_dz*(max(0.0_dp,sz(i,j,k)) - min(0.0_dp,sz(i,j,k+1)))

        p_minus(i,j,k) = dt_dx*(max(0.0_dp,sx(i+1,j,k)) - min(0.0_dp,sx(i,j,k))) + &
        dt_dy*(max(0.0_dp,sy(i,j+1,k)) - min(0.0_dp,sy(i,j,k))) + &
        dt_dz*(max(0.0_dp,sz(i,j,k+1)) - min(0.0_dp,sz(i,j,k)))

        q_plus(i,j,k) =  max(sn(i-1,j,k),sn(i,j,k),sn(i+1,j,k), sn(i,j-1,k),sn(i,j+1,k), &
          sn(i,j,k-1),sn(i,j,k+1)) - sn(i,j,k)

        q_minus(i,j,k) = sn(i,j,k) - min(sn(i-1,j,k),sn(i,j,k),sn(i+1,j,k), sn(i,j-1,k), &
          sn(i,j+1,k), sn(i,j,k-1),sn(i,j,k+1))

        if (p_plus(i,j,k) > 0.0_dp) then
          r_plus(i,j,k) = min(q_plus(i,j,k)/(p_plus(i,j,k)*porsat_inv(i,j,k)),1.0_dp)
        else
          r_plus(i,j,k) = 0.0_dp
        endif 
        if (p_minus(i,j,k) > 0.0_dp) then
          r_minus(i,j,k) = min(q_minus(i,j,k)/(p_minus(i,j,k)*porsat_inv(i,j,k)),1.0_dp)
        else
          r_minus(i,j,k) = 0.0_dp
        endif


      enddo
    enddo
  enddo

  if (dims(1) == 1) then
    do k=ks,ke
      do j=js,je
        do i=is,ie+1

          if (sx(i,j,k) >= 0.0_dp) then
            sx(i,j,k) = sx(i,j,k) * min(r_plus(i,j,k),r_minus(i-1,j,k))
          else
            sx(i,j,k) = sx(i,j,k) * min(r_plus(i-1,j,k),r_minus(i,j,k))
          endif

        enddo
      enddo
    enddo
  endif

  if (dims(2) == 1) then
    do k=ks,ke
      do j=js,je+1
        do i=is,ie
     
          if (sy(i,j,k) >= 0.0_dp) then
             sy(i,j,k) = sy(i,j,k) * min(r_plus(i,j,k),r_minus(i,j-1,k))
          else
             sy(i,j,k) = sy(i,j,k) * min(r_plus(i,j-1,k),r_minus(i,j,k))
          endif

        enddo
      enddo
    enddo
  endif

  if (dims(3) == 1) then
    do k=ks,ke+1
      do j=js,je
        do i=is,ie
  
          if (sz(i,j,k) >= 0.0_dp) then
             sz(i,j,k) = sz(i,j,k) * min(r_plus(i,j,k),r_minus(i,j,k-1))
          else
             sz(i,j,k) = sz(i,j,k) * min(r_plus(i,j,k-1),r_minus(i,j,k))
          endif
  
        enddo
      enddo
    enddo
  endif

  return
  end subroutine advect_limit




!      !! dispersion calculations -- likely need to be updated for boundary conditions
!      subroutine disperse(sn,uedge,vedge,wedge,dlo,dhi,hx,dt) 
!      implicit none
!
!      integer, parameter :: dp = selected_real_kind(15)            
!      integer i,j,k,ie,is,je,js,ke,ks
!      integer dlo(3), dhi(3)
!      real(dp)  hx(3), dt, al, at,dx,dy,dz
!      
!      real(dp) uedge(dlo(1)-2:dhi(1)+3,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+2) 
!      real(dp) vedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+3,dlo(3)-2:dhi(3)+2) 
!      real(dp) wedge(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+3) 
!      real(dp) v_xx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_xy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_xz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) v_yy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_yx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_yz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) v_zz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_zx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) v_zy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) sn(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
!      real(dp) abs_vx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) abs_vy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) abs_vz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) d_xx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_xy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_xz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) d_yy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_yx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_yz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) d_zz(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_zx(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) d_zy(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      real(dp) dc_x(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) dc_y(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1) 
!      real(dp) dc_z(dlo(1):dhi(1)+1,dlo(2):dhi(2)+1,dlo(3):dhi(3)+1)
!      
!      al = .00625
!      at = .000625
!      
!      is = dlo(1)
!      ie = dhi(1)
!      js = dlo(2)
!      je = dhi(2)
!      ks = dlo(3)
!      ke = dhi(3)
!      dx = hx(1)
!      dy = hx(2)
!      dz = hx(3)
!  
!      !velocities
!      do k=ks,ke+1
!      do j=js,je+1
!      do i=is,ie+1
!      
!      d_xx(i,j,k) = 0.0
!      d_xy(i,j,k) = 0.0
!      d_xz(i,j,k) = 0.0
!      
!      d_yy(i,j,k) = 0.0
!      d_yx(i,j,k) = 0.0
!      d_yz(i,j,k) = 0.0
!      
!      d_zz(i,j,k) = 0.0
!      d_zx(i,j,k) = 0.0
!      d_zy(i,j,k) = 0.0
!      
!     
!      
!     
!     v_xx(i,j,k)=uedge(i,j,k)
!     v_xy(i,j,k) = sum(vedge(i-1:i,j:j+1,k))/4.0
!     v_xz(i,j,k) = sum(wedge(i-1:i,j,k:k+1))/4.0
!    
!     v_yy(i,j,k)=vedge(i,j,k) 
!     v_yx(i,j,k) = sum(uedge(i:i+1,j-1:j,k))/4.0
!     v_yz(i,j,k) = sum(wedge(i,j-1:j,k:k+1))/4.0
!     
!     v_zz(i,j,k)=wedge(i,j,k)
!     v_zx(i,j,k) = sum(uedge(i:i+1,j,k-1:k))/4.0
!     v_zy(i,j,k) = sum(vedge(i,j:j+1,k-1:k))/4.0
!     
!     abs_vx(i,j,k) = (v_xx(i,j,k)**2 + v_xy(i,j,k)**2 + v_xz(i,j,k)**2)**0.5
!     abs_vy(i,j,k) = (v_yy(i,j,k)**2 + v_yx(i,j,k)**2 + v_yz(i,j,k)**2)**0.5
!     abs_vz(i,j,k) = (v_zz(i,j,k)**2 + v_zx(i,j,k)**2 + v_zy(i,j,k)**2)**0.5 
!     
!     enddo
!     enddo
!     enddo
!      
!     
!      do i=is,ie+1
!      do j=js,je
!      do k=ks,ke
!      
!      if (abs_vx(i,j,k) /= 0.0) THEN
!      
!      d_xx(i,j,k) = at*abs_vx(i,j,k) + (al-at)*v_xx(i,j,k)*v_xx(i,j,k)/abs_vx(i,j,k)
!      else 
!      d_xx(i,j,k) = 0.0 
!      endif
!      enddo
!      enddo
!      enddo
!      
!      do i=is,ie
!      do j=js,je+1
!      do k=ks,ke
!      
!      if (abs_vy(i,j,k) /= 0.0) THEN
!      
!      d_yy(i,j,k) = at*abs_vy(i,j,k) + (al-at)*v_yy(i,j,k)*v_yy(i,j,k)/abs_vy(i,j,k)
!      else 
!      d_yy(i,j,k) = 0.0 
!      endif
!      enddo
!      enddo
!      enddo
!      
!      
!      do i=is,ie
!      do j=js,je
!      do k=ks,ke+1
!      
!      if (abs_vz(i,j,k) /= 0.0) THEN
!      
!      d_zz(i,j,k) = at*abs_vz(i,j,k) + (al-at)*v_zz(i,j,k)*v_zz(i,j,k)/abs_vz(i,j,k)
!      else 
!      d_zz(i,j,k) = 0.0 
!      endif
!      enddo
!      enddo
!      enddo
!    
!    
!      do k=ks,ke
!      do j=js,je
!      do i=is,ie
!       sn(i,j,k) = sn(i,j,k) + dt*(d_xx(i,j,k)*(sn(i-1,j,k)-sn(i,j,k))/(dy*dz) - d_xx(i+1,j,k)*(sn(i,j,k)-sn(i+1,j,k))/(dy*dz) + &
!       d_yy(i,j,k)*(sn(i,j-1,k)-sn(i,j,k))/(dx*dz) - d_yy(i,j+1,k)*(sn(i,j,k)-sn(i,j+1,k))/(dx*dz) + &
!       d_zz(i,j,k)*(sn(i,j,k-1)-sn(i,j,k))/(dx*dy) - d_zz(i,j,k+1)*(sn(i,j,k)-sn(i,j,k+1))/(dx*dy))
!      enddo
!      enddo
!      enddo
!      end subroutine disperse

      real(selected_real_kind(15)) function mclimit(theta)
      implicit none
      real(selected_real_kind(15)) theta   
      mclimit = max(0.0d0,min((1.0d0 + theta)/2.0d0,2.0d0,2.0d0*theta))
      end function mclimit 
  
      real(selected_real_kind(15)) function transvel(a,b)
      implicit none
      real(selected_real_kind(15)) a,b,prod,avg 
      prod = a*b
      if (prod > 0.0d0) then
      avg = (abs(a) + abs(b))/2.0d0
      transvel = sign(avg,a)
      else
      transvel = 0.0d0
      endif
      end function transvel 
  
  
!      real(selected_real_kind(15)) function minmod4(a,b,c,d)
!      implicit none
!      real(selected_real_kind(15)) a,b,c,d,one
!      one = 1.0d0
!      minmod4 = (1.0d0/2.0d0)*(sign(one,a)+sign(one,b))*(1.0d0/2.0d0)*&
!      (sign(one,a)+sign(one,c))*(1.0d0/2.0d0)*(sign(one,a)+sign(one,d)) &
!      * min(abs(a),abs(b),abs(c),abs(d))   
!      end function minmod4
!  
!      real(selected_real_kind(15)) function minmod2(a,b)
!      implicit none
!      real(selected_real_kind(15)) a,b,one
!      one = 1.0d0
!      minmod2 = (1.0d0/2.0d0)*(sign(one,a)+sign(one,b))*min(abs(a),abs(b))   
!      end function minmod2
!  
!      real(selected_real_kind(15)) function median(a,b,c)
!      implicit none
!      real(selected_real_kind(15)) a,b,c,minmod2
!      median = a + minmod2(b-a,c-a)    
!      end function median
  





