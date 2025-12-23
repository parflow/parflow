C BHEADER**********************************************************************
C 
C   Copyright (c) 1995-2009, Lawrence Livermore National Security,
C   LLC. Produced at the Lawrence Livermore National Laboratory. Written
C   by the Parflow Team (see the CONTRIBUTORS file)
C   <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
C 
C   This file is part of Parflow. For details, see
C   http://www.llnl.gov/casc/parflow
C 
C   Please read the COPYRIGHT file or Our Notice and the LICENSE file
C   for the GNU Lesser General Public License.
C 
C   This program is free software; you can redistribute it and/or modify
C   it under the terms of the GNU General Public License (as published
C   by the Free Software Foundation) version 2.1 dated February 1999.
C 
C   This program is distributed in the hope that it will be useful, but
C   WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
C   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
C   and conditions of the GNU General Public License for more details.
C 
C   You should have received a copy of the GNU Lesser General Public
C   License along with this program; if not, write to the Free Software
C   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
C   USA
C **********************************************************************EHEADER

c**********************************************************************
c     advect, slopexy, slopez
c     
c     Godunov advection routine
c     
c**********************************************************************


#include "fortran_port.inc"

c----------------------------------------------------------------------
c     advect:
c     Godunov advection routine
c----------------------------------------------------------------------

      subroutine advect(s,sn,uedge,vedge,wedge,phi,
     $     slx,sly,slz,
     $     lo,hi,dlo,dhi,hx,dt,fstord,
     $     sbot,s_top,sbotp,sfrt,sbck,sleft,sright,sfluxz,
     $     dxscr,dyscr,dzscr,dzfrm) 
      implicit none

c     ::: argument declarations

      integer lo(3), hi(3)
      integer dlo(3), dhi(3)
      real*8  hx(3), dt
      integer fstord

      real*8 s(dlo(1)-3:dhi(1)+3,
     $         dlo(2)-3:dhi(2)+3,
     $         dlo(3)-3:dhi(3)+3) 
      real*8 sn(dlo(1)-3:dhi(1)+3,
     $          dlo(2)-3:dhi(2)+3,
     $          dlo(3)-3:dhi(3)+3)

      real*8 uedge(dlo(1)-1:dhi(1)+2,
     $             dlo(2)-1:dhi(2)+1,
     $             dlo(3)-1:dhi(3)+1) 
      real*8 vedge(dlo(1)-1:dhi(1)+1,
     $             dlo(2)-1:dhi(2)+2,
     $             dlo(3)-1:dhi(3)+1) 
      real*8 wedge(dlo(1)-2:dhi(1)+2,
     $             dlo(2)-2:dhi(2)+2,
     $             dlo(3)-2:dhi(3)+3) 

      real*8 phi(dlo(1)-2:dhi(1)+2,
     $           dlo(2)-2:dhi(2)+2,
     $           dlo(3)-2:dhi(3)+2)

      real*8  slx(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2)
      real*8  sly(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2) 
      real*8  slz(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2, 3) 

      real*8  sbot(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3)
      real*8  s_top(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3)
      real*8  sbotp(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3)

      real*8  sbck(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3)
      real*8  sfrt(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3)

      real*8  sleft(dlo(1)-3:dhi(1)+3)
      real*8  sright(dlo(1)-3:dhi(1)+3)

      real*8  sfluxz(dlo(1)-3:dhi(1)+3)

      real*8  dxscr(dlo(1)-3:dhi(1)+3, 4)
      real*8  dyscr(dlo(2)-3:dhi(2)+3, 4)
      real*8  dzscr(dlo(1)-3:dhi(1)+3, 3)
      real*8  dzfrm(dlo(1)-3:dhi(1)+3, 3)

      logical firstord
      integer is, ie, js, je, ks, ke
      integer i, j, k, km, kc, kp, kt
      real*8  dx, dy, dz, dth, dxh, dyh, dzh
      real*8  dxi, dyi, dzi, phiinv
      real*8  tlo_xlo, tlo_xhi, tlo_ylo, tlo_yhi, tlo_zlo, tlo_zhi
      real*8  thi_xlo, thi_xhi, thi_ylo, thi_yhi, thi_zlo, thi_zhi
      real*8  tlo_x,thi_x,tlo_y,thi_y,tlo_z,thi_z
      real*8  sux,suy,suz,cux,cuy,cuz
      real*8  supw,supw_m,supw_p
      real*8  half

      data    half/0.5d0/

      is = lo(1)
      ie = hi(1)
      js = lo(2)
      je = hi(2)
      ks = lo(3)
      ke = hi(3)
      dx = hx(1)
      dy = hx(2)
      dz = hx(3)
      dxh = half*dx
      dyh = half*dy
      dzh = half*dz
      dth = half*dt
      dxi = 1./dx
      dyi = 1./dy
      dzi = 1./dz

      if (fstord .eq. 1) then
         firstord = .true.
      else
         firstord = .false.
      endif

      km = 3
      kc = 1
      kp = 2

c----------------------------------------------------------
c     k = ks-1, ke+1 loop
c----------------------------------------------------------


      if (.not. firstord) call  slopez(s,slz,ks-1,kc,lo,hi,
     $    dlo,dhi,dzscr,dzfrm)

      do k = ks-1,ke+1

         if (.not. firstord) then
           call slopexy(s,slx,sly,k,lo,hi,dlo,dhi,dxscr,dyscr)
           if (k .le. ke) then
              call  slopez(s,slz,k+1,kp,lo,hi,dlo,dhi,dzscr,dzfrm)
           endif
         endif

         do j=js-1,je+1

            do i=is-1,ie+1

               phiinv = 1./phi(i,j,k)

               tlo_xlo = s(i-1,j,k)
               tlo_xhi = s(  i,j,k)
               thi_xlo = s(  i,j,k)
               thi_xhi = s(i+1,j,k)
               tlo_ylo = s(i,j-1,k)
               tlo_yhi = s(i,  j,k)
               thi_ylo = s(i,  j,k)
               thi_yhi = s(i,j+1,k)
               tlo_zlo = s(i,j,k-1)
               tlo_zhi = s(i,j,  k)
               thi_zlo = s(i,j,  k)
               thi_zhi = s(i,j,k+1)

               if (.not. firstord) then
                  tlo_xlo = tlo_xlo + (half - 
     $              uedge(i,j,k)*dth*dxi/phi(i-1,j,k))*slx(i-1,j) 
                  tlo_xhi = tlo_xhi - (half + 
     $              uedge(i,j,k)*dth*dxi*phiinv)*slx(  i,j) 

                  thi_xlo = thi_xlo + (half - 
     $              uedge(i+1,j,k)*dth*dxi*phiinv)*slx(  i,j) 
                  thi_xhi = thi_xhi - (half +
     $              uedge(i+1,j,k)*dth*dxi/phi(i+1,j,k))*slx(i+1,j) 

                  tlo_ylo = tlo_ylo + (half -
     $              vedge(i,j,k)*dth*dyi/phi(i,j-1,k))*sly(i,j-1) 
                  tlo_yhi = tlo_yhi - (half + 
     $              vedge(i,j,k)*dth*dyi*phiinv)*sly(i,  j) 

                 thi_ylo = thi_ylo + (half - 
     $              vedge(i,j+1,k)*dth*dyi*phiinv)*sly(i,  j) 
                  thi_yhi = thi_yhi - (half +
     $              vedge(i,j+1,k)*dth*dyi/phi(i,j+1,k))*sly(i,j+1) 

                  tlo_zlo = tlo_zlo + (half -
     $              wedge(i,j,k)*dth*dzi/phi(i,j,k-1))*slz(i,j,km) 
                  tlo_zhi = tlo_zhi - (half +
     $              wedge(i,j,k)*dth*dzi*phiinv)*slz(i,j,kc) 

                  thi_zlo = thi_zlo + (half -
     $              wedge(i,j,k+1)*dth*dzi*phiinv)*slz(i,j,kc) 
                  thi_zhi = thi_zhi - (half +
     $              wedge(i,j,k+1)*dth*dzi/phi(i,j,k+1))*slz(i,j,kp) 

               endif

               if (uedge(i,j,k) .ge. 0.0) then
                  tlo_x = tlo_xlo
               else
                  tlo_x = tlo_xhi
               endif

               if (uedge(i+1,j,k) .ge. 0.0) then
                  thi_x = thi_xlo
               else
                  thi_x = thi_xhi
               endif

               if (vedge(i,j,k) .ge. 0.0) then
                  tlo_y = tlo_ylo
               else
                  tlo_y = tlo_yhi
               endif

               if (vedge(i,j+1,k) .ge. 0.0) then
                  thi_y = thi_ylo
               else
                  thi_y = thi_yhi
               endif

               if (wedge(i,j,k) .ge. 0.0) then
                  tlo_z = tlo_zlo
               else
                  tlo_z = tlo_zhi
               endif

               if (wedge(i,j,k+1) .ge. 0.0) then
                  thi_z = thi_zlo
               else
                  thi_z = thi_zhi
               endif

               sux = (uedge(i+1,j,k)*thi_x - uedge(i,j,k)*tlo_x)*dxi
               suy = (vedge(i,j+1,k)*thi_y - vedge(i,j,k)*tlo_y)*dyi
               suz = (wedge(i,j,k+1)*thi_z - wedge(i,j,k)*tlo_z)*dzi

               cux = s(i,j,k)*(uedge(i+1,j,k) - uedge(i,j,k))*dxi
               cuy = s(i,j,k)*(vedge(i,j+1,k) - vedge(i,j,k))*dyi
               cuz = s(i,j,k)*(wedge(i,j,k+1) - wedge(i,j,k))*dzi

               sleft(i+1)  =
     $              thi_xlo - dth*( suy + suz + cux ) * phiinv
               sright(i)   =
     $              tlo_xhi - dth*( suy + suz + cux ) * phiinv

               sbck(i,j+1) =
     $              thi_ylo - dth*( sux + suz + cuy ) * phiinv
               sfrt(i,j)   =
     $              tlo_yhi - dth*( sux + suz + cuy ) * phiinv

               sbotp(i,j)  =
     $              thi_zlo - dth*( sux + suy + cuz ) * phiinv
               s_top(i,j)   =
     $              tlo_zhi - dth*( sux + suy + cuz ) * phiinv

            enddo

c     ::: add x contribution to sn

            if ((k .ge. ks) .and. (k .le. ke)) then
               if ((j .ge. js) .and. (j .le. je)) then

                  do i=is,ie

                     if (uedge(i,j,k) .ge. 0.0) then
                        supw_m = sleft(i)
                     else
                        supw_m = sright(i)
                     endif
                     if (uedge(i+1,j,k) .ge. 0.0) then
                        supw_p = sleft(i+1)
                     else
                        supw_p = sright(i+1)
                     endif

                     sn(i,j,k) = s(i,j,k) -
     $                    dt*(uedge(i+1,j,k)*supw_p -
     $                        uedge(  i,j,k)*supw_m)/(dx*phi(i,j,k)) 

                  enddo

               endif
            endif

         enddo

c     ::: add y contributions to sn

         if ((k .ge. ks) .and. (k .le. ke)) then

            do j=js,je

               do i=is,ie

                  if (vedge(i,j,k) .ge. 0.0) then
                     supw_m = sbck(i,j)
                  else
                     supw_m = sfrt(i,j)
                  endif
                  if (vedge(i,j+1,k) .ge. 0.0) then
                     supw_p = sbck(i,j+1)
                  else
                     supw_p = sfrt(i,j+1)
                  endif

                  sn(i,j,k) = sn(i,j,k) -
     $                 dt*(vedge(i,j+1,k)*supw_p -
     $                     vedge(i,  j,k)*supw_m)/(dy*phi(i,j,k)) 

               enddo

            enddo

         endif

c     ::: add z contributions to sn

         if ((k .ge. ks) .and. (k .le. (ke+1))) then

            do j=js,je

               do i=is,ie

                  if (wedge(i,j,k) .ge. 0.0) then
                     supw = sbot(i,j)
                  else
                     supw = s_top(i,j)
                  endif
                  sfluxz(i) = wedge(i,j,k)*supw*dzi

               enddo

               if (k .eq. ks) then

                  do i=is,ie

                     sn(i,j,k)   = sn(i,j,k  ) +
     $                             dt*sfluxz(i)/phi(i,j,k)

                  enddo

               else if (k .eq. (ke+1)) then

                  do i=is,ie

                     sn(i,j,k-1) = sn(i,j,k-1) -
     $                             dt*sfluxz(i)/phi(i,j,k-1)

                  enddo

               else 

                  do i=is,ie

                     sn(i,j,k)   = sn(i,j,k  ) +
     $                             dt*sfluxz(i)/phi(i,j,k)
                     sn(i,j,k-1) = sn(i,j,k-1) -
     $                             dt*sfluxz(i)/phi(i,j,k-1)

                  enddo

               endif

            enddo

         endif

c     ::: this should be done by rolling indices

         do j=js,je

            do i=is,ie

               sbot(i,j) = sbotp(i,j)

            enddo

         enddo

c     ::: roll km, kc, and kp values

         kt = km
         km = kc
         kc = kp
         kp = kt

      enddo

      return
      end


c----------------------------------------------------------------------
c     slopexy:
c     Compute slopes in x and y
c----------------------------------------------------------------------

      subroutine slopexy (s,slx,sly,k,lo,hi,dlo,dhi,dxscr,dyscr)
      implicit none

      integer lo(3), hi(3), dlo(3), dhi(3)
      real*8 s(dlo(1)-3:dhi(1)+3,
     $         dlo(2)-3:dhi(2)+3,
     $         dlo(3)-3:dhi(3)+3)
      real*8 slx(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2)
      real*8 sly(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2)
      real*8 dxscr(dlo(1)-3:dhi(1)+3,4)
      real*8 dyscr(dlo(2)-3:dhi(2)+3,4)

      integer cen,lim,flag,fromm

      parameter( cen   = 1 )
      parameter( lim   = 2 )
      parameter( flag  = 3 )
      parameter( fromm = 4 )

      real*8 dpls,dmin,ds

      integer is,js,ie,je,i,j,k
      logical firstord
      real*8 zero,sixth,half,two3rd,one,two

      data firstord /.false./
      data zero,sixth,half,two3rd,one,two/
     $     0.d0,
     $     0.166666666666667d0,
     $     0.5d0,
     $     0.666666666666667d0,
     $     1.d0,
     $     2.d0/ 

      is = lo(1)
      js = lo(2)
      ie = hi(1)
      je = hi(2)

      if (firstord) then
         do j = js-1, je+1 
            do i = is-1, ie+1
               slx(i,j) = zero
               sly(i,j) = zero
            enddo
         enddo
         return
      endif

c     :: SLOPES in the X direction
      do j = js-1,je+1 

c     ::::: compute second order limited (fromm) slopes     
         do i = is-2,ie+2
            dxscr(i,cen) = half*(s(i+1,j,k)-s(i-1,j,k))
            dmin = two*(s(i,j,k)-s(i-1,j,k))
            dpls = two*(s(i+1,j,k)-s(i ,j,k))
            dxscr(i,lim)= min(abs(dmin),abs(dpls))
            if (dpls*dmin .lt. 0.0) then
               dxscr(i,lim) = zero
            endif
            dxscr(i,flag) = dsign(one,dxscr(i,cen))
            dxscr(i,fromm)= dxscr(i,flag)*
     $           min(dxscr(i,lim),abs(dxscr(i,cen))) 
         enddo

         do i = is-1,ie+1
            ds = two * two3rd * dxscr(i,cen) -
     $           sixth * (dxscr(i+1,fromm) + dxscr(i-1,fromm)) 
            slx(i,j) = dxscr(i,flag)*min(abs(ds),dxscr(i,lim))
         enddo

      enddo

c     ::::: SLOPES in the Y direction
      do i = is-1,ie+1 
         do j = js-2,je+2
            dyscr(j,cen) = half*(s(i,j+1,k)-s(i,j-1,k))
            dmin = two*(s(i,j,k)-s(i,j-1,k))
            dpls = two*(s(i,j+1,k)-s(i ,j,k))
            dyscr(j,lim)= min(abs(dmin),abs(dpls))
            if (dpls*dmin .lt. 0.0) then
               dyscr(j,lim) = zero
            endif
            dyscr(j,flag) = dsign(one,dyscr(j,cen))
            dyscr(j,fromm)= dyscr(j,flag)*
     $           min(dyscr(j,lim),abs(dyscr(j,cen))) 
         enddo

         do j = js-1,je+1
            ds = two * two3rd * dyscr(j,cen) -
     $           sixth * (dyscr(j+1,fromm) + dyscr(j-1,fromm)) 
            sly(i,j) = dyscr(j,flag)*min(abs(ds),dyscr(j,lim))
         enddo
      enddo

      return
      end


c----------------------------------------------------------------------
c     slopez:
c     Compute slopes in z
c----------------------------------------------------------------------

      subroutine slopez (s,slz,k,kk,lo,hi,dlo,dhi,dzscr,dzfrm)
      implicit none

      integer is,js,ie,je,i,j,k,kk,kt
      integer lo(3), hi(3), dlo(3), dhi(3)
      real*8 s(dlo(1)-3:dhi(1)+3, dlo(2)-3:dhi(2)+3, dlo(3)-3:dhi(3)+3)
      real*8 slz(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2, 3)
      real*8 dzscr(dlo(1)-3:dhi(1)+3,3)
      real*8 dzfrm(dlo(1)-3:dhi(1)+3,k-1:k+1)

      integer cen,lim,flag
      parameter( cen = 1 )
      parameter( lim = 2 )
      parameter( flag = 3 )

      real*8 dpls,dmin,ds


      logical firstord
      real*8 zero,sixth,half,two3rd,one,two

      data firstord /.false./
      data zero,sixth,half,two3rd,one,two/
     $     0.d0,
     $     0.166666666666667d0,
     $     0.5d0,
     $     0.666666666666667d0,
     $     1.d0,
     $     2.d0/ 

      is = lo(1)
      js = lo(2)
      ie = hi(1)
      je = hi(2)

      if (firstord) then
         do j = js-1, je+1 
            do i = is-1, ie+1
               slz(i,j,kk) = zero
            enddo
         enddo
         return
      endif

c     ::::: SLOPES in the Z direction
      do j = js-1,je+1 
         do i = is-1,ie+1 
            kt = k-1
            dzscr(i,cen) = half*(s(i,j,kt+1)-s(i,j,kt-1))
            dmin = two*(s(i,j,kt)-s(i,j,kt-1))
            dpls = two*(s(i,j,kt+1)-s(i ,j,kt))
            dzscr(i,lim)= min(abs(dmin),abs(dpls))
            if (dpls*dmin .lt. 0.0) then
               dzscr(i,lim) = zero
            endif
            dzscr(i,flag) = dsign(one,dzscr(i,cen))
            dzfrm(i,kt)= dzscr(i,flag)*
     $           min(dzscr(i,lim),abs(dzscr(i,cen)))

            kt = k+1
            dzscr(i,cen) = half*(s(i,j,kt+1)-s(i,j,kt-1))
            dmin = two*(s(i,j,kt)-s(i,j,kt-1))
            dpls = two*(s(i,j,kt+1)-s(i ,j,kt))
            dzscr(i,lim)= min(abs(dmin),abs(dpls))
            if (dpls*dmin .lt. 0.0) then
               dzscr(i,lim) = zero
            endif
            dzscr(i,flag) = dsign(one,dzscr(i,cen))
            dzfrm(i,kt)= dzscr(i,flag)*
     $           min(dzscr(i,lim),abs(dzscr(i,cen)))

            dzscr(i,cen) = half*(s(i,j,k+1)-s(i,j,k-1))
            dmin = two*(s(i,j,k)-s(i,j,k-1))
            dpls = two*(s(i,j,k+1)-s(i ,j,k))
            dzscr(i,lim)= min(abs(dmin),abs(dpls))
            if (dpls*dmin .lt. 0.0) then
               dzscr(i,lim) = zero
            endif
            dzscr(i,flag) = dsign(one,dzscr(i,cen))
            dzfrm(i,k)= dzscr(i,flag)*
     $           min(dzscr(i,lim),abs(dzscr(i,cen))) 
         enddo

         do i = is-1,ie+1 
            ds = two * two3rd * dzscr(i,cen) -
     $           sixth * (dzfrm(i,k+1) + dzfrm(i,k-1)) 
            slz(i,j,kk) = dzscr(i,flag)*min(abs(ds),dzscr(i,lim))
         enddo
      enddo

      return
      end
