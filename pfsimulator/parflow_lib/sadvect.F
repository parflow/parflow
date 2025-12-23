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
c     sadvect, sslopexy, sslopez
c     
c     Godunov advection routine
c     
c**********************************************************************


#include "fortran_port.inc"

c----------------------------------------------------------------------
c     sadvect:
c     Godunov advection routine
c----------------------------------------------------------------------

      subroutine sadvect(s,sn,uedge,vedge,wedge,betaedge,phi,
     $     viscos,densty,gravty,
     $     slx,sly,slz,
     $     lohi,dlohi,hx,dt,
     $     sbot,s_top,sbotp,sfrt,sbck,sleft,sright,sfluxz,
     $     dxscr,dyscr,dzscr,dzfrm) 
      implicit none

c     ::: argument declarations

      integer lohi(3,2)
      integer dlohi(3,2)
      real*8  hx(3), dt

      real*8  s(dlohi(1,1)-3:dlohi(1,2)+3,
     $          dlohi(2,1)-3:dlohi(2,2)+3,
     $          dlohi(3,1)-3:dlohi(3,2)+3) 
      real*8  sn(dlohi(1,1)-3:dlohi(1,2)+3,
     $           dlohi(2,1)-3:dlohi(2,2)+3,
     $           dlohi(3,1)-3:dlohi(3,2)+3)

      real*8  uedge(dlohi(1,1)-1:dlohi(1,2)+2,
     $              dlohi(2,1)-1:dlohi(2,2)+1,
     $              dlohi(3,1)-1:dlohi(3,2)+1) 
      real*8  vedge(dlohi(1,1)-1:dlohi(1,2)+1,
     $              dlohi(2,1)-1:dlohi(2,2)+2,
     $              dlohi(3,1)-1:dlohi(3,2)+1) 
      real*8  wedge(dlohi(1,1)-2:dlohi(1,2)+2,
     $              dlohi(2,1)-2:dlohi(2,2)+2,
     $              dlohi(3,1)-2:dlohi(3,2)+3) 
      real*8  betaedge(dlohi(1,1)-2:dlohi(1,2)+2,
     $                 dlohi(2,1)-2:dlohi(2,2)+2,
     $                 dlohi(3,1)-2:dlohi(3,2)+3)

      real*8  phi(dlohi(1,1)-2:dlohi(1,2)+2,
     $            dlohi(2,1)-2:dlohi(2,2)+2,
     $            dlohi(3,1)-2:dlohi(3,2)+2)

      real*8  viscos(2), densty(2), gravty

      real*8  slx(dlohi(1,1)-2:dlohi(1,2)+2,
     $            dlohi(2,1)-2:dlohi(2,2)+2)
      real*8  sly(dlohi(1,1)-2:dlohi(1,2)+2,
     $            dlohi(2,1)-2:dlohi(2,2)+2) 
      real*8  slz(dlohi(1,1)-2:dlohi(1,2)+2,
     $            dlohi(2,1)-2:dlohi(2,2)+2,
     $            3) 

      real*8   sbot(dlohi(1,1)-3:dlohi(1,2)+3,
     $              dlohi(2,1)-3:dlohi(2,2)+3)
      real*8   s_top(dlohi(1,1)-3:dlohi(1,2)+3,
     $              dlohi(2,1)-3:dlohi(2,2)+3)
      real*8  sbotp(dlohi(1,1)-3:dlohi(1,2)+3,
     $              dlohi(2,1)-3:dlohi(2,2)+3)

      real*8  sbck(dlohi(1,1)-3:dlohi(1,2)+3,
     $             dlohi(2,1)-3:dlohi(2,2)+3)
      real*8  sfrt(dlohi(1,1)-3:dlohi(1,2)+3,
     $             dlohi(2,1)-3:dlohi(2,2)+3)

      real*8   sleft(dlohi(1,1)-3:dlohi(1,2)+3)
      real*8  sright(dlohi(1,1)-3:dlohi(1,2)+3)

      real*8  sfluxz(dlohi(1,1)-3:dlohi(1,2)+3)

      real*8  dxscr(dlohi(1,1)-3:dlohi(1,2)+3, 4)
      real*8  dyscr(dlohi(2,1)-3:dlohi(2,2)+3, 4)
      real*8  dzscr(dlohi(1,1)-3:dlohi(1,2)+3, 3)
      real*8  dzfrm(dlohi(1,1)-3:dlohi(1,2)+3, 3)

      logical firstord
      integer is, ie, js, je, ks, ke
      integer i, j, k, km, kc, kp, kt
      real*8  mu0, mu1, den0, den1, g
      real*8  mu0i, mu1i, den0i, den1i
      real*8  dx, dy, dz, dth, dxh, dyh, dzh, beta
      real*8  dxi, dyi, dzi, dxhi, dyhi, dzhi
      real*8  tlo_xlo, tlo_xhi, tlo_ylo, tlo_yhi, tlo_zlo, tlo_zhi
      real*8  thi_xlo, thi_xhi, thi_ylo, thi_yhi, thi_zlo, thi_zhi
      real*8  tlo_x, thi_x, tlo_y, thi_y, tlo_z, thi_z
      real*8  sux, suy, suz, cux, cuy, cuz
      real*8  wc, supw, supw_m, supw_p
      real*8  half

      real*8  mobl0,mobl1,dmobl0,dmobl1,ddmobl0,ddmobl1
      real*8  f,fp,h,hp,Q,Qp
      real*8  sat,m0,m1,v,b
      real*8  mobl0i,mobl1i,dmobl0i,dmobl1i,ddmobl0i,ddmobl1i
      real*8  fi,fpi,hi,hpi,Qi,Qpi
      real*8  m0i,m1i
      real*8  phiinv
      
      real*8  temp1
      dimension temp1(1)
      real*8  temp2
      dimension temp2(1)
      real*8  temp3
      dimension temp3(1)
      real*8  temp4
      dimension temp4(1)
      real*8  temp5
      dimension temp5(1)
      real*8  temp6
      dimension temp6(1)
      real*8  temp7
      dimension temp7(1)

      data    half/0.5d0/
      data firstord /.false./

c
c     All the statement functions are here.
c

      mobl0(sat,m0)     = sat**2/m0
      mobl0i(sat,m0i)   = sat*sat*m0i
      mobl1(sat,m1)     = (1.0-sat)**2/m1
      mobl1i(sat,m1i)   = (1.0-sat)*(1.0-sat)*m1i
      dmobl0(sat,m0)    =  2.0*sat/m0
      dmobl0i(sat,m0i)  =  2.0*sat*m0i
      dmobl1(sat,m1)    = -2.0*(1.0-sat)/m1
      dmobl1i(sat,m1i)  = -2.0*(1.0-sat)*m1i
      ddmobl0(sat,m0)   = 2.0/m0
      ddmobl0i(sat,m0i) = 2.0*m0i
      ddmobl1(sat,m1)   = 2.0/m1
      ddmobl1i(sat,m1i) = 2.0*m1i

      f(sat,m0,m1)  = mobl0(sat,m0)/(mobl0(sat,m0)+mobl1(sat,m1))
      fi(sat,m0i,m1i)   =
     $   mobl0i(sat,m0i)/(mobl0i(sat,m0i)+mobl1i(sat,m1i))
      fp(sat,m0,m1) = dmobl0(sat,m0)/(mobl0(sat,m0)+mobl1(sat,m1))
     $              - mobl0(sat,m0)*(dmobl0(sat,m0)+dmobl1(sat,m1))
     $                / (mobl0(sat,m0)+mobl1(sat,m1))**2
      fpi(sat,m0i,m1i)  =
     $   (dmobl0i(sat,m0i)*mobl1i(sat,m1i) 
     $   - mobl0i(sat,m0i)*dmobl1i(sat,m1i))
     $   / (mobl0i(sat,m0i)+mobl1i(sat,m1i))**2

      h(sat,m0,m1) = mobl0(sat,m0)*mobl1(sat,m1)
     $               / (mobl0(sat,m0)+mobl1(sat,m1))
      hi(sat,m0i,m1i) = mobl0i(sat,m0i)*mobl1i(sat,m1i)
     $               / (mobl0i(sat,m0i)+mobl1i(sat,m1i))
      hp(sat,m0,m1) = (dmobl0(sat,m0)*mobl1(sat,m1)
     $                 + mobl0(sat,m0)*dmobl1(sat,m1))
     $                / (mobl0(sat,m0)+mobl1(sat,m1))
     $              - (mobl0(sat,m0)*mobl1(sat,m1)
     $                 *(dmobl0(sat,m0)+dmobl1(sat,m1)))
     $                / (mobl0(sat,m0)+mobl1(sat,m1))**2
      hpi(sat,m0i,m1i) =
     $   (dmobl0i(sat,m0i)*mobl1i(sat,m1i)*mobl1i(sat,m1i)
     $   +dmobl1i(sat,m1i)*mobl0i(sat,m0i)*mobl0i(sat,m0i))
     $   / (mobl0i(sat,m0i)+mobl1i(sat,m1i))**2

      Q(sat,m0,m1,v,b)  = f(sat,m0,m1)*v+h(sat,m0,m1)*b
      Qi(sat,m0i,m1i,v,b)  = fi(sat,m0i,m1i)*v+hi(sat,m0i,m1i)*b
      Qp(sat,m0,m1,v,b) = fp(sat,m0,m1)*v+hp(sat,m0,m1)*b
      Qpi(sat,m0i,m1i,v,b) = fpi(sat,m0i,m1i)*v+hpi(sat,m0i,m1i)*b

c
c      Code starts here
c

      is = lohi(1,1)
      ie = lohi(1,2)
      js = lohi(2,1)
      je = lohi(2,2)
      ks = lohi(3,1)
      ke = lohi(3,2)
      dx = hx(1)
      dy = hx(2)
      dz = hx(3)
      dxh = half*dx
      dyh = half*dy
      dzh = half*dz
      dxi = 1./dx
      dyi = 1./dy
      dzi = 1./dz
      dxhi = 1./dxh
      dyhi = 1./dyh
      dzhi = 1./dzh
      dth = half*dt

      mu0  = viscos(1)
      mu1  = viscos(2)
      den0 = densty(1)
      den1 = densty(2)
      mu0i  = 1./mu0
      mu1i  = 1./mu1
      den0i = 1./den0
      den1i = 1./den1
      g    = -gravty

      beta = (den0 - den1)*g

      km = 3
      kc = 1
      kp = 2

c----------------------------------------------------------
c     k = ks-1, ke+1 loop
c----------------------------------------------------------

      if (.not. firstord) call sslopez(s,mu0,mu1,wedge,
     $             betaedge,beta,slz,ks-1,kc,
     $             lohi(1,1),lohi(1,2),
     $             dlohi(1,1),dlohi(1,2),
     $             dzscr,dzfrm)

      do k = ks-1,ke+1

         if (.not. firstord) then
           call sslopexy(s,mu0,mu1,slx,sly,
     $                 k,lohi(1,1),lohi(1,2),dlohi(1,1),dlohi(1,2),
     $                 dxscr,dyscr)
           if (k .le. ke) then
              call sslopez(s,mu0,mu1,wedge,betaedge,beta,slz,k+1,kp,
     $                   lohi(1,1),lohi(1,2),
     $                   dlohi(1,1),dlohi(1,2),
     $                   dzscr,dzfrm)
           endif
         endif

         do j=js-1,je+1

            do i=is-1,ie+1

               phiinv = 1./phi(i,j,k)
C              phiinv = 1./phi(i-1,j,k) was a mistake, I believe

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
     $              fpi(s(i-1,j,k),mu0i,mu1i)*uedge(i,j,k)*dth*dxi
     $              /phi(i-1,j,k))*slx(i-1,j) 
                  tlo_xhi = tlo_xhi - (half +
     $              fpi(s(i,j,k),mu0i,mu1i)*uedge(i,j,k)*dth*dxi
     $              *phiinv)*slx(  i,j) 

                  thi_xlo = thi_xlo + (half -
     $              fpi(s(i,j,k),mu0i,mu1i)*uedge(i+1,j,k)*dth*dxi
     $              *phiinv)*slx(  i,j) 
                  thi_xhi = thi_xhi - (half +
     $              fpi(s(i+1,j,k),mu0i,mu1i)*uedge(i+1,j,k)
     $              *dth*dxi/phi(i+1,j,k))
     $              *slx(i+1,j) 

                  tlo_ylo = tlo_ylo + (half -
     $              fpi(s(i,j-1,k),mu0i,mu1i)*vedge(i,j,k)*dth*dyi
     $              /phi(i,j-1,k))
     $              *sly(i,j-1) 
                  tlo_yhi = tlo_yhi - (half +
     $              fpi(s(i,j,k),mu0i,mu1i)*vedge(i,j,k)*dth*dyi
     $              *phiinv)
     $              *sly(i,  j) 

                  thi_ylo = thi_ylo + (half -
     $              fpi(s(i,j,k),mu0i,mu1i)*vedge(i,j+1,k)*dth*dyi
     $              *phiinv)
     $              *sly(i,  j) 
                  thi_yhi = thi_yhi - (half +
     $              fpi(s(i,j+1,k),mu0i,mu1i)*vedge(i,j+1,k)
     $              *dth*dyi/phi(i,j+1,k))
     $              *sly(i,j+1) 

                  tlo_zlo = tlo_zlo + (half -
     $              Qpi(s(i,j,k-1),mu0i,mu1i,wedge(i,j,k),
     $              beta*betaedge(i,j,k-1))*dth*dzi/phi(i,j,k-1))
     $              *slz(i,j,km) 
                  tlo_zhi = tlo_zhi - (half +
     $              Qpi(s(i,j,k),mu0i,mu1i,wedge(i,j,k),
     $              beta*betaedge(i,j,k))*dth*dzi*phiinv)
     $              *slz(i,j,kc) 

                  thi_zlo = thi_zlo + (half -
     $              Qpi(s(i,j,k),mu0i,mu1i,wedge(i,j,k+1),
     $              beta*betaedge(i,j,k))*dth*dzi*phiinv)
     $              *slz(i,j,kc) 
                  thi_zhi = thi_zhi - (half +
     $              Qpi(s(i,j,k+1),mu0i,mu1i,wedge(i,j,k+1),
     $              beta*betaedge(i,j,k+1))*dth*dzi/phi(i,j,k+1))
     $              *slz(i,j,kp) 

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

               temp1(1) = tlo_zlo
               temp2(1) = tlo_zhi
               temp3(1) = betaedge(i,j,k)*beta
               temp4(1) = mu0
               temp5(1) = mu1
               temp6(1) = wc
               temp7(1) = tlo_z
               call rpsolv(temp1,temp2,wedge(i,j,k),
     $                     temp3,temp4,temp5,
     $                     1,temp6,temp7)

               temp1(1) = thi_zlo
               temp2(1) = thi_zhi
               temp3(1) = beta*betaedge(i,j,k+1)
               temp7(1) = thi_z
               call rpsolv(temp1,temp2,wedge(i,j,k+1),
     $                     temp3,temp4,temp5,
     $                     1,temp6,temp7)

               sux = (uedge(i+1,j,k)*fi(thi_x,mu0i,mu1i)
     $             -  uedge(  i,j,k)*fi(tlo_x,mu0i,mu1i))*dxi
               suy = (vedge(i,j+1,k)*fi(thi_y,mu0i,mu1i)
     $             -  vedge(i,  j,k)*fi(tlo_y,mu0i,mu1i))*dyi
               suz =
     $         (Qi(thi_z,mu0i,mu1i,wedge(i,j,k+1),
     $            beta*betaedge(i,j,k+1)) -
     $            Qi(tlo_z,mu0i,mu1i,wedge(i,j,  k),
     $            beta*betaedge(i,j,  k)))
     $            *dzi

               cux = fi(s(i,j,k),mu0i,mu1i)
     $             *(uedge(i+1,j,k) - uedge(i,j,k))*dxi
               cuy = fi(s(i,j,k),mu0i,mu1i)
     $             *(vedge(i,j+1,k) - vedge(i,j,k))*dyi
               cuz = fi(s(i,j,k),mu0i,mu1i)
     $             *(wedge(i,j,k+1) - wedge(i,j,k))*dzi

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
     $                    dt*(fi(supw_p,mu0i,mu1i)*uedge(i+1,j,k) -
     $                        fi(supw_m,mu0i,mu1i)*uedge(  i,j,k))/
     $                    (dx*phi(i,j,k)) 

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
     $                 dt*(fi(supw_p,mu0i,mu1i)*vedge(i,j+1,k) -
     $                     fi(supw_m,mu0i,mu1i)*vedge(i,  j,k))/
     $                 (dy*phi(i,j,k)) 

               enddo

            enddo

         endif

c     ::: add z contributions to sn

         if ((k .ge. ks) .and. (k .le. (ke+1))) then

            do j=js,je

               do i=is,ie
                  
                  temp1(1) = beta*betaedge(i,j,k)
                  temp2(1) = mu0
                  temp3(1) = mu1
                  temp4(1) = wc
                  temp5(1) = supw
                  call rpsolv(sbot(i,j),s_top(i,j),wedge(i,j,k),
     $                        temp1,temp2,temp3,
     $                        1,temp4,temp5)
                  sfluxz(i) =
     $            Qi(supw,mu0i,mu1i,wedge(i,j,k),beta*betaedge(i,j,k))

               enddo

               if (k .eq. ks) then

                  do i=is,ie

                     sn(i,j,k)   = sn(i,j,k  ) +
     $                             dt*sfluxz(i)/(dz*phi(i,j,k))

                  enddo

               else if (k .eq. (ke+1)) then

                  do i=is,ie

                     sn(i,j,k-1) = sn(i,j,k-1) -
     $                             dt*sfluxz(i)/(dz*phi(i,j,k-1))

                  enddo

               else 

                  do i=is,ie

                     sn(i,j,k)   = sn(i,j,k  ) +
     $                             dt*sfluxz(i)/(dz*phi(i,j,k))
                     sn(i,j,k-1) = sn(i,j,k-1) -
     $                             dt*sfluxz(i)/(dz*phi(i,j,k-1))

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
c     sslopexy:
c     Compute slopes in x and y
c----------------------------------------------------------------------

      subroutine sslopexy (s,mu0,mu1,
     $                    slx,sly,k,lo,hi,dlo,dhi,dxscr,dyscr)
      implicit none

      integer lo(3), hi(3), dlo(3), dhi(3)
      real*8 s(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
      real*8 mu0,mu1
      real*8 slx(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2)
      real*8 sly(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2)
      real*8 dxscr(dlo(1)-3:dhi(1)+3,4)
      real*8 dyscr(dlo(2)-3:dhi(2)+3,4)

      integer cen,lim,flag,fromm

      parameter( cen   = 1 )
      parameter( lim   = 2 )
      parameter( flag  = 3 )
      parameter( fromm = 4 )

      real*8  dpls,dmin,ds

      logical firstord
      integer is,js,ie,je,i,j,k
      real*8  strm,strc,strp
      real*8  zero,sixth,half,two3rd,one,two

      real*8  mobl0,mobl1,dmobl0,dmobl1,ddmobl0,ddmobl1
      real*8  mobl0i,mobl1i,dmobl0i,dmobl1i,ddmobl0i,ddmobl1i
      real*8  fpp,fppi
      real*8  sat,m0,m1,m0i,m1i
      real*8  mu0i,mu1i

      data firstord /.false./
      data zero,sixth,half,two3rd,one,two/
     $     0.d0,
     $     0.166666666666667d0,
     $     0.5d0,
     $     0.666666666666667d0,
     $     1.d0,
     $     2.d0/ 

      mobl0(sat,m0)     = sat**2/m0
      mobl0i(sat,m0i)   = sat*sat*m0i
      mobl1(sat,m1)     = (1.0-sat)**2/m1
      mobl1i(sat,m1i)   = (1.0-sat)*(1.0-sat)*m1i
      dmobl0(sat,m0)    =  2.0*sat/m0
      dmobl0i(sat,m0i)  =  2.0*sat*m0i
      dmobl1(sat,m1)    = -2.0*(1.0-sat)/m1
      dmobl1i(sat,m1i)  = -2.0*(1.0-sat)*m1i
      ddmobl0(sat,m0)   = 2.0/m0
      ddmobl0i(sat,m0i) = 2.0*m0i
      ddmobl1(sat,m1)   = 2.0/m1
      ddmobl1i(sat,m1i) = 2.0*m1i

      fpp(sat,m0,m1)
     $             = ddmobl0(sat,m0)/(mobl0(sat,m0)+mobl1(sat,m1))
     $             - dmobl0(sat,m0)*(dmobl0(sat,m0)+dmobl1(sat,m1))
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $             - (dmobl0(sat,m0)*(dmobl0(sat,m0)+dmobl1(sat,m1))
     $                + mobl0(sat,m0)*(ddmobl0(sat,m0)+ddmobl1(sat,m1)))
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $             + 2.0*mobl0(sat,m0)*(mobl0(sat,m0)+mobl1(sat,m1))
     $               *(dmobl0(sat,m0)+dmobl1(sat,m1))**2
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**4
      fppi(sat,m0i,m1i)
     $   = ( ( ddmobl0i(sat,m0i)*mobl1i(sat,m1i) 
     $       - ddmobl1i(sat,m1i)*mobl0i(sat,m0i) )
     $       * ( mobl0i(sat,m0i) + mobl1i(sat,m1i) )
     $       - 2.0*( dmobl0i(sat,m0i)*mobl1i(sat,m1i)
     $             - dmobl1i(sat,m1i)*mobl0i(sat,m0i) )
     $       * ( dmobl0i(sat,m0i) + dmobl1i(sat,m1i) ) )
     $     / (mobl0i(sat,m0i)+mobl1i(sat,m1i))**3

      is = lo(1)
      js = lo(2)
      ie = hi(1)
      je = hi(2)
      mu0i = 1./mu0
      mu1i = 1./mu1

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
            if (dpls*dmin .lt. 0.0) then
               dxscr(i,lim) = zero
            else
               dxscr(i,lim)= min(abs(dmin),abs(dpls))
               strm = fppi(s(i-1,j,k),mu0i,mu1i)
               strc = fppi(s(  i,j,k),mu0i,mu1i)
               strp = fppi(s(i+1,j,k),mu0i,mu1i)
               if ( ((strm * strc) .le. 0.0) .or.
     $              ((strc * strp) .le. 0.0) ) then
                  dxscr(i,lim) = half*dxscr(i,lim)
               endif
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
            if (dpls*dmin .lt. 0.0) then
               dyscr(j,lim) = zero
            else
               dyscr(j,lim)= min(abs(dmin),abs(dpls))
               strm = fppi(s(i,j-1,k),mu0i,mu1i)
               strc = fppi(s(i,  j,k),mu0i,mu1i)
               strp = fppi(s(i,j+1,k),mu0i,mu1i)
               if ( ((strm * strc) .le. 0.0) .or.
     $              ((strc * strp) .le. 0.0) ) then
                  dyscr(j,lim) = half*dyscr(j,lim)
               endif
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
c     sslopez:
c     Compute slopes in z
c----------------------------------------------------------------------
      subroutine sslopez (s,mu0,mu1,w,betaedge,beta,
     $                   slz,k,kk,lo,hi,dlo,dhi,dzscr,dzfrm)
      implicit none

      integer is,js,ie,je,i,j,k,kk,kt
      integer lo(3), hi(3), dlo(3), dhi(3)
      real*8 s(dlo(1)-3:dhi(1)+3,dlo(2)-3:dhi(2)+3,dlo(3)-3:dhi(3)+3)
      real*8 mu0,mu1
      real*8 w(dlo(1)-2:dhi(1)+2,dlo(2)-2:dhi(2)+2,dlo(3)-2:dhi(3)+3)
      real*8 betaedge(dlo(1)-2:dhi(1)+2,
     $                dlo(2)-2:dhi(2)+2,
     $                dlo(3)-2:dhi(3)+3)
      real*8 beta
      real*8 slz(dlo(1)-2:dhi(1)+2, dlo(2)-2:dhi(2)+2,3)
      real*8 dzscr(dlo(1)-3:dhi(1)+3,3)
      real*8 dzfrm(dlo(1)-3:dhi(1)+3,k-1:k+1)

      integer cen,lim,flag
      parameter( cen = 1 )
      parameter( lim = 2 )
      parameter( flag = 3 )

      real*8 dpls,dmin,ds


      logical firstord
      real*8  strm,strcm,strcp,strp
      real*8  zero,sixth,half,two3rd,one,two

      real*8  mobl0,mobl1,dmobl0,dmobl1,ddmobl0,ddmobl1
      real*8  mobl0i,mobl1i,dmobl0i,dmobl1i,ddmobl0i,ddmobl1i
      real*8  fpp,hpp,Qpp
      real*8  fppi,hppi,Qppi
      real*8  sat,m0,m1,v,b
      real*8  m0i,m1i,mu0i,mu1i

      data firstord /.false./
      data zero,sixth,half,two3rd,one,two/
     $     0.d0,
     $     0.166666666666667d0,
     $     0.5d0,
     $     0.666666666666667d0,
     $     1.d0,
     $     2.d0/ 

      mobl0(sat,m0)     = sat**2/m0
      mobl0i(sat,m0i)   = sat*sat*m0i
      mobl1(sat,m1)     = (1.0-sat)**2/m1
      mobl1i(sat,m1i)   = (1.0-sat)*(1.0-sat)*m1i
      dmobl0(sat,m0)    =  2.0*sat/m0
      dmobl0i(sat,m0i)  =  2.0*sat*m0i
      dmobl1(sat,m1)    = -2.0*(1.0-sat)/m1
      dmobl1i(sat,m1i)  = -2.0*(1.0-sat)*m1i
      ddmobl0(sat,m0)   = 2.0/m0
      ddmobl0i(sat,m0i) = 2.0*m0i
      ddmobl1(sat,m1)   = 2.0/m1
      ddmobl1i(sat,m1i) = 2.0*m1i

      fpp(sat,m0,m1)
     $             = ddmobl0(sat,m0)/(mobl0(sat,m0)+mobl1(sat,m1))
     $             - dmobl0(sat,m0)*(dmobl0(sat,m0)+dmobl1(sat,m1))
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $             - (dmobl0(sat,m0)*(dmobl0(sat,m0)+dmobl1(sat,m1))
     $                + mobl0(sat,m0)*(ddmobl0(sat,m0)+ddmobl1(sat,m1)))
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $             + 2.0*mobl0(sat,m0)*(mobl0(sat,m0)+mobl1(sat,m1))
     $               *(dmobl0(sat,m0)+dmobl1(sat,m1))**2
     $               / (mobl0(sat,m0)+mobl1(sat,m1))**4
      fppi(sat,m0i,m1i)
     $   = ( ( ddmobl0i(sat,m0i)*mobl1i(sat,m1i) 
     $       - ddmobl1i(sat,m1i)*mobl0i(sat,m0i) )
     $       * ( mobl0i(sat,m0i) + mobl1i(sat,m1i) )
     $       - 2.0*( dmobl0i(sat,m0i)*mobl1i(sat,m1i)
     $             - dmobl1i(sat,m1i)*mobl0i(sat,m0i) )
     $       * ( dmobl0i(sat,m0i) + dmobl1i(sat,m1i) ) )
     $     / (mobl0i(sat,m0i)+mobl1i(sat,m1i))**3

      hpp(sat,m0,m1) = (ddmobl0(sat,m0)*mobl1(sat,m1)
     $                  + 2.0*dmobl0(sat,m0)*dmobl1(sat,m1)
     $                  + mobl0(sat,m0)*ddmobl1(sat,m1))
     $                 / (mobl0(sat,m0)+mobl1(sat,m1))
     $               - ((dmobl0(sat,m0)*mobl1(sat,m1) 
     $                   + mobl0(sat,m0)*dmobl1(sat,m1))
     $                  *(dmobl0(sat,m0)+dmobl1(sat,m1)))
     $                 / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $               - (dmobl0(sat,m0)*mobl1(sat,m1)
     $                  *(dmobl0(sat,m0)+dmobl1(sat,m1))
     $                 + mobl0(sat,m0)*dmobl1(sat,m1)
     $                  *(dmobl0(sat,m0)+dmobl1(sat,m1))
     $                 + mobl0(sat,m0)*mobl1(sat,m1)
     $                  *(ddmobl0(sat,m0)+ddmobl1(sat,m1)))
     $                 / (mobl0(sat,m0)+mobl1(sat,m1))**2
     $               + (2.0*mobl0(sat,m0)*mobl1(sat,m1)
     $                  *(mobl0(sat,m0)+mobl1(sat,m1))
     $                  *(dmobl0(sat,m0)+dmobl1(sat,m1))**2)
     $                 / (mobl0(sat,m0)+mobl1(sat,m1))**4
      hppi(sat,m0i,m1i) 
     $   = ( ( ddmobl0i(sat,m0i)*mobl1i(sat,m1i)*mobl1i(sat,m1i)
     $       + ddmobl1i(sat,m1i)*mobl0i(sat,m0i)*mobl0i(sat,m0i) )
     $       * ( mobl0(sat,m0i)+mobl1(sat,m1i) )
     $       - 2.0*( dmobl0i(sat,m0i)*mobl1i(sat,m1i)
     $             - dmobl1i(sat,m1i)*mobl0i(sat,m0i) )**2 )
     $     / (mobl0i(sat,m0i)+mobl1i(sat,m1i))**3

      Qpp(sat,m0,m1,v,b) =
     $         fpp(sat,m0,m1)*v+hpp(sat,m0,m1)*b
      Qppi(sat,m0i,m1i,v,b) =
     $         fppi(sat,m0i,m1i)*v+hppi(sat,m0i,m1i)*b

      is = lo(1)
      js = lo(2)
      ie = hi(1)
      je = hi(2)
      mu0i = 1./mu0
      mu1i = 1./mu1

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
            if ( (dpls * dmin) .lt. 0.0) then
               dzscr(i,lim) = zero
            else
               dzscr(i,lim)= min(abs(dmin),abs(dpls))
               strm =
     $          Qppi(s(i,j,kt-1),mu0i,mu1i,w(i,j,  kt),
     $              beta*betaedge(i,j,kt))
               strcm =
     $          Qppi(s(i,j,  kt),mu0i,mu1i,w(i,j,  kt),
     $              beta*betaedge(i,j,  kt))
               strcp =
     $          Qppi(s(i,j,  kt),mu0i,mu1i,w(i,j,kt+1),
     $              beta*betaedge(i,j,kt+1))
               strp =
     $          Qppi(s(i,j,kt+1),mu0i,mu1i,w(i,j,kt+1),
     $              beta*betaedge(i,j,kt+1))
               if ( ((strm * strcm) .le. 0.0) .or.
     $              ((strcp * strp) .le. 0.0) ) then
                  dzscr(i,lim) = half * dzscr(i,lim)
               endif
            endif
            dzscr(i,flag) = dsign(one,dzscr(i,cen))
            dzfrm(i,kt)= dzscr(i,flag)*
     $           min(dzscr(i,lim),abs(dzscr(i,cen)))

            kt = k+1
            dzscr(i,cen) = half*(s(i,j,kt+1)-s(i,j,kt-1))
            dmin = two*(s(i,j,kt)-s(i,j,kt-1))
            dpls = two*(s(i,j,kt+1)-s(i ,j,kt))
            if ( (dpls * dmin) .lt. 0.0) then
               dzscr(i,lim) = zero
            else
               dzscr(i,lim)= min(abs(dmin),abs(dpls))
               strm =
     $          Qppi(s(i,j,kt-1),mu0i,mu1i,w(i,j,  kt),
     $              beta*betaedge(i,j,kt-1))
               strcm =
     $          Qppi(s(i,j,  kt),mu0i,mu1i,w(i,j,  kt),
     $              beta*betaedge(i,j,  kt))
               strcp =
     $          Qppi(s(i,j,  kt),mu0i,mu1i,w(i,j,kt+1),
     $              beta*betaedge(i,j,kt+1))
               strp =
     $          Qppi(s(i,j,kt+1),mu0i,mu1i,w(i,j,kt+1),
     $              beta*betaedge(i,j,kt+1))
               if ( ((strm * strcm) .le. 0.0) .or.
     $              ((strcp * strp) .le. 0.0) ) then
                  dzscr(i,lim) = half * dzscr(i,lim)
               endif
            endif
            dzscr(i,flag) = dsign(one,dzscr(i,cen))
            dzfrm(i,kt)= dzscr(i,flag)*
     $           min(dzscr(i,lim),abs(dzscr(i,cen)))

            dzscr(i,cen) = half*(s(i,j,k+1)-s(i,j,k-1))
            dmin = two*(s(i,j,k)-s(i,j,k-1))
            dpls = two*(s(i,j,k+1)-s(i ,j,k))
            if ( (dpls * dmin) .lt. 0.0) then
               dzscr(i,lim) = zero
            else
               dzscr(i,lim)= min(abs(dmin),abs(dpls))
               strm =
     $          Qppi(s(i,j,k-1),mu0i,mu1i,w(i,j,k-1),
     $              beta*betaedge(i,j,k-1))
               strcm =
     $          Qppi(s(i,j,  k),mu0i,mu1i,w(i,j,  k),
     $              beta*betaedge(i,j,  k))
               strcp =
     $          Qppi(s(i,j,  k),mu0i,mu1i,w(i,j,k+1),
     $              beta*betaedge(i,j,k+1))
               strp =
     $          Qppi(s(i,j,k+1),mu0i,mu1i,w(i,j,k+1),
     $              beta*betaedge(i,j,k+1))
               if ( ((strm * strcm) .le. 0.0) .or.
     $              ((strcp * strp) .le. 0.0) ) then
                  dzscr(i,lim) = half * dzscr(i,lim)
               endif
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

      subroutine rpsolv(wl,wr,alpha,beta,visc0,visc1,jmax,wc,wrp)
      implicit real*8(a-h,o-z)
      integer jmax
      integer j
      integer k
      dimension visc0(jmax),visc1(jmax),alpha(jmax),beta(jmax)
      dimension wl(jmax),wr(jmax),wc(jmax),wrp(jmax)

c     we are solving for phase 0 here where indexes will likely be
c       0 - water
c       1 - air
c
c     this routine takes a vector of data of length jmax consisting of:
c     
c 	wl		left states
c	wr		right states
c	alpha		total velocity
c	beta		gravity
c	visc0		viscosity of phase 0
c	visc1		viscosity of phase 1
c	wc		scratch array used by Riemann solver
c	
c     the routine returns wrp which is the solution of the Riemann problem
      permb0(x) = x**2
      permb1(x) = (1.-x)**2
      dperm0(x) = 2.*x
      dperm1(x) = -2.*(1.-x)
      ddper0(x) = 2.
      ddper1(x) = 2.

      amob0(x,u0) = permb0(x)/u0
      amob1(x,u1) = permb1(x)/u1
      dmob0(x,u0) = dperm0(x)/u0
      dmb0i(x,u0) = u0/dperm0(x)
      dmob1(x,u1) = dperm1(x)/u1
      dmb1i(x,u1) = u1/dperm1(x)
      ddmb0(x,u0) = ddper0(x)/u0
      ddmb1(x,u1) = ddper1(x)/u1

      flux(a,b,x,u0,u1) = amob0(x,u0)*(a+b*amob1(x,u1))/(amob0(x,u0)+
     $  amob1(x,u1)) 

c
c     initial guess for newtons method.
c
      x0 = 1.
      x1 = 0.
      do 20 j = 1, jmax
      delta0 =
     $      -(amob0(x0,visc0(j))*alpha(j)-amob0(x0,visc0(j))**2*beta(j))
     $       *dmb0i(x0,visc0(j)) 
      delta1 =
     $       (amob1(x1,visc1(j))*alpha(j)+amob1(x1,visc1(j))**2*beta(j))
     $       *dmb1i(x1,visc1(j)) 
      if (delta0.eq.delta1) then
        deldif = 1.
      else
        deldif = delta1-delta0
      endif
      if (delta0*delta1.ge.0.) then
        wc(j) = 2.
      else
        wc(j) = delta1/deldif
      endif
  20  continue
c
c     linear interpolation--newtons method.
c
      do 31 k = 1, 5
      do 30 j = 1, jmax
c     delta =
c    $        alpha(j)*(amob1(wc(j),visc1(j))/dmob1(wc(j),visc1(j))
c    $                 -amob0(wc(j),visc0(j))/dmob0(wc(j),visc0(j)))
c    $      + beta(j)*(amob1(wc(j),visc1(j))**2/dmob1(wc(j),visc1(j))
c    $                +amob0(wc(j),visc0(j))**2/dmob0(wc(j),visc0(j))) 
      delta =
     $      ( alpha(j) + beta(j)*amob1(wc(j),visc1(j)) )
     $      * amob1(wc(j),visc1(j))*dmb1i(wc(j),visc1(j))
     $    + ( alpha(j) - beta(j)*amob0(wc(j),visc0(j)) )
     $      * amob0(wc(j),visc0(j))*dmb0i(wc(j),visc0(j))
      ddelta =
     $         amob0(wc(j),visc0(j))*ddmb0(wc(j),visc0(j))
     $         *(alpha(j)-beta(j)*amob0(wc(j),visc0(j)))
     $         *dmb0i(wc(j),visc0(j))**2
     $       - amob1(wc(j),visc1(j))*ddmb1(wc(j),visc1(j))
     $         *(alpha(j)+beta(j)*amob1(wc(j),visc1(j)))
     $         *dmb1i(wc(j),visc1(j))**2
     $       + 2.*beta(j)*(amob0(wc(j),visc0(j))+amob1(wc(j),visc1(j))) 
      if ( .not. (wc(j).le.1..and.wc(j).gt.0.)) then
        ddelta = 1.
      endif
      if (wc(j).le.1.) then
        wc(j) = wc(j)-delta/ddelta
      else
        wc(j) = 2.
      endif
      if (wc(j).le.0..or.wc(j).ge.1.) then
        wc(j) = 2.
      endif
  30  enddo
  31  enddo
c    
c     solve riemann problem.
c
      do 40 j = 1, jmax
      if (wl(j).lt.wr(j)) then
        flag = 1.
      else
        flag = -1.
      endif
      fleft  = flux(alpha(j),beta(j),wl(j),visc0(j),visc1(j))*flag
      frght  = flux(alpha(j),beta(j),wr(j),visc0(j),visc1(j))*flag
      fcrit  = flux(alpha(j),beta(j),wc(j),visc0(j),visc1(j))*flag
      flmin  = min(fleft,frght)
      wbtw   = (wl(j)-wc(j))*(wr(j)-wc(j))
      if (fleft.lt.frght) then
        wrp(j) = wl(j)
      else
        wrp(j) = wr(j)
      endif
      if (flmin.lt.fcrit) then
        wrtmp = wrp(j)
      else
        wrtmp = wc(j)
      endif
      if ( .not. (wbtw.ge.0.)) then
        wrp(j) = wrtmp
      endif
  40  continue
      return
      end
