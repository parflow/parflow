/* advect.f -- translated by f2c (version 19960315).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

#include "f2c.h"

/*BHEADER**********************************************************************
 *
 *  Copyright (c) 1995-2024, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER********************************************************************
 ****/
/****************************************************************************
 */
/* * */
/* * Machine specific porting hacks */
/* * */
/****************************************************************************
 */
/* ---------------------------------------------------------------------- */
/*     advect: */
/*     Godunov advection routine */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int advect_(s, sn, uedge, vedge, wedge, phi, slx, sly, slz,
                             lo, hi, dlo, dhi, hx, dt, fstord, sbot, stop, sbotp, sfrt, sbck,
                             sleft, sright, sfluxz, dxscr, dyscr, dzscr, dzfrm)
doublereal * s, *sn, *uedge, *vedge, *wedge, *phi, *slx, *sly, *slz;
integer *lo, *hi, *dlo, *dhi;
doublereal *hx, *dt;
integer *fstord;
doublereal *sbot, *stop, *sbotp, *sfrt, *sbck, *sleft, *sright, *sfluxz, *
           dxscr, *dyscr, *dzscr, *dzfrm;
{
  /* Initialized data */

  doublereal half = .5;

  /* System generated locals */
  integer s_dim1, s_dim2, s_offset, sn_dim1, sn_dim2, sn_offset, uedge_dim1,
          uedge_dim2, uedge_offset, vedge_dim1, vedge_dim2, vedge_offset,
          wedge_dim1, wedge_dim2, wedge_offset, phi_dim1, phi_dim2,
          phi_offset, slx_dim1, slx_offset, sly_dim1, sly_offset, slz_dim1,
          slz_dim2, slz_offset, sbot_dim1, sbot_offset, stop_dim1,
          stop_offset, sbotp_dim1, sbotp_offset, sbck_dim1, sbck_offset,
          sfrt_dim1, sfrt_offset, sleft_offset, sright_offset,
          sfluxz_offset, dxscr_dim1, dxscr_offset, dyscr_dim1, dyscr_offset,
          dzscr_dim1, dzscr_offset, dzfrm_dim1, dzfrm_offset, i__1, i__2,
          i__3;

  /* Local variables */
  doublereal supw;
  logical firstord;
  integer i__, j, k;
  doublereal thi_x__, thi_y__, thi_z__, tlo_x__, tlo_y__, tlo_z__;
  integer ie, je, ke, kc, km, is, js, ks, kp, kt;
  doublereal dx, dy, dz, phiinv, supw_m__;
  extern /* Subroutine */ int slopez_();
  doublereal supw_p__, dth, dxh, dyh, dzh, dxi, dyi, dzi, cux, cuy,
             cuz, thi_xhi__, thi_yhi__, thi_zhi__, sux, suy, suz, tlo_xhi__,
             tlo_yhi__, tlo_zhi__, thi_xlo__, thi_ylo__, thi_zlo__, tlo_xlo__,
             tlo_ylo__, tlo_zlo__;
  extern /* Subroutine */ int slopexy_();

/*     ::: argument declarations */
/* Parameter adjustments */
  --lo;
  --hi;
  --dlo;
  --dhi;
  dzfrm_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dzfrm_offset = dlo[1] - 3 + dzfrm_dim1;
  dzfrm -= dzfrm_offset;
  dzscr_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dzscr_offset = dlo[1] - 3 + dzscr_dim1;
  dzscr -= dzscr_offset;
  dyscr_dim1 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  dyscr_offset = dlo[2] - 3 + dyscr_dim1;
  dyscr -= dyscr_offset;
  dxscr_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dxscr_offset = dlo[1] - 3 + dxscr_dim1;
  dxscr -= dxscr_offset;
  sfluxz_offset = dlo[1] - 3;
  sfluxz -= sfluxz_offset;
  sright_offset = dlo[1] - 3;
  sright -= sright_offset;
  sleft_offset = dlo[1] - 3;
  sleft -= sleft_offset;
  sbck_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  sbck_offset = dlo[1] - 3 + sbck_dim1 * (dlo[2] - 3);
  sbck -= sbck_offset;
  sfrt_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  sfrt_offset = dlo[1] - 3 + sfrt_dim1 * (dlo[2] - 3);
  sfrt -= sfrt_offset;
  sbotp_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  sbotp_offset = dlo[1] - 3 + sbotp_dim1 * (dlo[2] - 3);
  sbotp -= sbotp_offset;
  stop_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  stop_offset = dlo[1] - 3 + stop_dim1 * (dlo[2] - 3);
  stop -= stop_offset;
  sbot_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  sbot_offset = dlo[1] - 3 + sbot_dim1 * (dlo[2] - 3);
  sbot -= sbot_offset;
  slz_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  slz_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  slz_offset = dlo[1] - 2 + slz_dim1 * (dlo[2] - 2 + slz_dim2);
  slz -= slz_offset;
  sly_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  sly_offset = dlo[1] - 2 + sly_dim1 * (dlo[2] - 2);
  sly -= sly_offset;
  slx_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  slx_offset = dlo[1] - 2 + slx_dim1 * (dlo[2] - 2);
  slx -= slx_offset;
  phi_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  phi_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  phi_offset = dlo[1] - 2 + phi_dim1 * (dlo[2] - 2 + phi_dim2 * (dlo[3] - 2)
                                        );
  phi -= phi_offset;
  wedge_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  wedge_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  wedge_offset = dlo[1] - 2 + wedge_dim1 * (dlo[2] - 2 + wedge_dim2 * (dlo[
                                                                         3] - 2));
  wedge -= wedge_offset;
  vedge_dim1 = dhi[1] + 1 - (dlo[1] - 1) + 1;
  vedge_dim2 = dhi[2] + 2 - (dlo[2] - 1) + 1;
  vedge_offset = dlo[1] - 1 + vedge_dim1 * (dlo[2] - 1 + vedge_dim2 * (dlo[
                                                                         3] - 1));
  vedge -= vedge_offset;
  uedge_dim1 = dhi[1] + 2 - (dlo[1] - 1) + 1;
  uedge_dim2 = dhi[2] + 1 - (dlo[2] - 1) + 1;
  uedge_offset = dlo[1] - 1 + uedge_dim1 * (dlo[2] - 1 + uedge_dim2 * (dlo[
                                                                         3] - 1));
  uedge -= uedge_offset;
  sn_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  sn_dim2 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  sn_offset = dlo[1] - 3 + sn_dim1 * (dlo[2] - 3 + sn_dim2 * (dlo[3] - 3));
  sn -= sn_offset;
  s_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  s_dim2 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  s_offset = dlo[1] - 3 + s_dim1 * (dlo[2] - 3 + s_dim2 * (dlo[3] - 3));
  s -= s_offset;
  --hx;

  /* Function Body */
  is = lo[1];
  ie = hi[1];
  js = lo[2];
  je = hi[2];
  ks = lo[3];
  ke = hi[3];
  dx = hx[1];
  dy = hx[2];
  dz = hx[3];
  dxh = half * dx;
  dyh = half * dy;
  dzh = half * dz;
  dth = half * *dt;
  dxi = (float)1. / dx;
  dyi = (float)1. / dy;
  dzi = (float)1. / dz;
  if (*fstord == 1)
  {
    firstord = TRUE_;
  }
  else
  {
    firstord = FALSE_;
  }
  km = 3;
  kc = 1;
  kp = 2;
/* ---------------------------------------------------------- */
/*     k = ks-1, ke+1 loop */
/* ---------------------------------------------------------- */
  if (!firstord)
  {
    i__1 = ks - 1;
    slopez_(&s[s_offset], &slz[slz_offset], &i__1, &kc, &lo[1], &hi[1], &
            dlo[1], &dhi[1], &dzscr[dzscr_offset], &dzfrm[dzfrm_offset]);
  }
  i__1 = ke + 1;
  for (k = ks - 1; k <= i__1; ++k)
  {
    if (!firstord)
    {
      slopexy_(&s[s_offset], &slx[slx_offset], &sly[sly_offset], &k, &
               lo[1], &hi[1], &dlo[1], &dhi[1], &dxscr[dxscr_offset], &
               dyscr[dyscr_offset]);
      if (k <= ke)
      {
        i__2 = k + 1;
        slopez_(&s[s_offset], &slz[slz_offset], &i__2, &kp, &lo[1], &
                hi[1], &dlo[1], &dhi[1], &dzscr[dzscr_offset], &dzfrm[
                  dzfrm_offset]);
      }
    }
    i__2 = je + 1;
    for (j = js - 1; j <= i__2; ++j)
    {
      i__3 = ie + 1;
      for (i__ = is - 1; i__ <= i__3; ++i__)
      {
        phiinv = (float)1. / phi[i__ + (j + k * phi_dim2) * phi_dim1];
        tlo_xlo__ = s[i__ - 1 + (j + k * s_dim2) * s_dim1];
        tlo_xhi__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_xlo__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_xhi__ = s[i__ + 1 + (j + k * s_dim2) * s_dim1];
        tlo_ylo__ = s[i__ + (j - 1 + k * s_dim2) * s_dim1];
        tlo_yhi__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_ylo__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_yhi__ = s[i__ + (j + 1 + k * s_dim2) * s_dim1];
        tlo_zlo__ = s[i__ + (j + (k - 1) * s_dim2) * s_dim1];
        tlo_zhi__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_zlo__ = s[i__ + (j + k * s_dim2) * s_dim1];
        thi_zhi__ = s[i__ + (j + (k + 1) * s_dim2) * s_dim1];
        if (!firstord)
        {
          tlo_xlo__ += (half - uedge[i__ + (j + k * uedge_dim2) *
                                     uedge_dim1] * dth * dxi / phi[i__ - 1 + (j + k *
                                                                              phi_dim2) * phi_dim1]) * slx[i__ - 1 + j *
                                                                                                           slx_dim1];
          tlo_xhi__ -= (half + uedge[i__ + (j + k * uedge_dim2) *
                                     uedge_dim1] * dth * dxi * phiinv) * slx[i__ + j *
                                                                             slx_dim1];
          thi_xlo__ += (half - uedge[i__ + 1 + (j + k * uedge_dim2)
                                     * uedge_dim1] * dth * dxi * phiinv) * slx[i__ + j
                                                                               * slx_dim1];
          thi_xhi__ -= (half + uedge[i__ + 1 + (j + k * uedge_dim2)
                                     * uedge_dim1] * dth * dxi / phi[i__ + 1 + (j + k *
                                                                                phi_dim2) * phi_dim1]) * slx[i__ + 1 + j *
                                                                                                             slx_dim1];
          tlo_ylo__ += (half - vedge[i__ + (j + k * vedge_dim2) *
                                     vedge_dim1] * dth * dyi / phi[i__ + (j - 1 + k *
                                                                          phi_dim2) * phi_dim1]) * sly[i__ + (j - 1) *
                                                                                                       sly_dim1];
          tlo_yhi__ -= (half + vedge[i__ + (j + k * vedge_dim2) *
                                     vedge_dim1] * dth * dyi * phiinv) * sly[i__ + j *
                                                                             sly_dim1];
          thi_ylo__ += (half - vedge[i__ + (j + 1 + k * vedge_dim2)
                                     * vedge_dim1] * dth * dyi * phiinv) * sly[i__ + j
                                                                               * sly_dim1];
          thi_yhi__ -= (half + vedge[i__ + (j + 1 + k * vedge_dim2)
                                     * vedge_dim1] * dth * dyi / phi[i__ + (j + 1 + k *
                                                                            phi_dim2) * phi_dim1]) * sly[i__ + (j + 1) *
                                                                                                         sly_dim1];
          tlo_zlo__ += (half - wedge[i__ + (j + k * wedge_dim2) *
                                     wedge_dim1] * dth * dzi / phi[i__ + (j + (k - 1) *
                                                                          phi_dim2) * phi_dim1]) * slz[i__ + (j + km *
                                                                                                              slz_dim2) * slz_dim1];
          tlo_zhi__ -= (half + wedge[i__ + (j + k * wedge_dim2) *
                                     wedge_dim1] * dth * dzi * phiinv) * slz[i__ + (j
                                                                                    + kc * slz_dim2) * slz_dim1];
          thi_zlo__ += (half - wedge[i__ + (j + (k + 1) *
                                            wedge_dim2) * wedge_dim1] * dth * dzi * phiinv) *
                       slz[i__ + (j + kc * slz_dim2) * slz_dim1];
          thi_zhi__ -= (half + wedge[i__ + (j + (k + 1) *
                                            wedge_dim2) * wedge_dim1] * dth * dzi / phi[i__ +
                                                                                        (j + (k + 1) * phi_dim2) * phi_dim1]) * slz[i__ +
                                                                                                                                    (j + kp * slz_dim2) * slz_dim1];
        }
        if (uedge[i__ + (j + k * uedge_dim2) * uedge_dim1] >= (float)
            0.)
        {
          tlo_x__ = tlo_xlo__;
        }
        else
        {
          tlo_x__ = tlo_xhi__;
        }
        if (uedge[i__ + 1 + (j + k * uedge_dim2) * uedge_dim1] >= (
                                                                   float)0.)
        {
          thi_x__ = thi_xlo__;
        }
        else
        {
          thi_x__ = thi_xhi__;
        }
        if (vedge[i__ + (j + k * vedge_dim2) * vedge_dim1] >= (float)
            0.)
        {
          tlo_y__ = tlo_ylo__;
        }
        else
        {
          tlo_y__ = tlo_yhi__;
        }
        if (vedge[i__ + (j + 1 + k * vedge_dim2) * vedge_dim1] >= (
                                                                   float)0.)
        {
          thi_y__ = thi_ylo__;
        }
        else
        {
          thi_y__ = thi_yhi__;
        }
        if (wedge[i__ + (j + k * wedge_dim2) * wedge_dim1] >= (float)
            0.)
        {
          tlo_z__ = tlo_zlo__;
        }
        else
        {
          tlo_z__ = tlo_zhi__;
        }
        if (wedge[i__ + (j + (k + 1) * wedge_dim2) * wedge_dim1] >= (
                                                                     float)0.)
        {
          thi_z__ = thi_zlo__;
        }
        else
        {
          thi_z__ = thi_zhi__;
        }
        sux = (uedge[i__ + 1 + (j + k * uedge_dim2) * uedge_dim1] *
               thi_x__ - uedge[i__ + (j + k * uedge_dim2) *
                               uedge_dim1] * tlo_x__) * dxi;
        suy = (vedge[i__ + (j + 1 + k * vedge_dim2) * vedge_dim1] *
               thi_y__ - vedge[i__ + (j + k * vedge_dim2) *
                               vedge_dim1] * tlo_y__) * dyi;
        suz = (wedge[i__ + (j + (k + 1) * wedge_dim2) * wedge_dim1] *
               thi_z__ - wedge[i__ + (j + k * wedge_dim2) *
                               wedge_dim1] * tlo_z__) * dzi;
        cux = s[i__ + (j + k * s_dim2) * s_dim1] * (uedge[i__ + 1 + (
                                                                     j + k * uedge_dim2) * uedge_dim1] - uedge[i__ + (j +
                                                                                                                      k * uedge_dim2) * uedge_dim1]) * dxi;
        cuy = s[i__ + (j + k * s_dim2) * s_dim1] * (vedge[i__ + (j +
                                                                 1 + k * vedge_dim2) * vedge_dim1] - vedge[i__ + (j +
                                                                                                                  k * vedge_dim2) * vedge_dim1]) * dyi;
        cuz = s[i__ + (j + k * s_dim2) * s_dim1] * (wedge[i__ + (j + (
                                                                      k + 1) * wedge_dim2) * wedge_dim1] - wedge[i__ + (j +
                                                                                                                        k * wedge_dim2) * wedge_dim1]) * dzi;
        sleft[i__ + 1] = thi_xlo__ - dth * (suy + suz + cux) * phiinv;
        sright[i__] = tlo_xhi__ - dth * (suy + suz + cux) * phiinv;
        sbck[i__ + (j + 1) * sbck_dim1] = thi_ylo__ - dth * (sux +
                                                             suz + cuy) * phiinv;
        sfrt[i__ + j * sfrt_dim1] = tlo_yhi__ - dth * (sux + suz +
                                                       cuy) * phiinv;
        sbotp[i__ + j * sbotp_dim1] = thi_zlo__ - dth * (sux + suy +
                                                         cuz) * phiinv;
        stop[i__ + j * stop_dim1] = tlo_zhi__ - dth * (sux + suy +
                                                       cuz) * phiinv;
      }
/*     ::: add x contribution to sn */
      if (k >= ks && k <= ke)
      {
        if (j >= js && j <= je)
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            if (uedge[i__ + (j + k * uedge_dim2) * uedge_dim1] >=
                (float)0.)
            {
              supw_m__ = sleft[i__];
            }
            else
            {
              supw_m__ = sright[i__];
            }
            if (uedge[i__ + 1 + (j + k * uedge_dim2) * uedge_dim1]
                >= (float)0.)
            {
              supw_p__ = sleft[i__ + 1];
            }
            else
            {
              supw_p__ = sright[i__ + 1];
            }
            sn[i__ + (j + k * sn_dim2) * sn_dim1] = s[i__ + (j +
                                                             k * s_dim2) * s_dim1] - *dt * (uedge[i__ + 1
                                                                                                  + (j + k * uedge_dim2) * uedge_dim1] *
                                                                                            supw_p__ - uedge[i__ + (j + k * uedge_dim2) *
                                                                                                             uedge_dim1] * supw_m__) / (dx * phi[i__ + (j
                                                                                                                                                        + k * phi_dim2) * phi_dim1]);
          }
        }
      }
    }
/*     ::: add y contributions to sn */
    if (k >= ks && k <= ke)
    {
      i__2 = je;
      for (j = js; j <= i__2; ++j)
      {
        i__3 = ie;
        for (i__ = is; i__ <= i__3; ++i__)
        {
          if (vedge[i__ + (j + k * vedge_dim2) * vedge_dim1] >= (
                                                                 float)0.)
          {
            supw_m__ = sbck[i__ + j * sbck_dim1];
          }
          else
          {
            supw_m__ = sfrt[i__ + j * sfrt_dim1];
          }
          if (vedge[i__ + (j + 1 + k * vedge_dim2) * vedge_dim1] >=
              (float)0.)
          {
            supw_p__ = sbck[i__ + (j + 1) * sbck_dim1];
          }
          else
          {
            supw_p__ = sfrt[i__ + (j + 1) * sfrt_dim1];
          }
          sn[i__ + (j + k * sn_dim2) * sn_dim1] -= *dt * (vedge[i__
                                                                + (j + 1 + k * vedge_dim2) * vedge_dim1] *
                                                          supw_p__ - vedge[i__ + (j + k * vedge_dim2) *
                                                                           vedge_dim1] * supw_m__) / (dy * phi[i__ + (j + k *
                                                                                                                      phi_dim2) * phi_dim1]);
        }
      }
    }
/*     ::: add z contributions to sn */
    if (k >= ks && k <= ke + 1)
    {
      i__2 = je;
      for (j = js; j <= i__2; ++j)
      {
        i__3 = ie;
        for (i__ = is; i__ <= i__3; ++i__)
        {
          if (wedge[i__ + (j + k * wedge_dim2) * wedge_dim1] >= (
                                                                 float)0.)
          {
            supw = sbot[i__ + j * sbot_dim1];
          }
          else
          {
            supw = stop[i__ + j * stop_dim1];
          }
          sfluxz[i__] = wedge[i__ + (j + k * wedge_dim2) *
                              wedge_dim1] * supw * dzi;
        }
        if (k == ks)
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + k * sn_dim2) * sn_dim1] += *dt * sfluxz[
              i__] / phi[i__ + (j + k * phi_dim2) *
                         phi_dim1];
          }
        }
        else if (k == ke + 1)
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + (k - 1) * sn_dim2) * sn_dim1] -= *dt *
                                                           sfluxz[i__] / phi[i__ + (j + (k - 1) *
                                                                                    phi_dim2) * phi_dim1];
          }
        }
        else
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + k * sn_dim2) * sn_dim1] += *dt * sfluxz[
              i__] / phi[i__ + (j + k * phi_dim2) *
                         phi_dim1];
            sn[i__ + (j + (k - 1) * sn_dim2) * sn_dim1] -= *dt *
                                                           sfluxz[i__] / phi[i__ + (j + (k - 1) *
                                                                                    phi_dim2) * phi_dim1];
          }
        }
      }
    }
/*     ::: this should be done by rolling indices */
    i__2 = je;
    for (j = js; j <= i__2; ++j)
    {
      i__3 = ie;
      for (i__ = is; i__ <= i__3; ++i__)
      {
        sbot[i__ + j * sbot_dim1] = sbotp[i__ + j * sbotp_dim1];
      }
    }
/*     ::: roll km, kc, and kp values */
    kt = km;
    km = kc;
    kc = kp;
    kp = kt;
  }
  return 0;
} /* advect_ */

/* ---------------------------------------------------------------------- */
/*     slopexy: */
/*     Compute slopes in x and y */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int slopexy_(s, slx, sly, k, lo, hi, dlo, dhi, dxscr, dyscr)
doublereal * s, *slx, *sly;
integer *k, *lo, *hi, *dlo, *dhi;
doublereal *dxscr, *dyscr;
{
  /* Initialized data */

  logical firstord = FALSE_;
  doublereal zero = 0.;
  doublereal sixth = .166666666666667;
  doublereal half = .5;
  doublereal two3rd = .666666666666667;
  doublereal one = 1.;
  doublereal two = 2.;

  /* System generated locals */
  integer s_dim1, s_dim2, s_offset, slx_dim1, slx_offset, sly_dim1,
          sly_offset, dxscr_dim1, dxscr_offset, dyscr_dim1, dyscr_offset,
          i__1, i__2;
  doublereal d__1, d__2, d__3;

  /* Builtin functions */
  double d_sign();

  /* Local variables */
  doublereal dmin__, dpls;
  integer i__, j, ie, je;
  doublereal ds;
  integer is, js;

  /* Parameter adjustments */
  --lo;
  --hi;
  --dlo;
  --dhi;
  dyscr_dim1 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  dyscr_offset = dlo[2] - 3 + dyscr_dim1;
  dyscr -= dyscr_offset;
  dxscr_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dxscr_offset = dlo[1] - 3 + dxscr_dim1;
  dxscr -= dxscr_offset;
  sly_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  sly_offset = dlo[1] - 2 + sly_dim1 * (dlo[2] - 2);
  sly -= sly_offset;
  slx_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  slx_offset = dlo[1] - 2 + slx_dim1 * (dlo[2] - 2);
  slx -= slx_offset;
  s_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  s_dim2 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  s_offset = dlo[1] - 3 + s_dim1 * (dlo[2] - 3 + s_dim2 * (dlo[3] - 3));
  s -= s_offset;

  /* Function Body */
  is = lo[1];
  js = lo[2];
  ie = hi[1];
  je = hi[2];
  if (firstord)
  {
    i__1 = je + 1;
    for (j = js - 1; j <= i__1; ++j)
    {
      i__2 = ie + 1;
      for (i__ = is - 1; i__ <= i__2; ++i__)
      {
        slx[i__ + j * slx_dim1] = zero;
        sly[i__ + j * sly_dim1] = zero;
      }
    }
    return 0;
  }
/*     :: SLOPES in the X direction */
  i__1 = je + 1;
  for (j = js - 1; j <= i__1; ++j)
  {
/*     ::::: compute second order limited (fromm) slopes */
    i__2 = ie + 2;
    for (i__ = is - 2; i__ <= i__2; ++i__)
    {
      dxscr[i__ + dxscr_dim1] = half * (s[i__ + 1 + (j + *k * s_dim2) *
                                          s_dim1] - s[i__ - 1 + (j + *k * s_dim2) * s_dim1]);
      dmin__ = two * (s[i__ + (j + *k * s_dim2) * s_dim1] - s[i__ - 1 +
                                                              (j + *k * s_dim2) * s_dim1]);
      dpls = two * (s[i__ + 1 + (j + *k * s_dim2) * s_dim1] - s[i__ + (
                                                                       j + *k * s_dim2) * s_dim1]);
/* Computing MIN */
      d__1 = abs(dmin__), d__2 = abs(dpls);
      dxscr[i__ + (dxscr_dim1 << 1)] = pfmin(d__1, d__2);
      if (dpls * dmin__ < (float)0.)
      {
        dxscr[i__ + (dxscr_dim1 << 1)] = zero;
      }
      dxscr[i__ + dxscr_dim1 * 3] = d_sign(&one, &dxscr[i__ +
                                                        dxscr_dim1]);
/* Computing MIN */
      d__2 = dxscr[i__ + (dxscr_dim1 << 1)], d__3 = (d__1 = dxscr[i__ +
                                                                  dxscr_dim1], abs(d__1));
      dxscr[i__ + (dxscr_dim1 << 2)] = dxscr[i__ + dxscr_dim1 * 3] *
                                       pfmin(d__2, d__3);
    }
    i__2 = ie + 1;
    for (i__ = is - 1; i__ <= i__2; ++i__)
    {
      ds = two * two3rd * dxscr[i__ + dxscr_dim1] - sixth * (dxscr[i__
                                                                   + 1 + (dxscr_dim1 << 2)] + dxscr[i__ - 1 + (dxscr_dim1 <<
                                                                                                               2)]);
/* Computing MIN */
      d__1 = abs(ds), d__2 = dxscr[i__ + (dxscr_dim1 << 1)];
      slx[i__ + j * slx_dim1] = dxscr[i__ + dxscr_dim1 * 3] * pfmin(d__1,
                                                                    d__2);
    }
  }
/*     ::::: SLOPES in the Y direction */
  i__1 = ie + 1;
  for (i__ = is - 1; i__ <= i__1; ++i__)
  {
    i__2 = je + 2;
    for (j = js - 2; j <= i__2; ++j)
    {
      dyscr[j + dyscr_dim1] = half * (s[i__ + (j + 1 + *k * s_dim2) *
                                        s_dim1] - s[i__ + (j - 1 + *k * s_dim2) * s_dim1]);
      dmin__ = two * (s[i__ + (j + *k * s_dim2) * s_dim1] - s[i__ + (j
                                                                     - 1 + *k * s_dim2) * s_dim1]);
      dpls = two * (s[i__ + (j + 1 + *k * s_dim2) * s_dim1] - s[i__ + (
                                                                       j + *k * s_dim2) * s_dim1]);
/* Computing MIN */
      d__1 = abs(dmin__), d__2 = abs(dpls);
      dyscr[j + (dyscr_dim1 << 1)] = pfmin(d__1, d__2);
      if (dpls * dmin__ < (float)0.)
      {
        dyscr[j + (dyscr_dim1 << 1)] = zero;
      }
      dyscr[j + dyscr_dim1 * 3] = d_sign(&one, &dyscr[j + dyscr_dim1]);
/* Computing MIN */
      d__2 = dyscr[j + (dyscr_dim1 << 1)], d__3 = (d__1 = dyscr[j +
                                                                dyscr_dim1], abs(d__1));
      dyscr[j + (dyscr_dim1 << 2)] = dyscr[j + dyscr_dim1 * 3] * pfmin(
                                                                       d__2, d__3);
    }
    i__2 = je + 1;
    for (j = js - 1; j <= i__2; ++j)
    {
      ds = two * two3rd * dyscr[j + dyscr_dim1] - sixth * (dyscr[j + 1
                                                                 + (dyscr_dim1 << 2)] + dyscr[j - 1 + (dyscr_dim1 << 2)]);
/* Computing MIN */
      d__1 = abs(ds), d__2 = dyscr[j + (dyscr_dim1 << 1)];
      sly[i__ + j * sly_dim1] = dyscr[j + dyscr_dim1 * 3] * pfmin(d__1,
                                                                  d__2);
    }
  }
  return 0;
} /* slopexy_ */

/* ---------------------------------------------------------------------- */
/*     slopez: */
/*     Compute slopes in z */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int slopez_(s, slz, k, kk, lo, hi, dlo, dhi, dzscr, dzfrm)
doublereal * s, *slz;
integer *k, *kk, *lo, *hi, *dlo, *dhi;
doublereal *dzscr, *dzfrm;
{
  /* Initialized data */

  logical firstord = FALSE_;
  doublereal zero = 0.;
  doublereal sixth = .166666666666667;
  doublereal half = .5;
  doublereal two3rd = .666666666666667;
  doublereal one = 1.;
  doublereal two = 2.;

  /* System generated locals */
  integer s_dim1, s_dim2, s_offset, slz_dim1, slz_dim2, slz_offset,
          dzscr_dim1, dzscr_offset, dzfrm_dim1, dzfrm_offset, i__1, i__2;
  doublereal d__1, d__2, d__3;

  /* Builtin functions */
  double d_sign();

  /* Local variables */
  doublereal dmin__, dpls;
  integer i__, j, ie, je;
  doublereal ds;
  integer is, js, kt;

  /* Parameter adjustments */
  --lo;
  --hi;
  --dlo;
  --dhi;
  dzfrm_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dzfrm_offset = dlo[1] - 3 + dzfrm_dim1 * (*k - 1);
  dzfrm -= dzfrm_offset;
  dzscr_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  dzscr_offset = dlo[1] - 3 + dzscr_dim1;
  dzscr -= dzscr_offset;
  slz_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  slz_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  slz_offset = dlo[1] - 2 + slz_dim1 * (dlo[2] - 2 + slz_dim2);
  slz -= slz_offset;
  s_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  s_dim2 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  s_offset = dlo[1] - 3 + s_dim1 * (dlo[2] - 3 + s_dim2 * (dlo[3] - 3));
  s -= s_offset;

  /* Function Body */
  is = lo[1];
  js = lo[2];
  ie = hi[1];
  je = hi[2];
  if (firstord)
  {
    i__1 = je + 1;
    for (j = js - 1; j <= i__1; ++j)
    {
      i__2 = ie + 1;
      for (i__ = is - 1; i__ <= i__2; ++i__)
      {
        slz[i__ + (j + *kk * slz_dim2) * slz_dim1] = zero;
      }
    }
    return 0;
  }
/*     ::::: SLOPES in the Z direction */
  i__1 = je + 1;
  for (j = js - 1; j <= i__1; ++j)
  {
    i__2 = ie + 1;
    for (i__ = is - 1; i__ <= i__2; ++i__)
    {
      kt = *k - 1;
      dzscr[i__ + dzscr_dim1] = half * (s[i__ + (j + (kt + 1) * s_dim2)
                                          * s_dim1] - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]);
      dmin__ = two * (s[i__ + (j + kt * s_dim2) * s_dim1] - s[i__ + (j
                                                                     + (kt - 1) * s_dim2) * s_dim1]);
      dpls = two * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] - s[i__ +
                                                                  (j + kt * s_dim2) * s_dim1]);
/* Computing MIN */
      d__1 = abs(dmin__), d__2 = abs(dpls);
      dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      dzscr[i__ + dzscr_dim1 * 3] = d_sign(&one, &dzscr[i__ +
                                                        dzscr_dim1]);
/* Computing MIN */
      d__2 = dzscr[i__ + (dzscr_dim1 << 1)], d__3 = (d__1 = dzscr[i__ +
                                                                  dzscr_dim1], abs(d__1));
      dzfrm[i__ + kt * dzfrm_dim1] = dzscr[i__ + dzscr_dim1 * 3] * pfmin(
                                                                         d__2, d__3);
      kt = *k + 1;
      dzscr[i__ + dzscr_dim1] = half * (s[i__ + (j + (kt + 1) * s_dim2)
                                          * s_dim1] - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]);
      dmin__ = two * (s[i__ + (j + kt * s_dim2) * s_dim1] - s[i__ + (j
                                                                     + (kt - 1) * s_dim2) * s_dim1]);
      dpls = two * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] - s[i__ +
                                                                  (j + kt * s_dim2) * s_dim1]);
/* Computing MIN */
      d__1 = abs(dmin__), d__2 = abs(dpls);
      dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      dzscr[i__ + dzscr_dim1 * 3] = d_sign(&one, &dzscr[i__ +
                                                        dzscr_dim1]);
/* Computing MIN */
      d__2 = dzscr[i__ + (dzscr_dim1 << 1)], d__3 = (d__1 = dzscr[i__ +
                                                                  dzscr_dim1], abs(d__1));
      dzfrm[i__ + kt * dzfrm_dim1] = dzscr[i__ + dzscr_dim1 * 3] * pfmin(
                                                                         d__2, d__3);
      dzscr[i__ + dzscr_dim1] = half * (s[i__ + (j + (*k + 1) * s_dim2)
                                          * s_dim1] - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]);
      dmin__ = two * (s[i__ + (j + *k * s_dim2) * s_dim1] - s[i__ + (j
                                                                     + (*k - 1) * s_dim2) * s_dim1]);
      dpls = two * (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] - s[i__ +
                                                                  (j + *k * s_dim2) * s_dim1]);
/* Computing MIN */
      d__1 = abs(dmin__), d__2 = abs(dpls);
      dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      dzscr[i__ + dzscr_dim1 * 3] = d_sign(&one, &dzscr[i__ +
                                                        dzscr_dim1]);
/* Computing MIN */
      d__2 = dzscr[i__ + (dzscr_dim1 << 1)], d__3 = (d__1 = dzscr[i__ +
                                                                  dzscr_dim1], abs(d__1));
      dzfrm[i__ + *k * dzfrm_dim1] = dzscr[i__ + dzscr_dim1 * 3] * pfmin(
                                                                         d__2, d__3);
    }
    i__2 = ie + 1;
    for (i__ = is - 1; i__ <= i__2; ++i__)
    {
      ds = two * two3rd * dzscr[i__ + dzscr_dim1] - sixth * (dzfrm[i__
                                                                   + (*k + 1) * dzfrm_dim1] + dzfrm[i__ + (*k - 1) *
                                                                                                    dzfrm_dim1]);
/* Computing MIN */
      d__1 = abs(ds), d__2 = dzscr[i__ + (dzscr_dim1 << 1)];
      slz[i__ + (j + *kk * slz_dim2) * slz_dim1] = dzscr[i__ +
                                                         dzscr_dim1 * 3] * pfmin(d__1, d__2);
    }
  }
  return 0;
} /* slopez_ */

