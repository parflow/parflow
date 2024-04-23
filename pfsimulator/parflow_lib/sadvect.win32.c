/* sadvect.f -- translated by f2c (version 19960315).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

#include "f2c.h"

/* Table of constant values */

integer c__1 = 1;

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
/*     sadvect: */
/*     Godunov advection routine */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int sadvect_(s, sn, uedge, vedge, wedge, betaedge, phi,
                              viscos, densty, gravty, slx, sly, slz, lohi, dlohi, hx, dt, sbot,
                              stop, sbotp, sfrt, sbck, sleft, sright, sfluxz, dxscr, dyscr, dzscr,
                              dzfrm)
doublereal * s, *sn, *uedge, *vedge, *wedge, *betaedge, *phi, *viscos, *densty,
*gravty, *slx, *sly, *slz;
integer *lohi, *dlohi;
doublereal *hx, *dt, *sbot, *stop, *sbotp, *sfrt, *sbck, *sleft, *sright, *
  sfluxz, *dxscr, *dyscr, *dzscr, *dzfrm;
{
  /* Initialized data */

  doublereal half = .5;
  logical firstord = FALSE_;

  /* System generated locals */
  integer s_dim1, s_dim2, s_offset, sn_dim1, sn_dim2, sn_offset, uedge_dim1,
    uedge_dim2, uedge_offset, vedge_dim1, vedge_dim2, vedge_offset,
    wedge_dim1, wedge_dim2, wedge_offset, betaedge_dim1,
    betaedge_dim2, betaedge_offset, phi_dim1, phi_dim2, phi_offset,
    slx_dim1, slx_offset, sly_dim1, sly_offset, slz_dim1, slz_dim2,
    slz_offset, sbot_dim1, sbot_offset, stop_dim1, stop_offset,
    sbotp_dim1, sbotp_offset, sbck_dim1, sbck_offset, sfrt_dim1,
    sfrt_offset, sleft_offset, sright_offset, sfluxz_offset,
    dxscr_dim1, dxscr_offset, dyscr_dim1, dyscr_offset, dzscr_dim1,
    dzscr_offset, dzfrm_dim1, dzfrm_offset, i__1, i__2, i__3;
  doublereal d__1, d__2, d__3;

  /* Local variables */
  doublereal beta, dxhi, dyhi, dzhi, supw, den0i, den1i;
  extern /* Subroutine */ int sslopexy_();
  doublereal g;
  integer i__, j, k;
  doublereal thi_x__, thi_y__, thi_z__, tlo_x__, tlo_y__, tlo_z__;
  integer ie, je, ke, kc, km;
  doublereal wc;
  integer kp, is, js, ks, kt;
  doublereal dx, dy, dz, phiinv, supw_m__, supw_p__, mu0, mu1;
  extern /* Subroutine */ int rpsolv_();
  doublereal dth, dxh, dyh, dzh, dxi, dyi, dzi, cux, cuy, cuz,
    thi_xhi__, thi_yhi__, thi_zhi__, sux, suy, suz, tlo_xhi__, den0,
    den1, tlo_yhi__, tlo_zhi__, thi_xlo__, thi_ylo__, thi_zlo__,
    tlo_xlo__, tlo_ylo__, tlo_zlo__, mu0i, mu1i;
  extern /* Subroutine */ int sslopez_();

/*     ::: argument declarations */
/* Parameter adjustments */
  --viscos;
  --densty;
  lohi -= 4;
  dlohi -= 4;
  dzfrm_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  dzfrm_offset = dlohi[4] - 3 + dzfrm_dim1;
  dzfrm -= dzfrm_offset;
  dzscr_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  dzscr_offset = dlohi[4] - 3 + dzscr_dim1;
  dzscr -= dzscr_offset;
  dyscr_dim1 = dlohi[8] + 3 - (dlohi[5] - 3) + 1;
  dyscr_offset = dlohi[5] - 3 + dyscr_dim1;
  dyscr -= dyscr_offset;
  dxscr_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  dxscr_offset = dlohi[4] - 3 + dxscr_dim1;
  dxscr -= dxscr_offset;
  sfluxz_offset = dlohi[4] - 3;
  sfluxz -= sfluxz_offset;
  sright_offset = dlohi[4] - 3;
  sright -= sright_offset;
  sleft_offset = dlohi[4] - 3;
  sleft -= sleft_offset;
  sbck_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  sbck_offset = dlohi[4] - 3 + sbck_dim1 * (dlohi[5] - 3);
  sbck -= sbck_offset;
  sfrt_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  sfrt_offset = dlohi[4] - 3 + sfrt_dim1 * (dlohi[5] - 3);
  sfrt -= sfrt_offset;
  sbotp_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  sbotp_offset = dlohi[4] - 3 + sbotp_dim1 * (dlohi[5] - 3);
  sbotp -= sbotp_offset;
  stop_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  stop_offset = dlohi[4] - 3 + stop_dim1 * (dlohi[5] - 3);
  stop -= stop_offset;
  sbot_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  sbot_offset = dlohi[4] - 3 + sbot_dim1 * (dlohi[5] - 3);
  sbot -= sbot_offset;
  slz_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  slz_dim2 = dlohi[8] + 2 - (dlohi[5] - 2) + 1;
  slz_offset = dlohi[4] - 2 + slz_dim1 * (dlohi[5] - 2 + slz_dim2);
  slz -= slz_offset;
  sly_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  sly_offset = dlohi[4] - 2 + sly_dim1 * (dlohi[5] - 2);
  sly -= sly_offset;
  slx_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  slx_offset = dlohi[4] - 2 + slx_dim1 * (dlohi[5] - 2);
  slx -= slx_offset;
  phi_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  phi_dim2 = dlohi[8] + 2 - (dlohi[5] - 2) + 1;
  phi_offset = dlohi[4] - 2 + phi_dim1 * (dlohi[5] - 2 + phi_dim2 * (dlohi[
                                                                       6] - 2));
  phi -= phi_offset;
  betaedge_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  betaedge_dim2 = dlohi[8] + 2 - (dlohi[5] - 2) + 1;
  betaedge_offset = dlohi[4] - 2 + betaedge_dim1 * (dlohi[5] - 2 +
                                                    betaedge_dim2 * (dlohi[6] - 2));
  betaedge -= betaedge_offset;
  wedge_dim1 = dlohi[7] + 2 - (dlohi[4] - 2) + 1;
  wedge_dim2 = dlohi[8] + 2 - (dlohi[5] - 2) + 1;
  wedge_offset = dlohi[4] - 2 + wedge_dim1 * (dlohi[5] - 2 + wedge_dim2 * (
                                                                           dlohi[6] - 2));
  wedge -= wedge_offset;
  vedge_dim1 = dlohi[7] + 1 - (dlohi[4] - 1) + 1;
  vedge_dim2 = dlohi[8] + 2 - (dlohi[5] - 1) + 1;
  vedge_offset = dlohi[4] - 1 + vedge_dim1 * (dlohi[5] - 1 + vedge_dim2 * (
                                                                           dlohi[6] - 1));
  vedge -= vedge_offset;
  uedge_dim1 = dlohi[7] + 2 - (dlohi[4] - 1) + 1;
  uedge_dim2 = dlohi[8] + 1 - (dlohi[5] - 1) + 1;
  uedge_offset = dlohi[4] - 1 + uedge_dim1 * (dlohi[5] - 1 + uedge_dim2 * (
                                                                           dlohi[6] - 1));
  uedge -= uedge_offset;
  sn_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  sn_dim2 = dlohi[8] + 3 - (dlohi[5] - 3) + 1;
  sn_offset = dlohi[4] - 3 + sn_dim1 * (dlohi[5] - 3 + sn_dim2 * (dlohi[6]
                                                                  - 3));
  sn -= sn_offset;
  s_dim1 = dlohi[7] + 3 - (dlohi[4] - 3) + 1;
  s_dim2 = dlohi[8] + 3 - (dlohi[5] - 3) + 1;
  s_offset = dlohi[4] - 3 + s_dim1 * (dlohi[5] - 3 + s_dim2 * (dlohi[6] - 3)
                                      );
  s -= s_offset;
  --hx;

  /* Function Body */

/*     All the statement functions are here. */


/*      Code starts here */

  is = lohi[4];
  ie = lohi[7];
  js = lohi[5];
  je = lohi[8];
  ks = lohi[6];
  ke = lohi[9];
  dx = hx[1];
  dy = hx[2];
  dz = hx[3];
  dxh = half * dx;
  dyh = half * dy;
  dzh = half * dz;
  dxi = (float)1. / dx;
  dyi = (float)1. / dy;
  dzi = (float)1. / dz;
  dxhi = (float)1. / dxh;
  dyhi = (float)1. / dyh;
  dzhi = (float)1. / dzh;
  dth = half * *dt;
  mu0 = viscos[1];
  mu1 = viscos[2];
  den0 = densty[1];
  den1 = densty[2];
  mu0i = (float)1. / mu0;
  mu1i = (float)1. / mu1;
  den0i = (float)1. / den0;
  den1i = (float)1. / den1;
  g = -(*gravty);
  beta = (den0 - den1) * g;
  km = 3;
  kc = 1;
  kp = 2;
/* ---------------------------------------------------------- */
/*     k = ks-1, ke+1 loop */
/* ---------------------------------------------------------- */
  if (!firstord)
  {
    i__1 = ks - 1;
    sslopez_(&s[s_offset], &mu0, &mu1, &wedge[wedge_offset], &betaedge[
               betaedge_offset], &beta, &slz[slz_offset], &i__1, &kc, &lohi[
               4], &lohi[7], &dlohi[4], &dlohi[7], &dzscr[dzscr_offset], &
             dzfrm[dzfrm_offset]);
  }
  i__1 = ke + 1;
  for (k = ks - 1; k <= i__1; ++k)
  {
    if (!firstord)
    {
      sslopexy_(&s[s_offset], &mu0, &mu1, &slx[slx_offset], &sly[
                  sly_offset], &k, &lohi[4], &lohi[7], &dlohi[4], &dlohi[7],
                &dxscr[dxscr_offset], &dyscr[dyscr_offset]);
      if (k <= ke)
      {
        i__2 = k + 1;
        sslopez_(&s[s_offset], &mu0, &mu1, &wedge[wedge_offset], &
                 betaedge[betaedge_offset], &beta, &slz[slz_offset], &
                 i__2, &kp, &lohi[4], &lohi[7], &dlohi[4], &dlohi[7], &
                 dzscr[dzscr_offset], &dzfrm[dzfrm_offset]);
      }
    }
    i__2 = je + 1;
    for (j = js - 1; j <= i__2; ++j)
    {
      i__3 = ie + 1;
      for (i__ = is - 1; i__ <= i__3; ++i__)
      {
        phiinv = (float)1. / phi[i__ + (j + k * phi_dim2) * phi_dim1];
/*              phiinv = 1./phi(i-1,j,k) was a mistake, I beli
 * eve */
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
/* Computing 2nd power */
          d__1 = s[i__ - 1 + (j + k * s_dim2) * s_dim1] * s[i__ - 1
                                                            + (j + k * s_dim2) * s_dim1] * mu0i + ((float)1.
                                                                                                   - s[i__ - 1 + (j + k * s_dim2) * s_dim1]) * ((
                                                                                                                                                 float)1. - s[i__ - 1 + (j + k * s_dim2) * s_dim1])
                 * mu1i;
          tlo_xlo__ += (half - (s[i__ - 1 + (j + k * s_dim2) *
                                  s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                               - 1 + (j + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                    s[i__ - 1 + (j + k * s_dim2) * s_dim1]) * mu1i) -
                                s[i__ - 1 + (j + k * s_dim2) * s_dim1] * s[i__ -
                                                                           1 + (j + k * s_dim2) * s_dim1] * mu0i * (((float)
                                                                                                                     1. - s[i__ - 1 + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                     float)-2. * mu1i)) / (d__1 * d__1) * uedge[i__ + (
                                                                                                                                                                                                                       j + k * uedge_dim2) * uedge_dim1] * dth * dxi /
                        phi[i__ - 1 + (j + k * phi_dim2) * phi_dim1]) *
                       slx[i__ - 1 + j * slx_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          tlo_xhi__ -= (half + (s[i__ + (j + k * s_dim2) * s_dim1] *
                                (float)2. * mu0i * (((float)1. - s[i__ + (j + k *
                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + k
                                                                                                                     * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                   s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                           s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__1 *
                                                                                                                                                                                                                                                                       d__1) * uedge[i__ + (j + k * uedge_dim2) *
                                                                                                                                                                                                                                                                                     uedge_dim1] * dth * dxi * phiinv) * slx[i__ + j *
                                                                                                                                                                                                                                                                                                                             slx_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          thi_xlo__ += (half - (s[i__ + (j + k * s_dim2) * s_dim1] *
                                (float)2. * mu0i * (((float)1. - s[i__ + (j + k *
                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + k
                                                                                                                     * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                   s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                           s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__1 *
                                                                                                                                                                                                                                                                       d__1) * uedge[i__ + 1 + (j + k * uedge_dim2) *
                                                                                                                                                                                                                                                                                     uedge_dim1] * dth * dxi * phiinv) * slx[i__ + j *
                                                                                                                                                                                                                                                                                                                             slx_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + 1 + (j + k * s_dim2) * s_dim1] * s[i__ + 1
                                                            + (j + k * s_dim2) * s_dim1] * mu0i + ((float)1.
                                                                                                   - s[i__ + 1 + (j + k * s_dim2) * s_dim1]) * ((
                                                                                                                                                 float)1. - s[i__ + 1 + (j + k * s_dim2) * s_dim1])
                 * mu1i;
          thi_xhi__ -= (half + (s[i__ + 1 + (j + k * s_dim2) *
                                  s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                               + 1 + (j + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                    s[i__ + 1 + (j + k * s_dim2) * s_dim1]) * mu1i) -
                                s[i__ + 1 + (j + k * s_dim2) * s_dim1] * s[i__ +
                                                                           1 + (j + k * s_dim2) * s_dim1] * mu0i * (((float)
                                                                                                                     1. - s[i__ + 1 + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                     float)-2. * mu1i)) / (d__1 * d__1) * uedge[i__ +
                                                                                                                                                                                                                1 + (j + k * uedge_dim2) * uedge_dim1] * dth *
                        dxi / phi[i__ + 1 + (j + k * phi_dim2) * phi_dim1]
                        ) * slx[i__ + 1 + j * slx_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j - 1 + k * s_dim2) * s_dim1] * s[i__ + (
                                                                   j - 1 + k * s_dim2) * s_dim1] * mu0i + ((float)1.
                                                                                                           - s[i__ + (j - 1 + k * s_dim2) * s_dim1]) * ((
                                                                                                                                                         float)1. - s[i__ + (j - 1 + k * s_dim2) * s_dim1])
                 * mu1i;
          tlo_ylo__ += (half - (s[i__ + (j - 1 + k * s_dim2) *
                                  s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                               + (j - 1 + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                    s[i__ + (j - 1 + k * s_dim2) * s_dim1]) * mu1i) -
                                s[i__ + (j - 1 + k * s_dim2) * s_dim1] * s[i__ + (
                                                                                  j - 1 + k * s_dim2) * s_dim1] * mu0i * (((float)
                                                                                                                           1. - s[i__ + (j - 1 + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                           float)-2. * mu1i)) / (d__1 * d__1) * vedge[i__ + (
                                                                                                                                                                                                                             j + k * vedge_dim2) * vedge_dim1] * dth * dyi /
                        phi[i__ + (j - 1 + k * phi_dim2) * phi_dim1]) *
                       sly[i__ + (j - 1) * sly_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          tlo_yhi__ -= (half + (s[i__ + (j + k * s_dim2) * s_dim1] *
                                (float)2. * mu0i * (((float)1. - s[i__ + (j + k *
                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + k
                                                                                                                     * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                   s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                           s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__1 *
                                                                                                                                                                                                                                                                       d__1) * vedge[i__ + (j + k * vedge_dim2) *
                                                                                                                                                                                                                                                                                     vedge_dim1] * dth * dyi * phiinv) * sly[i__ + j *
                                                                                                                                                                                                                                                                                                                             sly_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          thi_ylo__ += (half - (s[i__ + (j + k * s_dim2) * s_dim1] *
                                (float)2. * mu0i * (((float)1. - s[i__ + (j + k *
                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + k
                                                                                                                     * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                   s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                           s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__1 *
                                                                                                                                                                                                                                                                       d__1) * vedge[i__ + (j + 1 + k * vedge_dim2) *
                                                                                                                                                                                                                                                                                     vedge_dim1] * dth * dyi * phiinv) * sly[i__ + j *
                                                                                                                                                                                                                                                                                                                             sly_dim1];
/* Computing 2nd power */
          d__1 = s[i__ + (j + 1 + k * s_dim2) * s_dim1] * s[i__ + (
                                                                   j + 1 + k * s_dim2) * s_dim1] * mu0i + ((float)1.
                                                                                                           - s[i__ + (j + 1 + k * s_dim2) * s_dim1]) * ((
                                                                                                                                                         float)1. - s[i__ + (j + 1 + k * s_dim2) * s_dim1])
                 * mu1i;
          thi_yhi__ -= (half + (s[i__ + (j + 1 + k * s_dim2) *
                                  s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                               + (j + 1 + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                    s[i__ + (j + 1 + k * s_dim2) * s_dim1]) * mu1i) -
                                s[i__ + (j + 1 + k * s_dim2) * s_dim1] * s[i__ + (
                                                                                  j + 1 + k * s_dim2) * s_dim1] * mu0i * (((float)
                                                                                                                           1. - s[i__ + (j + 1 + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                           float)-2. * mu1i)) / (d__1 * d__1) * vedge[i__ + (
                                                                                                                                                                                                                             j + 1 + k * vedge_dim2) * vedge_dim1] * dth * dyi
                        / phi[i__ + (j + 1 + k * phi_dim2) * phi_dim1]) *
                       sly[i__ + (j + 1) * sly_dim1];
          d__1 = beta * betaedge[i__ + (j + (k - 1) * betaedge_dim2)
                                 * betaedge_dim1];
/* Computing 2nd power */
          d__2 = s[i__ + (j + (k - 1) * s_dim2) * s_dim1] * s[i__ +
                                                              (j + (k - 1) * s_dim2) * s_dim1] * mu0i + ((float)
                                                                                                         1. - s[i__ + (j + (k - 1) * s_dim2) * s_dim1]) * (
                                                                                                                                                           (float)1. - s[i__ + (j + (k - 1) * s_dim2) *
                                                                                                                                                                         s_dim1]) * mu1i;
/* Computing 2nd power */
          d__3 = s[i__ + (j + (k - 1) * s_dim2) * s_dim1] * s[i__ +
                                                              (j + (k - 1) * s_dim2) * s_dim1] * mu0i + ((float)
                                                                                                         1. - s[i__ + (j + (k - 1) * s_dim2) * s_dim1]) * (
                                                                                                                                                           (float)1. - s[i__ + (j + (k - 1) * s_dim2) *
                                                                                                                                                                         s_dim1]) * mu1i;
          tlo_zlo__ += (half - ((s[i__ + (j + (k - 1) * s_dim2) *
                                   s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                                + (j + (k - 1) * s_dim2) * s_dim1]) * ((float)1.
                                                                                                                       - s[i__ + (j + (k - 1) * s_dim2) * s_dim1]) *
                                                                 mu1i) - s[i__ + (j + (k - 1) * s_dim2) * s_dim1] *
                                 s[i__ + (j + (k - 1) * s_dim2) * s_dim1] * mu0i *
                                 (((float)1. - s[i__ + (j + (k - 1) * s_dim2) *
                                                 s_dim1]) * (float)-2. * mu1i)) / (d__2 * d__2) *
                                wedge[i__ + (j + k * wedge_dim2) * wedge_dim1] + (
                                                                                  s[i__ + (j + (k - 1) * s_dim2) * s_dim1] * (float)
                                                                                  2. * mu0i * (((float)1. - s[i__ + (j + (k - 1) *
                                                                                                                     s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (k
                                                                                                                                                                     - 1) * s_dim2) * s_dim1]) * mu1i) * (((float)1. -
                                                                                                                                                                                                           s[i__ + (j + (k - 1) * s_dim2) * s_dim1]) * ((
                                                                                                                                                                                                                                                         float)1. - s[i__ + (j + (k - 1) * s_dim2) *
                                                                                                                                                                                                                                                                      s_dim1]) * mu1i) + ((float)1. - s[i__ + (j + (k -
                                                                                                                                                                                                                                                                                                                    1) * s_dim2) * s_dim1]) * (float)-2. * mu1i * (s[
                                                                                                                                                                                                                                                                                                                                                                     i__ + (j + (k - 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                                                                                                                                                                                                                                                                                                                                                                       j + (k - 1) * s_dim2) * s_dim1] * mu0i) * (s[i__
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    + (j + (k - 1) * s_dim2) * s_dim1] * s[i__ + (j +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (k - 1) * s_dim2) * s_dim1] * mu0i)) / (d__3 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          d__3) * d__1) * dth * dzi / phi[i__ + (j + (k - 1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 * phi_dim2) * phi_dim1]) * slz[i__ + (j + km *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       slz_dim2) * slz_dim1];
          d__1 = beta * betaedge[i__ + (j + k * betaedge_dim2) *
                                 betaedge_dim1];
/* Computing 2nd power */
          d__2 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
/* Computing 2nd power */
          d__3 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          tlo_zhi__ -= (half + ((s[i__ + (j + k * s_dim2) * s_dim1]
                                 * (float)2. * mu0i * (((float)1. - s[i__ + (j + k
                                                                             * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                          k * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                                    s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                          s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                                  s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__2 *
                                                                                                                                                                                                                                                                              d__2) * wedge[i__ + (j + k * wedge_dim2) *
                                                                                                                                                                                                                                                                                            wedge_dim1] + (s[i__ + (j + k * s_dim2) * s_dim1]
                                                                                                                                                                                                                                                                                                           * (float)2. * mu0i * (((float)1. - s[i__ + (j + k
                                                                                                                                                                                                                                                                                                                                                       * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                                                                                                    k * s_dim2) * s_dim1]) * mu1i) * (((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                         i__ + (j + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              s[i__ + (j + k * s_dim2) * s_dim1]) * mu1i) + ((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                float)-2. * mu1i * (s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim1] * s[i__ + (j + k * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    mu0i) * (s[i__ + (j + k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               i__ + (j + k * s_dim2) * s_dim1] * mu0i)) / (d__3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            * d__3) * d__1) * dth * dzi * phiinv) * slz[i__ +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (j + kc * slz_dim2) * slz_dim1];
          d__1 = beta * betaedge[i__ + (j + k * betaedge_dim2) *
                                 betaedge_dim1];
/* Computing 2nd power */
          d__2 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
/* Computing 2nd power */
          d__3 = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j +
                                                               k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__
                                                                                                             + (j + k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                i__ + (j + k * s_dim2) * s_dim1]) * mu1i;
          thi_zlo__ += (half - ((s[i__ + (j + k * s_dim2) * s_dim1]
                                 * (float)2. * mu0i * (((float)1. - s[i__ + (j + k
                                                                             * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                          k * s_dim2) * s_dim1]) * mu1i) - s[i__ + (j + k *
                                                                                                                                                                    s_dim2) * s_dim1] * s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                          s_dim1] * mu0i * (((float)1. - s[i__ + (j + k *
                                                                                                                                                                                                                                  s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__2 *
                                                                                                                                                                                                                                                                              d__2) * wedge[i__ + (j + (k + 1) * wedge_dim2) *
                                                                                                                                                                                                                                                                                            wedge_dim1] + (s[i__ + (j + k * s_dim2) * s_dim1]
                                                                                                                                                                                                                                                                                                           * (float)2. * mu0i * (((float)1. - s[i__ + (j + k
                                                                                                                                                                                                                                                                                                                                                       * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                                                                                                    k * s_dim2) * s_dim1]) * mu1i) * (((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                         i__ + (j + k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              s[i__ + (j + k * s_dim2) * s_dim1]) * mu1i) + ((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                float)-2. * mu1i * (s[i__ + (j + k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim1] * s[i__ + (j + k * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    mu0i) * (s[i__ + (j + k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               i__ + (j + k * s_dim2) * s_dim1] * mu0i)) / (d__3
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            * d__3) * d__1) * dth * dzi * phiinv) * slz[i__ +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (j + kc * slz_dim2) * slz_dim1];
          d__1 = beta * betaedge[i__ + (j + (k + 1) * betaedge_dim2)
                                 * betaedge_dim1];
/* Computing 2nd power */
          d__2 = s[i__ + (j + (k + 1) * s_dim2) * s_dim1] * s[i__ +
                                                              (j + (k + 1) * s_dim2) * s_dim1] * mu0i + ((float)
                                                                                                         1. - s[i__ + (j + (k + 1) * s_dim2) * s_dim1]) * (
                                                                                                                                                           (float)1. - s[i__ + (j + (k + 1) * s_dim2) *
                                                                                                                                                                         s_dim1]) * mu1i;
/* Computing 2nd power */
          d__3 = s[i__ + (j + (k + 1) * s_dim2) * s_dim1] * s[i__ +
                                                              (j + (k + 1) * s_dim2) * s_dim1] * mu0i + ((float)
                                                                                                         1. - s[i__ + (j + (k + 1) * s_dim2) * s_dim1]) * (
                                                                                                                                                           (float)1. - s[i__ + (j + (k + 1) * s_dim2) *
                                                                                                                                                                         s_dim1]) * mu1i;
          thi_zhi__ -= (half + ((s[i__ + (j + (k + 1) * s_dim2) *
                                   s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                                + (j + (k + 1) * s_dim2) * s_dim1]) * ((float)1.
                                                                                                                       - s[i__ + (j + (k + 1) * s_dim2) * s_dim1]) *
                                                                 mu1i) - s[i__ + (j + (k + 1) * s_dim2) * s_dim1] *
                                 s[i__ + (j + (k + 1) * s_dim2) * s_dim1] * mu0i *
                                 (((float)1. - s[i__ + (j + (k + 1) * s_dim2) *
                                                 s_dim1]) * (float)-2. * mu1i)) / (d__2 * d__2) *
                                wedge[i__ + (j + (k + 1) * wedge_dim2) *
                                      wedge_dim1] + (s[i__ + (j + (k + 1) * s_dim2) *
                                                       s_dim1] * (float)2. * mu0i * (((float)1. - s[i__
                                                                                                    + (j + (k + 1) * s_dim2) * s_dim1]) * ((float)1.
                                                                                                                                           - s[i__ + (j + (k + 1) * s_dim2) * s_dim1]) *
                                                                                     mu1i) * (((float)1. - s[i__ + (j + (k + 1) *
                                                                                                                    s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (k
                                                                                                                                                                    + 1) * s_dim2) * s_dim1]) * mu1i) + ((float)1. -
                                                                                                                                                                                                         s[i__ + (j + (k + 1) * s_dim2) * s_dim1]) * (
                                                                                                                                                                                                                                                      float)-2. * mu1i * (s[i__ + (j + (k + 1) * s_dim2)
                                                                                                                                                                                                                                                                            * s_dim1] * s[i__ + (j + (k + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                          s_dim1] * mu0i) * (s[i__ + (j + (k + 1) * s_dim2)
                                                                                                                                                                                                                                                                                                               * s_dim1] * s[i__ + (j + (k + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                             s_dim1] * mu0i)) / (d__3 * d__3) * d__1) * dth *
                        dzi / phi[i__ + (j + (k + 1) * phi_dim2) *
                                  phi_dim1]) * slz[i__ + (j + kp * slz_dim2) *
                                                   slz_dim1];
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
        d__1 = betaedge[i__ + (j + k * betaedge_dim2) * betaedge_dim1]
               * beta;
        rpsolv_(&tlo_zlo__, &tlo_zhi__, &wedge[i__ + (j + k *
                                                      wedge_dim2) * wedge_dim1], &d__1, &mu0, &mu1, &c__1, &
                wc, &tlo_z__);
        d__1 = beta * betaedge[i__ + (j + (k + 1) * betaedge_dim2) *
                               betaedge_dim1];
        rpsolv_(&thi_zlo__, &thi_zhi__, &wedge[i__ + (j + (k + 1) *
                                                      wedge_dim2) * wedge_dim1], &d__1, &mu0, &mu1, &c__1, &
                wc, &thi_z__);
        sux = (uedge[i__ + 1 + (j + k * uedge_dim2) * uedge_dim1] * (
                                                                     thi_x__ * thi_x__ * mu0i / (thi_x__ * thi_x__ * mu0i
                                                                                                 + ((float)1. - thi_x__) * ((float)1. - thi_x__) *
                                                                                                 mu1i)) - uedge[i__ + (j + k * uedge_dim2) *
                                                                                                                uedge_dim1] * (tlo_x__ * tlo_x__ * mu0i / (tlo_x__ *
                                                                                                                                                           tlo_x__ * mu0i + ((float)1. - tlo_x__) * ((float)1. -
                                                                                                                                                                                                     tlo_x__) * mu1i))) * dxi;
        suy = (vedge[i__ + (j + 1 + k * vedge_dim2) * vedge_dim1] * (
                                                                     thi_y__ * thi_y__ * mu0i / (thi_y__ * thi_y__ * mu0i
                                                                                                 + ((float)1. - thi_y__) * ((float)1. - thi_y__) *
                                                                                                 mu1i)) - vedge[i__ + (j + k * vedge_dim2) *
                                                                                                                vedge_dim1] * (tlo_y__ * tlo_y__ * mu0i / (tlo_y__ *
                                                                                                                                                           tlo_y__ * mu0i + ((float)1. - tlo_y__) * ((float)1. -
                                                                                                                                                                                                     tlo_y__) * mu1i))) * dyi;
        d__1 = beta * betaedge[i__ + (j + (k + 1) * betaedge_dim2) *
                               betaedge_dim1];
        d__2 = beta * betaedge[i__ + (j + k * betaedge_dim2) *
                               betaedge_dim1];
        suz = (thi_z__ * thi_z__ * mu0i / (thi_z__ * thi_z__ * mu0i +
                                           ((float)1. - thi_z__) * ((float)1. - thi_z__) * mu1i)
               * wedge[i__ + (j + (k + 1) * wedge_dim2) * wedge_dim1]
               + thi_z__ * thi_z__ * mu0i * (((float)1. - thi_z__) *
                                             ((float)1. - thi_z__) * mu1i) / (thi_z__ * thi_z__ *
                                                                              mu0i + ((float)1. - thi_z__) * ((float)1. - thi_z__) *
                                                                              mu1i) * d__1 - (tlo_z__ * tlo_z__ * mu0i / (tlo_z__ *
                                                                                                                          tlo_z__ * mu0i + ((float)1. - tlo_z__) * ((float)1.
                                                                                                                                                                    - tlo_z__) * mu1i) * wedge[i__ + (j + k * wedge_dim2)
                                                                                                                                                                                               * wedge_dim1] + tlo_z__ * tlo_z__ * mu0i * (((float)
                                                                                                                                                                                                                                            1. - tlo_z__) * ((float)1. - tlo_z__) * mu1i) / (
                                                                                                                                                                                                                                                                                             tlo_z__ * tlo_z__ * mu0i + ((float)1. - tlo_z__) * ((
                                                                                                                                                                                                                                                                                                                                                  float)1. - tlo_z__) * mu1i) * d__2)) * dzi;
        cux = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j + k *
                                                            s_dim2) * s_dim1] * mu0i / (s[i__ + (j + k * s_dim2) *
                                                                                          s_dim1] * s[i__ + (j + k * s_dim2) * s_dim1] * mu0i
                                                                                        + ((float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                              (float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) *
                                                                                        mu1i) * (uedge[i__ + 1 + (j + k * uedge_dim2) *
                                                                                                       uedge_dim1] - uedge[i__ + (j + k * uedge_dim2) *
                                                                                                                           uedge_dim1]) * dxi;
        cuy = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j + k *
                                                            s_dim2) * s_dim1] * mu0i / (s[i__ + (j + k * s_dim2) *
                                                                                          s_dim1] * s[i__ + (j + k * s_dim2) * s_dim1] * mu0i
                                                                                        + ((float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                              (float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) *
                                                                                        mu1i) * (vedge[i__ + (j + 1 + k * vedge_dim2) *
                                                                                                       vedge_dim1] - vedge[i__ + (j + k * vedge_dim2) *
                                                                                                                           vedge_dim1]) * dyi;
        cuz = s[i__ + (j + k * s_dim2) * s_dim1] * s[i__ + (j + k *
                                                            s_dim2) * s_dim1] * mu0i / (s[i__ + (j + k * s_dim2) *
                                                                                          s_dim1] * s[i__ + (j + k * s_dim2) * s_dim1] * mu0i
                                                                                        + ((float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) * (
                                                                                                                                              (float)1. - s[i__ + (j + k * s_dim2) * s_dim1]) *
                                                                                        mu1i) * (wedge[i__ + (j + (k + 1) * wedge_dim2) *
                                                                                                       wedge_dim1] - wedge[i__ + (j + k * wedge_dim2) *
                                                                                                                           wedge_dim1]) * dzi;
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
                                                             k * s_dim2) * s_dim1] - *dt * (supw_p__ *
                                                                                            supw_p__ * mu0i / (supw_p__ * supw_p__ * mu0i
                                                                                                               + ((float)1. - supw_p__) * ((float)1. -
                                                                                                                                           supw_p__) * mu1i) * uedge[i__ + 1 + (j + k *
                                                                                                                                                                                uedge_dim2) * uedge_dim1] - supw_m__ *
                                                                                            supw_m__ * mu0i / (supw_m__ * supw_m__ * mu0i
                                                                                                               + ((float)1. - supw_m__) * ((float)1. -
                                                                                                                                           supw_m__) * mu1i) * uedge[i__ + (j + k *
                                                                                                                                                                            uedge_dim2) * uedge_dim1]) / (dx * phi[i__ + (
                                                                                                                                                                                                                          j + k * phi_dim2) * phi_dim1]);
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
          sn[i__ + (j + k * sn_dim2) * sn_dim1] -= *dt * (supw_p__ *
                                                          supw_p__ * mu0i / (supw_p__ * supw_p__ * mu0i + (
                                                                                                           (float)1. - supw_p__) * ((float)1. - supw_p__) *
                                                                             mu1i) * vedge[i__ + (j + 1 + k * vedge_dim2) *
                                                                                           vedge_dim1] - supw_m__ * supw_m__ * mu0i / (
                                                                                                                                       supw_m__ * supw_m__ * mu0i + ((float)1. -
                                                                                                                                                                     supw_m__) * ((float)1. - supw_m__) * mu1i) *
                                                          vedge[i__ + (j + k * vedge_dim2) * vedge_dim1]) /
                                                   (dy * phi[i__ + (j + k * phi_dim2) * phi_dim1]);
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
          d__1 = beta * betaedge[i__ + (j + k * betaedge_dim2) *
                                 betaedge_dim1];
          rpsolv_(&sbot[i__ + j * sbot_dim1], &stop[i__ + j *
                                                    stop_dim1], &wedge[i__ + (j + k * wedge_dim2) *
                                                                       wedge_dim1], &d__1, &mu0, &mu1, &c__1, &wc, &supw)
          ;
          d__1 = beta * betaedge[i__ + (j + k * betaedge_dim2) *
                                 betaedge_dim1];
          sfluxz[i__] = supw * supw * mu0i / (supw * supw * mu0i + (
                                                                    (float)1. - supw) * ((float)1. - supw) * mu1i) *
                        wedge[i__ + (j + k * wedge_dim2) * wedge_dim1] +
                        supw * supw * mu0i * (((float)1. - supw) * ((
                                                                     float)1. - supw) * mu1i) / (supw * supw * mu0i + (
                                                                                                                       (float)1. - supw) * ((float)1. - supw) * mu1i) *
                        d__1;
        }
        if (k == ks)
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + k * sn_dim2) * sn_dim1] += *dt * sfluxz[
              i__] / (dz * phi[i__ + (j + k * phi_dim2) *
                               phi_dim1]);
          }
        }
        else if (k == ke + 1)
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + (k - 1) * sn_dim2) * sn_dim1] -= *dt *
                                                           sfluxz[i__] / (dz * phi[i__ + (j + (k - 1) *
                                                                                          phi_dim2) * phi_dim1]);
          }
        }
        else
        {
          i__3 = ie;
          for (i__ = is; i__ <= i__3; ++i__)
          {
            sn[i__ + (j + k * sn_dim2) * sn_dim1] += *dt * sfluxz[
              i__] / (dz * phi[i__ + (j + k * phi_dim2) *
                               phi_dim1]);
            sn[i__ + (j + (k - 1) * sn_dim2) * sn_dim1] -= *dt *
                                                           sfluxz[i__] / (dz * phi[i__ + (j + (k - 1) *
                                                                                          phi_dim2) * phi_dim1]);
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
} /* sadvect_ */

/* ---------------------------------------------------------------------- */
/*     sslopexy: */
/*     Compute slopes in x and y */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int sslopexy_(s, mu0, mu1, slx, sly, k, lo, hi, dlo, dhi,
                               dxscr, dyscr)
doublereal * s, *mu0, *mu1, *slx, *sly;
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
  doublereal dmin__, dpls, strc, strm, strp;
  integer i__, j, ie, je;
  doublereal ds;
  integer is, js;
  doublereal mu0i, mu1i;

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
  mu0i = (float)1. / *mu0;
  mu1i = (float)1. / *mu1;
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
      if (dpls * dmin__ < (float)0.)
      {
        dxscr[i__ + (dxscr_dim1 << 1)] = zero;
      }
      else
      {
/* Computing MIN */
        d__1 = abs(dmin__), d__2 = abs(dpls);
        dxscr[i__ + (dxscr_dim1 << 1)] = pfmin(d__1, d__2);
/* Computing 3rd power */
        d__1 = s[i__ - 1 + (j + *k * s_dim2) * s_dim1] * s[i__ - 1 + (
                                                                      j + *k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[
                                                                                                             i__ - 1 + (j + *k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                       s[i__ - 1 + (j + *k * s_dim2) * s_dim1]) * mu1i, d__2
          = d__1;
        strm = ((mu0i * (float)2. * (((float)1. - s[i__ - 1 + (j + *k
                                                               * s_dim2) * s_dim1]) * ((float)1. - s[i__ - 1 + (j + *
                                                                                                                k * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (
                                                                                                                                                                     s[i__ - 1 + (j + *k * s_dim2) * s_dim1] * s[i__ - 1 +
                                                                                                                                                                                                                 (j + *k * s_dim2) * s_dim1] * mu0i)) * (s[i__ - 1 + (
                                                                                                                                                                                                                                                                      j + *k * s_dim2) * s_dim1] * s[i__ - 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ - 1 + (
                                                                                                                                                                                                                                                                                                                                                                     j + *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ - 1
                                                                                                                                                                                                                                                                                                                                                                                                                  + (j + *k * s_dim2) * s_dim1]) * mu1i) - (s[i__ - 1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                              (j + *k * s_dim2) * s_dim1] * (float)2. * mu0i * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  float)1. - s[i__ - 1 + (j + *k * s_dim2) * s_dim1]) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ((float)1. - s[i__ - 1 + (j + *k * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                * mu1i) - ((float)1. - s[i__ - 1 + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim1]) * (float)-2. * mu1i * (s[i__ - 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1] * s[i__ - 1 + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            s_dim1] * mu0i)) * (float)2. * (s[i__ - 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim2) * s_dim1] * (float)2. * mu0i + ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   i__ - 1 + (j + *k * s_dim2) * s_dim1]) * (float)-2. *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            mu1i)) / (d__2 * (d__1 * d__1));
/* Computing 3rd power */
        d__1 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__2 = d__1;
        strc = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + *k *
                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                      s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                         i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                                                                                                                                                                      s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + *k *
                                                                                                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                   s_dim1] * mu0i + ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                     s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                            s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                     s[i__ + (j + *k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               i__ + (j + *k * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             1. - s[i__ + (j + *k * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                   -2. * mu1i * (s[i__ + (j + *k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                   i__ + (j + *k * s_dim2) * s_dim1] * mu0i)) * (float)
                2. * (s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2.
                      * mu0i + ((float)1. - s[i__ + (j + *k * s_dim2) *
                                              s_dim1]) * (float)-2. * mu1i)) / (d__2 * (d__1 * d__1)
                                                                                );
/* Computing 3rd power */
        d__1 = s[i__ + 1 + (j + *k * s_dim2) * s_dim1] * s[i__ + 1 + (
                                                                      j + *k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[
                                                                                                             i__ + 1 + (j + *k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                       s[i__ + 1 + (j + *k * s_dim2) * s_dim1]) * mu1i, d__2
          = d__1;
        strp = ((mu0i * (float)2. * (((float)1. - s[i__ + 1 + (j + *k
                                                               * s_dim2) * s_dim1]) * ((float)1. - s[i__ + 1 + (j + *
                                                                                                                k * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (
                                                                                                                                                                     s[i__ + 1 + (j + *k * s_dim2) * s_dim1] * s[i__ + 1 +
                                                                                                                                                                                                                 (j + *k * s_dim2) * s_dim1] * mu0i)) * (s[i__ + 1 + (
                                                                                                                                                                                                                                                                      j + *k * s_dim2) * s_dim1] * s[i__ + 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + 1 + (
                                                                                                                                                                                                                                                                                                                                                                     j + *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + 1
                                                                                                                                                                                                                                                                                                                                                                                                                  + (j + *k * s_dim2) * s_dim1]) * mu1i) - (s[i__ + 1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                              (j + *k * s_dim2) * s_dim1] * (float)2. * mu0i * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  float)1. - s[i__ + 1 + (j + *k * s_dim2) * s_dim1]) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ((float)1. - s[i__ + 1 + (j + *k * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                * mu1i) - ((float)1. - s[i__ + 1 + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1] * s[i__ + 1 + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            s_dim1] * mu0i)) * (float)2. * (s[i__ + 1 + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim2) * s_dim1] * (float)2. * mu0i + ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   i__ + 1 + (j + *k * s_dim2) * s_dim1]) * (float)-2. *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            mu1i)) / (d__2 * (d__1 * d__1));
        if (strm * strc <= (float)0. || strc * strp <= (float)0.)
        {
          dxscr[i__ + (dxscr_dim1 << 1)] = half * dxscr[i__ + (
                                                               dxscr_dim1 << 1)];
        }
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
      if (dpls * dmin__ < (float)0.)
      {
        dyscr[j + (dyscr_dim1 << 1)] = zero;
      }
      else
      {
/* Computing MIN */
        d__1 = abs(dmin__), d__2 = abs(dpls);
        dyscr[j + (dyscr_dim1 << 1)] = pfmin(d__1, d__2);
/* Computing 3rd power */
        d__1 = s[i__ + (j - 1 + *k * s_dim2) * s_dim1] * s[i__ + (j -
                                                                  1 + *k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[
                                                                                                         i__ + (j - 1 + *k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                   s[i__ + (j - 1 + *k * s_dim2) * s_dim1]) * mu1i, d__2
          = d__1;
        strm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j - 1 + *k
                                                           * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j - 1 + *
                                                                                                        k * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (
                                                                                                                                                             s[i__ + (j - 1 + *k * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                                                                                                - 1 + *k * s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j -
                                                                                                                                                                                                                                                                  1 + *k * s_dim2) * s_dim1] * s[i__ + (j - 1 + *k *
                                                                                                                                                                                                                                                                                                        s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j -
                                                                                                                                                                                                                                                                                                                                                         1 + *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j
                                                                                                                                                                                                                                                                                                                                                                                                             - 1 + *k * s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j -
                                                                                                                                                                                                                                                                                                                                                                                                                                                               1 + *k * s_dim2) * s_dim1] * (float)2. * mu0i * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  float)1. - s[i__ + (j - 1 + *k * s_dim2) * s_dim1]) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ((float)1. - s[i__ + (j - 1 + *k * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                * mu1i) - ((float)1. - s[i__ + (j - 1 + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j - 1 + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  s_dim2) * s_dim1] * s[i__ + (j - 1 + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        s_dim1] * mu0i)) * (float)2. * (s[i__ + (j - 1 + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim2) * s_dim1] * (float)2. * mu0i + ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           i__ + (j - 1 + *k * s_dim2) * s_dim1]) * (float)-2. *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        mu1i)) / (d__2 * (d__1 * d__1));
/* Computing 3rd power */
        d__1 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__2 = d__1;
        strc = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + *k *
                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                      s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                         i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                                                                                                                                                                      s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + *k *
                                                                                                                                                                                                                                             s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                   s_dim1] * mu0i + ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                          s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                     s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                            s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                     s[i__ + (j + *k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               i__ + (j + *k * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             1. - s[i__ + (j + *k * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                   -2. * mu1i * (s[i__ + (j + *k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                   i__ + (j + *k * s_dim2) * s_dim1] * mu0i)) * (float)
                2. * (s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2.
                      * mu0i + ((float)1. - s[i__ + (j + *k * s_dim2) *
                                              s_dim1]) * (float)-2. * mu1i)) / (d__2 * (d__1 * d__1)
                                                                                );
/* Computing 3rd power */
        d__1 = s[i__ + (j + 1 + *k * s_dim2) * s_dim1] * s[i__ + (j +
                                                                  1 + *k * s_dim2) * s_dim1] * mu0i + ((float)1. - s[
                                                                                                         i__ + (j + 1 + *k * s_dim2) * s_dim1]) * ((float)1. -
                                                                                                                                                   s[i__ + (j + 1 + *k * s_dim2) * s_dim1]) * mu1i, d__2
          = d__1;
        strp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + 1 + *k
                                                           * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + 1 + *
                                                                                                        k * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (
                                                                                                                                                             s[i__ + (j + 1 + *k * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                                                                                                + 1 + *k * s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j +
                                                                                                                                                                                                                                                                  1 + *k * s_dim2) * s_dim1] * s[i__ + (j + 1 + *k *
                                                                                                                                                                                                                                                                                                        s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                                                         1 + *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j
                                                                                                                                                                                                                                                                                                                                                                                                             + 1 + *k * s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j +
                                                                                                                                                                                                                                                                                                                                                                                                                                                               1 + *k * s_dim2) * s_dim1] * (float)2. * mu0i * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  float)1. - s[i__ + (j + 1 + *k * s_dim2) * s_dim1]) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ((float)1. - s[i__ + (j + 1 + *k * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                * mu1i) - ((float)1. - s[i__ + (j + 1 + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + 1 + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  s_dim2) * s_dim1] * s[i__ + (j + 1 + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        s_dim1] * mu0i)) * (float)2. * (s[i__ + (j + 1 + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim2) * s_dim1] * (float)2. * mu0i + ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           i__ + (j + 1 + *k * s_dim2) * s_dim1]) * (float)-2. *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        mu1i)) / (d__2 * (d__1 * d__1));
        if (strm * strc <= (float)0. || strc * strp <= (float)0.)
        {
          dyscr[j + (dyscr_dim1 << 1)] = half * dyscr[j + (
                                                           dyscr_dim1 << 1)];
        }
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
} /* sslopexy_ */

/* ---------------------------------------------------------------------- */
/*     sslopez: */
/*     Compute slopes in z */
/* ---------------------------------------------------------------------- */
/* Subroutine */ int sslopez_(s, mu0, mu1, w, betaedge, beta, slz, k, kk, lo,
                              hi, dlo, dhi, dzscr, dzfrm)
doublereal * s, *mu0, *mu1, *w, *betaedge, *beta, *slz;
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
  integer s_dim1, s_dim2, s_offset, w_dim1, w_dim2, w_offset, betaedge_dim1,
    betaedge_dim2, betaedge_offset, slz_dim1, slz_dim2, slz_offset,
    dzscr_dim1, dzscr_offset, dzfrm_dim1, dzfrm_offset, i__1, i__2;
  doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8;

  /* Builtin functions */
  double d_sign();

  /* Local variables */
  doublereal dmin__, dpls, strm, strp;
  integer i__, j;
  doublereal strcm, strcp;
  integer ie, je;
  doublereal ds;
  integer is, js, kt;
  doublereal mu0i, mu1i;

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
  betaedge_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  betaedge_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  betaedge_offset = dlo[1] - 2 + betaedge_dim1 * (dlo[2] - 2 +
                                                  betaedge_dim2 * (dlo[3] - 2));
  betaedge -= betaedge_offset;
  w_dim1 = dhi[1] + 2 - (dlo[1] - 2) + 1;
  w_dim2 = dhi[2] + 2 - (dlo[2] - 2) + 1;
  w_offset = dlo[1] - 2 + w_dim1 * (dlo[2] - 2 + w_dim2 * (dlo[3] - 2));
  w -= w_offset;
  s_dim1 = dhi[1] + 3 - (dlo[1] - 3) + 1;
  s_dim2 = dhi[2] + 3 - (dlo[2] - 3) + 1;
  s_offset = dlo[1] - 3 + s_dim1 * (dlo[2] - 3 + s_dim2 * (dlo[3] - 3));
  s -= s_offset;

  /* Function Body */
  is = lo[1];
  js = lo[2];
  ie = hi[1];
  je = hi[2];
  mu0i = (float)1. / *mu0;
  mu1i = (float)1. / *mu1;
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
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      else
      {
/* Computing MIN */
        d__1 = abs(dmin__), d__2 = abs(dpls);
        dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
        d__1 = *beta * betaedge[i__ + (j + kt * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (kt - 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (kt - 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (kt - 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (kt - 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (kt -
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (
                                                                                                                     kt - 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (kt - 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (kt - 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (kt
                                                                                                                                                                                                                              - 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (kt -
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + kt * w_dim2) * w_dim1]
               + ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (
                                                                  kt - 1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (
                                                                                                                       j + (kt - 1) * s_dim2) * s_dim1]) * mu1i) * (((float)
                                                                                                                                                                     1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((
                                                                                                                                                                                                                         float)1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1])
                                                                                                                                                                    * mu1i) + mu1i * (float)2. * (s[i__ + (j + (kt - 1) *
                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + (kt - 1) * s_dim2) *
                                                                                                                                                                                                                                 s_dim1] * mu0i) * (s[i__ + (j + (kt - 1) * s_dim2) *
                                                                                                                                                                                                                                                      s_dim1] * s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                    mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 / mu1i) -
                  d__6 * d__6 * (float)2.) / (d__8 * (d__7 * d__7)) *
               d__1;
        d__1 = *beta * betaedge[i__ + (j + kt * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + kt * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + kt *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + kt * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + kt * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + kt * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + kt * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + kt * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + kt * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + kt * w_dim2) * w_dim1] + ((mu0i * (
                                                                                                                                     float)2. * (((float)1. - s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                s_dim1]) * ((float)1. - s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                          s_dim1]) * mu1i) * (((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                    s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                               s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (s[
                                                                                                                                                                                                                                                                                                                                  i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                               s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + kt * s_dim2)
                                                                                                                                                                                                                                                                                                                                                                                                              * s_dim1] * s[i__ + (j + kt * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                                                                                                                                                                            mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 / mu1i) -
                                                                                                                            d__6 * d__6 * (float)2.) / (d__8 * (d__7 * d__7)) *
                d__1;
        d__1 = *beta * betaedge[i__ + (j + (kt + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + kt * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + kt *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + kt * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + kt * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + kt * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + kt * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + kt * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + kt * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + (kt + 1) * w_dim2) * w_dim1] + ((
                                                                                                                                   mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                                                                                                                                             s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                        s_dim2) * s_dim1]) * mu1i) * (((float)1. - s[i__ + (j
                                                                                                                                                                                                                                                                            + kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                              kt * s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (
                                                                                                                                                                                                                                                                                                                                                                                    s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt
                                                                                                                                                                                                                                                                                                                                                                                                                                   * s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 d__7)) * d__1;
        d__1 = *beta * betaedge[i__ + (j + (kt + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (kt + 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (kt + 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (kt + 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (kt + 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (kt +
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (
                                                                                                                     kt + 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (kt + 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (kt + 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (kt
                                                                                                                                                                                                                              + 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (kt +
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + (kt + 1) * w_dim2) *
                                                                                                                                                                                                                                                                                        w_dim1] + ((mu0i * (float)2. * (((float)1. - s[i__ + (
                                                                                                                                                                                                                                                                                                                                              j + (kt + 1) * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                     i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * mu1i) * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                           float)1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                         * ((float)1. - s[i__ + (j + (kt + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                          s_dim1]) * mu1i) + mu1i * (float)2. * (s[i__ + (j + (
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               kt + 1) * s_dim2) * s_dim1] * s[i__ + (j + (kt + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + (kt + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            s_dim2) * s_dim1] * s[i__ + (j + (kt + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  d__7)) * d__1;
        if (strm * strcm <= (float)0. || strcp * strp <= (float)0.)
        {
          dzscr[i__ + (dzscr_dim1 << 1)] = half * dzscr[i__ + (
                                                               dzscr_dim1 << 1)];
        }
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
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      else
      {
/* Computing MIN */
        d__1 = abs(dmin__), d__2 = abs(dpls);
        dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
        d__1 = *beta * betaedge[i__ + (j + (kt - 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (kt - 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (kt - 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (kt - 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (kt - 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (kt -
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (
                                                                                                                     kt - 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (kt - 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (kt - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (kt - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (kt - 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (kt
                                                                                                                                                                                                                              - 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (kt -
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + kt * w_dim2) * w_dim1]
               + ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (
                                                                  kt - 1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (
                                                                                                                       j + (kt - 1) * s_dim2) * s_dim1]) * mu1i) * (((float)
                                                                                                                                                                     1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1]) * ((
                                                                                                                                                                                                                         float)1. - s[i__ + (j + (kt - 1) * s_dim2) * s_dim1])
                                                                                                                                                                    * mu1i) + mu1i * (float)2. * (s[i__ + (j + (kt - 1) *
                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + (kt - 1) * s_dim2) *
                                                                                                                                                                                                                                 s_dim1] * mu0i) * (s[i__ + (j + (kt - 1) * s_dim2) *
                                                                                                                                                                                                                                                      s_dim1] * s[i__ + (j + (kt - 1) * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                    mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 / mu1i) -
                  d__6 * d__6 * (float)2.) / (d__8 * (d__7 * d__7)) *
               d__1;
        d__1 = *beta * betaedge[i__ + (j + kt * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + kt * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + kt *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + kt * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + kt * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + kt * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + kt * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + kt * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + kt * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + kt * w_dim2) * w_dim1] + ((mu0i * (
                                                                                                                                     float)2. * (((float)1. - s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                s_dim1]) * ((float)1. - s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                          s_dim1]) * mu1i) * (((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                    s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                               s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (s[
                                                                                                                                                                                                                                                                                                                                  i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                               s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + kt * s_dim2)
                                                                                                                                                                                                                                                                                                                                                                                                              * s_dim1] * s[i__ + (j + kt * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                                                                                                                                                                            mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 / mu1i) -
                                                                                                                            d__6 * d__6 * (float)2.) / (d__8 * (d__7 * d__7)) *
                d__1;
        d__1 = *beta * betaedge[i__ + (j + (kt + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + kt * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + kt * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + kt * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + kt *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                               kt * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + kt * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + kt * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + kt * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + kt * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + kt * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + kt * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + kt * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + (kt + 1) * w_dim2) * w_dim1] + ((
                                                                                                                                   mu0i * (float)2. * (((float)1. - s[i__ + (j + kt *
                                                                                                                                                                             s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + kt *
                                                                                                                                                                                                                        s_dim2) * s_dim1]) * mu1i) * (((float)1. - s[i__ + (j
                                                                                                                                                                                                                                                                            + kt * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                              kt * s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (
                                                                                                                                                                                                                                                                                                                                                                                    s[i__ + (j + kt * s_dim2) * s_dim1] * s[i__ + (j + kt
                                                                                                                                                                                                                                                                                                                                                                                                                                   * s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + kt *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + kt * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 d__7)) * d__1;
        d__1 = *beta * betaedge[i__ + (j + (kt + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (kt + 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (kt + 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (kt + 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (kt + 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (kt +
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (
                                                                                                                     kt + 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (kt + 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (kt + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (kt + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (kt + 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (kt
                                                                                                                                                                                                                              + 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (kt + 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (kt + 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (kt +
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + (kt + 1) * w_dim2) *
                                                                                                                                                                                                                                                                                        w_dim1] + ((mu0i * (float)2. * (((float)1. - s[i__ + (
                                                                                                                                                                                                                                                                                                                                              j + (kt + 1) * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                     i__ + (j + (kt + 1) * s_dim2) * s_dim1]) * mu1i) * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                           float)1. - s[i__ + (j + (kt + 1) * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                         * ((float)1. - s[i__ + (j + (kt + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                          s_dim1]) * mu1i) + mu1i * (float)2. * (s[i__ + (j + (
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               kt + 1) * s_dim2) * s_dim1] * s[i__ + (j + (kt + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + (kt + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            s_dim2) * s_dim1] * s[i__ + (j + (kt + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  d__7)) * d__1;
        if (strm * strcm <= (float)0. || strcp * strp <= (float)0.)
        {
          dzscr[i__ + (dzscr_dim1 << 1)] = half * dzscr[i__ + (
                                                               dzscr_dim1 << 1)];
        }
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
      if (dpls * dmin__ < (float)0.)
      {
        dzscr[i__ + (dzscr_dim1 << 1)] = zero;
      }
      else
      {
/* Computing MIN */
        d__1 = abs(dmin__), d__2 = abs(dpls);
        dzscr[i__ + (dzscr_dim1 << 1)] = pfmin(d__1, d__2);
        d__1 = *beta * betaedge[i__ + (j + (*k - 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (*k - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (*k - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (*k - 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (*k - 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (*k - 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (*k - 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (*k - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (*k -
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (*
                                                                                                                     k - 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (*k - 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (*k - 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (*k - 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (*k - 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (*k
                                                                                                                                                                                                                              - 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (*k - 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (*k - 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (*k - 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (*k -
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + (*k - 1) * w_dim2) *
                                                                                                                                                                                                                                                                                        w_dim1] + ((mu0i * (float)2. * (((float)1. - s[i__ + (
                                                                                                                                                                                                                                                                                                                                              j + (*k - 1) * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                     i__ + (j + (*k - 1) * s_dim2) * s_dim1]) * mu1i) * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                           float)1. - s[i__ + (j + (*k - 1) * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                         * ((float)1. - s[i__ + (j + (*k - 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                          s_dim1]) * mu1i) + mu1i * (float)2. * (s[i__ + (j + (*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               k - 1) * s_dim2) * s_dim1] * s[i__ + (j + (*k - 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + (*k - 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + (*k - 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 d__7)) * d__1;
        d__1 = *beta * betaedge[i__ + (j + *k * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + *k * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + *k * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + *k * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + *k * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + *k * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + *k *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcm = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + *k *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + *k *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + *k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + *k * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + *k * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + *k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + *k * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + *k * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + *k * w_dim2) * w_dim1] + ((mu0i * (
                                                                                                                                     float)2. * (((float)1. - s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                s_dim1]) * ((float)1. - s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                          s_dim1]) * mu1i) * (((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                    s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                               s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (s[
                                                                                                                                                                                                                                                                                                                                  i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                               s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + *k * s_dim2)
                                                                                                                                                                                                                                                                                                                                                                                                              * s_dim1] * s[i__ + (j + *k * s_dim2) * s_dim1] *
                                                                                                                                                                                                                                                                                                                                                                                                            mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 / mu1i) -
                                                                                                                            d__6 * d__6 * (float)2.) / (d__8 * (d__7 * d__7)) *
                d__1;
        d__1 = *beta * betaedge[i__ + (j + (*k + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + *k * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + *k * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2. * mu0i
               * (((float)1. - s[i__ + (j + *k * s_dim2) * s_dim1]) *
                  ((float)1. - s[i__ + (j + *k * s_dim2) * s_dim1]) *
                  mu1i) - ((float)1. - s[i__ + (j + *k * s_dim2) *
                                         s_dim1]) * (float)-2. * mu1i * (s[i__ + (j + *k *
                                                                                  s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                        s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                              s_dim2) * s_dim1] * mu0i + ((float)1. - s[i__ + (j +
                                                                                                               *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *
                                                                                                                                                               k * s_dim2) * s_dim1]) * mu1i, d__8 = d__7;
        strcp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + *k *
                                                            s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                       s_dim2) * s_dim1]) * mu1i) - mu1i * (float)2. * (s[
                                                                                                                                                          i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k *
                                                                                                                                                                                                       s_dim2) * s_dim1] * mu0i)) * (s[i__ + (j + *k *
                                                                                                                                                                                                                                              s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                    s_dim1] * mu0i + ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                      s_dim2) * s_dim1]) * mu1i) - (s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                             s_dim2) * s_dim1] * (float)2. * mu0i * (((float)1. -
                                                                                                                                                                                                                                                                                                                                                                                                                                      s[i__ + (j + *k * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                i__ + (j + *k * s_dim2) * s_dim1]) * mu1i) - ((float)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1. - s[i__ + (j + *k * s_dim2) * s_dim1]) * (float)
                                                                                                                                                                                                                                                                                                                                                                                    -2. * mu1i * (s[i__ + (j + *k * s_dim2) * s_dim1] * s[
                                                                                                                                                                                                                                                                                                                                                                                                    i__ + (j + *k * s_dim2) * s_dim1] * mu0i)) * (float)
                 2. * (s[i__ + (j + *k * s_dim2) * s_dim1] * (float)2.
                       * mu0i + ((float)1. - s[i__ + (j + *k * s_dim2) *
                                               s_dim1]) * (float)-2. * mu1i)) / (d__3 * (d__2 * d__2)
                                                                                 ) * w[i__ + (j + (*k + 1) * w_dim2) * w_dim1] + ((
                                                                                                                                   mu0i * (float)2. * (((float)1. - s[i__ + (j + *k *
                                                                                                                                                                             s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + *k *
                                                                                                                                                                                                                        s_dim2) * s_dim1]) * mu1i) * (((float)1. - s[i__ + (j
                                                                                                                                                                                                                                                                            + *k * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j +
                                                                                                                                                                                                                                                                                                                              *k * s_dim2) * s_dim1]) * mu1i) + mu1i * (float)2. * (
                                                                                                                                                                                                                                                                                                                                                                                    s[i__ + (j + *k * s_dim2) * s_dim1] * s[i__ + (j + *k
                                                                                                                                                                                                                                                                                                                                                                                                                                   * s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + *k *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + *k * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 d__7)) * d__1;
        d__1 = *beta * betaedge[i__ + (j + (*k + 1) * betaedge_dim2) *
                                betaedge_dim1];
/* Computing 3rd power */
        d__2 = s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (*k + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) *
               mu1i, d__3 = d__2;
/* Computing 2nd power */
        d__4 = s[i__ + (j + (*k + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__5 = (float)1. - s[i__ + (j + (*k + 1) * s_dim2) * s_dim1];
/* Computing 2nd power */
        d__6 = s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * (float)2. *
               mu0i * (((float)1. - s[i__ + (j + (*k + 1) * s_dim2)
                                      * s_dim1]) * ((float)1. - s[i__ + (j + (*k + 1) *
                                                                         s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[i__ + (j
                                                                                                                            + (*k + 1) * s_dim2) * s_dim1]) * (float)-2. * mu1i *
               (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * s[i__ + (
                                                                     j + (*k + 1) * s_dim2) * s_dim1] * mu0i);
/* Computing 3rd power */
        d__7 = s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                    + (*k + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                             s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                           1. - s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) *
               mu1i, d__8 = d__7;
        strp = ((mu0i * (float)2. * (((float)1. - s[i__ + (j + (*k +
                                                                1) * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (*
                                                                                                                     k + 1) * s_dim2) * s_dim1]) * mu1i) - mu1i * (float)
                 2. * (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * s[
                         i__ + (j + (*k + 1) * s_dim2) * s_dim1] * mu0i)) * (s[
                                                                               i__ + (j + (*k + 1) * s_dim2) * s_dim1] * s[i__ + (j
                                                                                                                                  + (*k + 1) * s_dim2) * s_dim1] * mu0i + ((float)1. -
                                                                                                                                                                           s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) * ((float)
                                                                                                                                                                                                                         1. - s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]) *
                                                                             mu1i) - (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * (
                                                                                                                                   float)2. * mu0i * (((float)1. - s[i__ + (j + (*k + 1)
                                                                                                                                                                            * s_dim2) * s_dim1]) * ((float)1. - s[i__ + (j + (*k
                                                                                                                                                                                                                              + 1) * s_dim2) * s_dim1]) * mu1i) - ((float)1. - s[
                                                                                                                                                                                                                                                                     i__ + (j + (*k + 1) * s_dim2) * s_dim1]) * (float)-2.
                                                                                      * mu1i * (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] *
                                                                                                s[i__ + (j + (*k + 1) * s_dim2) * s_dim1] * mu0i)) * (
                                                                                                                                                      float)2. * (s[i__ + (j + (*k + 1) * s_dim2) * s_dim1]
                                                                                                                                                                  * (float)2. * mu0i + ((float)1. - s[i__ + (j + (*k +
                                                                                                                                                                                                                  1) * s_dim2) * s_dim1]) * (float)-2. * mu1i)) / (d__3
                                                                                                                                                                                                                                                                   * (d__2 * d__2)) * w[i__ + (j + (*k + 1) * w_dim2) *
                                                                                                                                                                                                                                                                                        w_dim1] + ((mu0i * (float)2. * (((float)1. - s[i__ + (
                                                                                                                                                                                                                                                                                                                                              j + (*k + 1) * s_dim2) * s_dim1]) * ((float)1. - s[
                                                                                                                                                                                                                                                                                                                                                                                     i__ + (j + (*k + 1) * s_dim2) * s_dim1]) * mu1i) * (((
                                                                                                                                                                                                                                                                                                                                                                                                                                           float)1. - s[i__ + (j + (*k + 1) * s_dim2) * s_dim1])
                                                                                                                                                                                                                                                                                                                                                                                                                                         * ((float)1. - s[i__ + (j + (*k + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                          s_dim1]) * mu1i) + mu1i * (float)2. * (s[i__ + (j + (*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               k + 1) * s_dim2) * s_dim1] * s[i__ + (j + (*k + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_dim2) * s_dim1] * mu0i) * (s[i__ + (j + (*k + 1) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           s_dim2) * s_dim1] * s[i__ + (j + (*k + 1) * s_dim2) *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 s_dim1] * mu0i)) * (d__4 * d__4 / mu0i + d__5 * d__5 /
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     mu1i) - d__6 * d__6 * (float)2.) / (d__8 * (d__7 *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 d__7)) * d__1;
        if (strm * strcm <= (float)0. || strcp * strp <= (float)0.)
        {
          dzscr[i__ + (dzscr_dim1 << 1)] = half * dzscr[i__ + (
                                                               dzscr_dim1 << 1)];
        }
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
} /* sslopez_ */

/* Subroutine */ int rpsolv_(wl, wr, alpha, beta, visc0, visc1, jmax, wc, wrp)
doublereal * wl, *wr, *alpha, *beta, *visc0, *visc1;
integer *jmax;
doublereal *wc, *wrp;
{
  /* System generated locals */
  integer i__1;
  doublereal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8;

  /* Local variables */
  doublereal flag__, wbtw;
  integer j, k;
  doublereal delta, fleft, flmin, fcrit, frght, x0, x1, delta0,
    delta1, wrtmp, deldif, ddelta;

/*     we are solving for phase 0 here where indexes will likely be */
/*       0 - water */
/*       1 - air */

/*     this routine takes a vector of data of length jmax consisting of:
 */

/*      wl              left states */
/*      wr              right states */
/*      alpha           total velocity */
/*      beta            gravity */
/*      visc0           viscosity of phase 0 */
/*      visc1           viscosity of phase 1 */
/*      wc              scratch array used by Riemann solver */

/*    the routine returns wrp which is the solution of the Riemann problem
 */

/*     initial guess for newtons method. */

  /* Parameter adjustments */
  --wrp;
  --wc;
  --visc1;
  --visc0;
  --beta;
  --alpha;
  --wr;
  --wl;

  /* Function Body */
  x0 = (float)1.;
  x1 = (float)0.;
  i__1 = *jmax;
  for (j = 1; j <= i__1; ++j)
  {
/* Computing 2nd power */
    d__1 = x0;
/* Computing 2nd power */
    d__3 = x0;
/* Computing 2nd power */
    d__2 = d__3 * d__3 / visc0[j];
    delta0 = -(d__1 * d__1 / visc0[j] * alpha[j] - d__2 * d__2 * beta[j])
             * (visc0[j] / (x0 * (float)2.));
/* Computing 2nd power */
    d__1 = (float)1. - x1;
/* Computing 2nd power */
    d__3 = (float)1. - x1;
/* Computing 2nd power */
    d__2 = d__3 * d__3 / visc1[j];
    delta1 = (d__1 * d__1 / visc1[j] * alpha[j] + d__2 * d__2 * beta[j]) *
             (visc1[j] / (((float)1. - x1) * (float)-2.));
    if (delta0 == delta1)
    {
      deldif = (float)1.;
    }
    else
    {
      deldif = delta1 - delta0;
    }
    if (delta0 * delta1 >= (float)0.)
    {
      wc[j] = (float)2.;
    }
    else
    {
      wc[j] = delta1 / deldif;
    }
/* L20: */
  }

/*     linear interpolation--newtons method. */

  for (k = 1; k <= 5; ++k)
  {
    i__1 = *jmax;
    for (j = 1; j <= i__1; ++j)
    {
/*     delta = */
/*    $        alpha(j)*(amob1(wc(j),visc1(j))/dmob1(wc(j),visc1(j
 * )) */
/*    $                 -amob0(wc(j),visc0(j))/dmob0(wc(j),visc0(j
 * ))) */
/*    $      + beta(j)*(amob1(wc(j),visc1(j))**2/dmob1(wc(j),visc1
 * (j)) */
/*    $                +amob0(wc(j),visc0(j))**2/dmob0(wc(j),visc0
 * (j))) */
/* Computing 2nd power */
      d__1 = (float)1. - wc[j];
/* Computing 2nd power */
      d__2 = (float)1. - wc[j];
/* Computing 2nd power */
      d__3 = wc[j];
/* Computing 2nd power */
      d__4 = wc[j];
      delta = (alpha[j] + beta[j] * (d__1 * d__1 / visc1[j])) * (d__2 *
                                                                 d__2 / visc1[j]) * (visc1[j] / (((float)1. - wc[j]) * (
                                                                                                                        float)-2.)) + (alpha[j] - beta[j] * (d__3 * d__3 / visc0[
                                                                                                                                                               j])) * (d__4 * d__4 / visc0[j]) * (visc0[j] / (wc[j] * (
                                                                                                                                                                                                                       float)2.));
/* Computing 2nd power */
      d__1 = wc[j];
/* Computing 2nd power */
      d__2 = wc[j];
/* Computing 2nd power */
      d__3 = visc0[j] / (wc[j] * (float)2.);
/* Computing 2nd power */
      d__4 = (float)1. - wc[j];
/* Computing 2nd power */
      d__5 = (float)1. - wc[j];
/* Computing 2nd power */
      d__6 = visc1[j] / (((float)1. - wc[j]) * (float)-2.);
/* Computing 2nd power */
      d__7 = wc[j];
/* Computing 2nd power */
      d__8 = (float)1. - wc[j];
      ddelta = d__1 * d__1 / visc0[j] * (2. / visc0[j]) * (alpha[j] -
                                                           beta[j] * (d__2 * d__2 / visc0[j])) * (d__3 * d__3) -
               d__4 * d__4 / visc1[j] * (2. / visc1[j]) * (alpha[j] +
                                                           beta[j] * (d__5 * d__5 / visc1[j])) * (d__6 * d__6) +
               beta[j] * (float)2. * (d__7 * d__7 / visc0[j] + d__8 *
                                      d__8 / visc1[j]);
      if (!(wc[j] <= (float)1. && wc[j] > (float)0.))
      {
        ddelta = (float)1.;
      }
      if (wc[j] <= (float)1.)
      {
        wc[j] -= delta / ddelta;
      }
      else
      {
        wc[j] = (float)2.;
      }
      if (wc[j] <= (float)0. || wc[j] >= (float)1.)
      {
        wc[j] = (float)2.;
      }
/* L30: */
    }
  }

/*     solve riemann problem. */

  i__1 = *jmax;
  for (j = 1; j <= i__1; ++j)
  {
    if (wl[j] < wr[j])
    {
      flag__ = (float)1.;
    }
    else
    {
      flag__ = (float)-1.;
    }
/* Computing 2nd power */
    d__1 = wl[j];
/* Computing 2nd power */
    d__2 = (float)1. - wl[j];
/* Computing 2nd power */
    d__3 = wl[j];
/* Computing 2nd power */
    d__4 = (float)1. - wl[j];
    fleft = d__1 * d__1 / visc0[j] * (alpha[j] + beta[j] * (d__2 * d__2 /
                                                            visc1[j])) / (d__3 * d__3 / visc0[j] + d__4 * d__4 / visc1[j])
            * flag__;
/* Computing 2nd power */
    d__1 = wr[j];
/* Computing 2nd power */
    d__2 = (float)1. - wr[j];
/* Computing 2nd power */
    d__3 = wr[j];
/* Computing 2nd power */
    d__4 = (float)1. - wr[j];
    frght = d__1 * d__1 / visc0[j] * (alpha[j] + beta[j] * (d__2 * d__2 /
                                                            visc1[j])) / (d__3 * d__3 / visc0[j] + d__4 * d__4 / visc1[j])
            * flag__;
/* Computing 2nd power */
    d__1 = wc[j];
/* Computing 2nd power */
    d__2 = (float)1. - wc[j];
/* Computing 2nd power */
    d__3 = wc[j];
/* Computing 2nd power */
    d__4 = (float)1. - wc[j];
    fcrit = d__1 * d__1 / visc0[j] * (alpha[j] + beta[j] * (d__2 * d__2 /
                                                            visc1[j])) / (d__3 * d__3 / visc0[j] + d__4 * d__4 / visc1[j])
            * flag__;
    flmin = pfmin(fleft, frght);
    wbtw = (wl[j] - wc[j]) * (wr[j] - wc[j]);
    if (fleft < frght)
    {
      wrp[j] = wl[j];
    }
    else
    {
      wrp[j] = wr[j];
    }
    if (flmin < fcrit)
    {
      wrtmp = wrp[j];
    }
    else
    {
      wrtmp = wc[j];
    }
    if (!(wbtw >= (float)0.))
    {
      wrp[j] = wrtmp;
    }
/* L40: */
  }
  return 0;
} /* rpsolv_ */

