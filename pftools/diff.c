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
**********************************************************************EHEADER*/
/*****************************************************************************
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*-----------------------------------------------------------------------------
* $Revision: 1.16 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "diff.h"

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef max
#define max(a, b)  ((a) > (b) ? (a) : (b))
#endif


/*-----------------------------------------------------------------------
 * SigDiff:
 *   If (m >= 0), this routine prints the absolute differences
 *   of databox elements differing in more than m significant digits.
 *   If (m < 0), this routine prints the minimum number of agreeing
 *   significant digits.
 *
 *   We use the following to determine if two numbers, x1 and x2, have
 *   at least m significant digits:
 *
 *     xM = max(|x1|, |x2|)
 *
 *     if ( xM > a0 )
 *        "x1 and x2 have m significant digits"
 *     else
 *     {
 *       y = | x1 - x2 | / xM
 *
 *       if (y <= 0.5*10^{-m})
 *          "x1 and x2 have m significant digits"
 *     }
 *
 *   The number a0 is "absolute zero".  If xM is less than 0, it is
 *   treated as if it were exactly 0.
 *-----------------------------------------------------------------------*/

void      SigDiff(
                  Databox *v1,
                  Databox *v2,
                  int      m,
                  double   absolute_zero,
                  FILE *   fp)
{
  double         *v1_p, *v2_p;

  double sig_dig_rhs;
  double adiff, amax, sdiff;
  double max_adiff = 0.0, max_sdiff;

  int i, j, k;
  int mi = 0, mj = 0, mk = 0;
  int nx, ny, nz;

  int sig_digs;
  int m_sig_digs, m_sig_digs_everywhere;

  /*-----------------------------------------------------------------------
   * check that dimensions are the same
   *-----------------------------------------------------------------------*/

  nx = DataboxNx(v1);
  ny = DataboxNy(v1);
  nz = DataboxNz(v1);

  /*-----------------------------------------------------------------------
   * diff the values and print the results
   *-----------------------------------------------------------------------*/

  if (m >= 0)
    sig_dig_rhs = 0.5 / pow(10.0, ((double)m));
  else
    sig_dig_rhs = 0.0;

  v1_p = DataboxCoeffs(v1);
  v2_p = DataboxCoeffs(v2);

  m_sig_digs_everywhere = TRUE;
  max_sdiff = 0.0;
  for (k = 0; k < nz; k++)
  {
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        adiff = fabs((*v1_p) - (*v2_p));
        amax = max(fabs(*v1_p), fabs(*v2_p));

        m_sig_digs = TRUE;
        if (amax > absolute_zero)
        {
          sdiff = adiff / amax;
          if (sdiff > sig_dig_rhs)
            m_sig_digs = FALSE;
        }

        if (!m_sig_digs)
        {
          if (m >= 0)
          {
            fprintf(fp, "(%d,%d,%d) : %e, %e, %e\n",
                    i, j, k, adiff, *v1_p, *v2_p);
          }

          if (sdiff > max_sdiff)
          {
            max_sdiff = sdiff;
            max_adiff = adiff;
            mi = i;
            mj = j;
            mk = k;
          }

          m_sig_digs_everywhere = FALSE;
        }

        v1_p++;
        v2_p++;
      }
    }
  }

  if (!m_sig_digs_everywhere)
  {
    /* compute min number of sig digs */
    sig_digs = 0;
    sdiff = max_sdiff;
    while (sdiff <= 0.5e-01)
    {
      sdiff *= 10.0;
      sig_digs++;
    }

    fprintf(fp, "Minimum significant digits at (% 3d, %3d, %3d) = %2d\n",
            mi, mj, mk, sig_digs);
    fprintf(fp, "Maximum absolute difference = %e\n", max_adiff);
  }
}


void         MSigDiff(
                      Tcl_Interp *interp,
                      Databox *   v1,
                      Databox *   v2,
                      int         m,
                      double      absolute_zero,
                      Tcl_Obj *   result)
{
  double         *v1_p, *v2_p;

  double sig_dig_rhs;
  double adiff, amax, sdiff;
  double max_adiff = 0.0, max_sdiff;

  int i, j, k;
  int mi = 0, mj = 0, mk = 0;
  int nx, ny, nz;

  int sig_digs;
  int m_sig_digs, m_sig_digs_everywhere;

  /*-----------------------------------------------------------------------
   * check that dimensions are the same
   *-----------------------------------------------------------------------*/
  nx = DataboxNx(v1);
  ny = DataboxNy(v1);
  nz = DataboxNz(v1);

  /*-----------------------------------------------------------------------
   * diff the values and print the results
   *-----------------------------------------------------------------------*/
  if (m >= 0)
    sig_dig_rhs = 0.5 / pow(10.0, ((double)m));
  else
    sig_dig_rhs = 0.0;

  v1_p = DataboxCoeffs(v1);
  v2_p = DataboxCoeffs(v2);

  m_sig_digs_everywhere = TRUE;
  max_sdiff = 0.0;
  for (k = 0; k < nz; k++)
  {
    for (j = 0; j < ny; j++)
    {
      for (i = 0; i < nx; i++)
      {
        adiff = fabs((*v1_p) - (*v2_p));
        amax = max(fabs(*v1_p), fabs(*v2_p));

        if (max_adiff < adiff)
          max_adiff = adiff;

        m_sig_digs = TRUE;
        if (amax > absolute_zero)
        {
          sdiff = adiff / amax;
          if (sdiff > sig_dig_rhs)
            m_sig_digs = FALSE;
        }

        if (!m_sig_digs)
        {
          if (sdiff > max_sdiff)
          {
            max_sdiff = sdiff;

            mi = i;
            mj = j;
            mk = k;
          }

          m_sig_digs_everywhere = FALSE;
        }

        v1_p++;
        v2_p++;
      }
    }
  }

  if (!m_sig_digs_everywhere)
  {
    Tcl_Obj     *double_obj;
    Tcl_Obj     *int_obj;
    Tcl_Obj     *list_obj;

    /* compute min number of sig digs */
    sig_digs = 0;
    sdiff = max_sdiff;
    while (sdiff <= 0.5e-01)
    {
      sdiff *= 10.0;
      sig_digs++;
    }

    /* Create a Tcl list of the form: {{mi mj mk sig_digs} max_adiff} */
    /* and append it to the result string.                            */

    list_obj = Tcl_NewListObj(0, NULL);

    int_obj = Tcl_NewIntObj(mi);
    Tcl_ListObjAppendElement(interp, list_obj, int_obj);

    int_obj = Tcl_NewIntObj(mj);
    Tcl_ListObjAppendElement(interp, list_obj, int_obj);

    int_obj = Tcl_NewIntObj(mk);
    Tcl_ListObjAppendElement(interp, list_obj, int_obj);

    int_obj = Tcl_NewIntObj(sig_digs);
    Tcl_ListObjAppendElement(interp, list_obj, int_obj);

    Tcl_ListObjAppendElement(interp, result, list_obj);

    double_obj = Tcl_NewDoubleObj(max_adiff);
    Tcl_ListObjAppendElement(interp, result, double_obj);
  }
}


double    DiffElt(
                  Databox *v1,
                  Databox *v2,
                  int      i,
                  int      j,
                  int      k,
                  int      m,
                  double   absolute_zero)
{
  double         *v1_p, *v2_p;

  double sig_dig_rhs;
  double adiff, amax, sdiff;
  int m_sig_digs;

  /*-----------------------------------------------------------------------
   * diff the values and print the results
   *-----------------------------------------------------------------------*/

  sig_dig_rhs = 0.5 / pow(10.0, ((double)m));

  v1_p = DataboxCoeff(v1, i, j, k);
  v2_p = DataboxCoeff(v2, i, j, k);

  adiff = fabs((*v1_p) - (*v2_p));
  amax = max(fabs(*v1_p), fabs(*v2_p));

  m_sig_digs = TRUE;
  if (amax > absolute_zero)
  {
    sdiff = adiff / amax;
    if (sdiff > sig_dig_rhs)
      m_sig_digs = FALSE;
  }

  if (m_sig_digs)
  {
    return -1;
  }

  return adiff;
}
