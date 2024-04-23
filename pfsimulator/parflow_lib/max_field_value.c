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
/****************************************************************************
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * MaxFieldValue
 *--------------------------------------------------------------------------*/

double  MaxFieldValue(
                      Vector *field,
                      Vector *phi,
                      int     dir)
{
  Grid         *grid;
  Subgrid      *subgrid;

  Subvector    *f_sub;
  Subvector    *p_sub;

  double       *fp;
  double       *plp, *prp;

  double max_field_value, tmp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_f, ny_f, nz_f;
  int nx_p, ny_p, nz_p;
  int stx = 0, sty = 0, stz = 0;

  int i_s, i, j, k, fi, pi;

  amps_Invoice result_invoice;

  max_field_value = 0.0;

  switch (dir)
  {
    case 0:
      stx = 1;
      sty = 0;
      stz = 0;
      break;

    case 1:
      stx = 0;
      sty = 1;
      stz = 0;
      break;

    case 2:
      stx = 0;
      sty = 0;
      stz = 1;
      break;
  }

  max_field_value = 0.0;

  grid = VectorGrid(field);
  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    f_sub = VectorSubvector(field, i_s);
    p_sub = VectorSubvector(phi, i_s);

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    fp = SubvectorElt(f_sub, ix, iy, iz);
    plp = SubvectorElt(p_sub, ix - stx, iy - sty, iz - stz);
    prp = SubvectorElt(p_sub, ix, iy, iz);

    fi = 0;
    pi = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              fi, nx_f, ny_f, nz_f, 1, 1, 1,
              pi, nx_p, ny_p, nz_p, 1, 1, 1,
    {
      tmp = fabs(fp[fi]) / pfmax(plp[pi], prp[pi]);
      if (tmp > max_field_value)
        max_field_value = tmp;
    });
  }

  result_invoice = amps_NewInvoice("%d", &max_field_value);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);

#if 1
  /****************************************************/
  /*                                                  */
  /* Print out some diagnostics on the computed value */
  /*                                                  */
  /****************************************************/

  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Maximum Field Value = %e\n", max_field_value);
  }
#endif

  return max_field_value;
}


/*--------------------------------------------------------------------------
 * MaxPhaseFieldValue
 *--------------------------------------------------------------------------*/

double  MaxPhaseFieldValue(
                           Vector *x_velocity,
                           Vector *y_velocity,
                           Vector *z_velocity,
                           Vector *phi)
{
  Grid         *grid;
  Subgrid      *subgrid;

  Subvector    *v_sub;
  Subvector    *p_sub;

  Vector       *velocity = NULL;

  double       *vp;
  double       *plp, *prp;

  double max_field_value, psi_max, ds = 0.0;
  double max_xdir_value, max_ydir_value, max_zdir_value;
  double tmp_max;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_p, ny_p, nz_p;
  int stx = 0, sty = 0, stz = 0;

  int i_s, dir, i, j, k, vi, pi;

  amps_Invoice result_invoice;

  max_xdir_value = 0.0;
  max_ydir_value = 0.0;
  max_zdir_value = 0.0;

  for (dir = 0; dir < 3; dir++)
  {
    switch (dir)
    {
      case 0:
        velocity = x_velocity;
        stx = 1;
        sty = 0;
        stz = 0;
        break;

      case 1:
        velocity = y_velocity;
        stx = 0;
        sty = 1;
        stz = 0;
        break;

      case 2:
        velocity = z_velocity;
        stx = 0;
        sty = 0;
        stz = 1;
        break;
    }

    grid = VectorGrid(velocity);
    ForSubgridI(i_s, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, i_s);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      switch (dir)
      {
        case 0:
          ds = SubgridDX(GridSubgrid(grid, i_s));
          break;

        case 1:
          ds = SubgridDY(GridSubgrid(grid, i_s));
          break;

        case 2:
          ds = SubgridDZ(GridSubgrid(grid, i_s));
          break;
      }

      v_sub = VectorSubvector(velocity, i_s);
      p_sub = VectorSubvector(phi, i_s);

      nx_v = SubvectorNX(v_sub);
      ny_v = SubvectorNY(v_sub);
      nz_v = SubvectorNZ(v_sub);

      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);

      vp = SubvectorElt(v_sub, ix, iy, iz);
      plp = SubvectorElt(p_sub, ix - stx, iy - sty, iz - stz);
      prp = SubvectorElt(p_sub, ix, iy, iz);

      vi = 0;
      pi = 0;
      BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
                vi, nx_v, ny_v, nz_v, 1, 1, 1,
                pi, nx_p, ny_p, nz_p, 1, 1, 1,
      {
        psi_max = pfmax(fabs(plp[pi]), fabs(prp[pi])) * ds;
        if (psi_max != 0.0)
        {
          tmp_max = fabs(vp[vi]) / psi_max;
          switch (dir)
          {
            case 0:
              {
                if (tmp_max > max_xdir_value)
                {
                  max_xdir_value = tmp_max;
                }
                break;
              }

            case 1:
              {
                if (tmp_max > max_ydir_value)
                {
                  max_ydir_value = tmp_max;
                }
                break;
              }

            case 2:
              {
                if (tmp_max > max_zdir_value)
                {
                  max_zdir_value = tmp_max;
                }
                break;
              }
          }
        }
      });
    }
  }

  result_invoice = amps_NewInvoice("%d%d%d",
                                   &max_xdir_value,
                                   &max_ydir_value,
                                   &max_zdir_value);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);

  max_field_value = pfmax(max_xdir_value, pfmax(max_ydir_value, max_zdir_value));

#if 1
  /*****************************************************/
  /*                                                   */
  /* Print out some diagnostics on the computed values */
  /*                                                   */
  /*****************************************************/

  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Courant Numbers : [%e , %e, %e]\n",
                max_xdir_value,
                max_ydir_value,
                max_zdir_value);
    amps_Printf("Maximum Field Value = %e\n", max_field_value);
  }
#endif

  return max_field_value;
}


/*--------------------------------------------------------------------------
 * MaxTotalFieldValue
 *--------------------------------------------------------------------------*/

double  MaxTotalFieldValue(
                           Problem *   problem,
                           EvalStruct *eval_struct,
                           Vector *    x_velocity,
                           Vector *    y_velocity,
                           Vector *    z_velocity,
                           Vector *    saturation,
                           Vector *    beta,
                           Vector *    phi)
{
  Grid         *grid;
  Subgrid      *subgrid;

  Subvector    *v_sub;
  Subvector    *s_sub;
  Subvector    *b_sub = NULL;
  Subvector    *p_sub;

  Vector       *velocity = NULL;

  PFModule     *phase_density = ProblemPhaseDensity(problem);

  double       *vp, *bp = NULL;
  double       *slp, *srp, *plp, *prp;

  double point, value;
  double s_lower, s_upper;
  double f_prime_max, h_prime_max = 0.0;
  double max_field_value, psi_max, ds = 0.0;
  double max_xdir_value, max_ydir_value, max_zdir_value,
    max_total_value, max_gravity_value;
  double tmp, tmp_max, tmp_total = 0.0, tmp_gravity = 0.0;

  double a, b, den0, den1, dtmp, g, constant;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_s, ny_s, nz_s;
  int nx_p, ny_p, nz_p;
  int stx = 0, sty = 0, stz = 0;

  int dir, i_s, i, j, k, pnt, vi, si, pi;

  amps_Invoice result_invoice;

  a = 1.0 / ProblemPhaseViscosity(problem, 0);
  b = 1.0 / ProblemPhaseViscosity(problem, 1);

  /* CSW  Hard-coded in an assumption here for constant density.
   *      Use dtmp as dummy argument.  */
  PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, NULL, NULL, &dtmp, &den0, CALCFCN));
  PFModuleInvokeType(PhaseDensityInvoke, phase_density, (1, NULL, NULL, &dtmp, &den1, CALCFCN));

  g = -ProblemGravity(problem);

  constant = fabs(g * (den0 - den1));

  max_xdir_value = 0.0;
  max_ydir_value = 0.0;
  max_zdir_value = 0.0;
  max_total_value = 0.0;
  max_gravity_value = 0.0;

  for (dir = 0; dir < 3; dir++)
  {
    switch (dir)
    {
      case 0:
        velocity = x_velocity;
        stx = 1;
        sty = 0;
        stz = 0;
        break;

      case 1:
        velocity = y_velocity;
        stx = 0;
        sty = 1;
        stz = 0;
        break;

      case 2:
        velocity = z_velocity;
        stx = 0;
        sty = 0;
        stz = 1;
        break;
    }

    grid = VectorGrid(velocity);
    ForSubgridI(i_s, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, i_s);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      switch (dir)
      {
        case 0:
          ds = SubgridDX(GridSubgrid(grid, i_s));
          break;

        case 1:
          ds = SubgridDY(GridSubgrid(grid, i_s));
          break;

        case 2:
          ds = SubgridDZ(GridSubgrid(grid, i_s));
          break;
      }

      v_sub = VectorSubvector(velocity, i_s);
      s_sub = VectorSubvector(saturation, i_s);
      if (dir == 2)
      {
        b_sub = VectorSubvector(beta, i_s);
      }
      p_sub = VectorSubvector(phi, i_s);

      nx_v = SubvectorNX(v_sub);
      ny_v = SubvectorNY(v_sub);
      nz_v = SubvectorNZ(v_sub);

      nx_s = SubvectorNX(s_sub);
      ny_s = SubvectorNY(s_sub);
      nz_s = SubvectorNZ(s_sub);

      nx_p = SubvectorNX(p_sub);
      ny_p = SubvectorNY(p_sub);
      nz_p = SubvectorNZ(p_sub);

      vp = SubvectorElt(v_sub, ix, iy, iz);
      slp = SubvectorElt(s_sub, ix - stx, iy - sty, iz - stz);
      srp = SubvectorElt(s_sub, ix, iy, iz);
      if (dir == 2)
      {
        bp = SubvectorElt(b_sub, ix, iy, iz);
      }
      plp = SubvectorElt(p_sub, ix - stx, iy - sty, iz - stz);
      prp = SubvectorElt(p_sub, ix, iy, iz);

      vi = 0;
      si = 0;
      pi = 0;

      BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
                vi, nx_v, ny_v, nz_v, 1, 1, 1,
                si, nx_s, ny_s, nz_s, 1, 1, 1,
                pi, nx_p, ny_p, nz_p, 1, 1, 1,
      {
        if (slp[si] <= srp[si])
        {
          s_lower = slp[si];
          s_upper = srp[si];
        }
        else
        {
          s_lower = srp[si];
          s_upper = slp[si];
        }

        /**************************************************************
        *                                                            *
        * Find the maximum value of f' over the interval, given that *
        *      the real roots of f'' have been previously found      *
        *                                                            *
        **************************************************************/

        f_prime_max = fabs(Fprime_OF_S(s_lower, a, b));
        for (pnt = 0; pnt < EvalNumFPoints(eval_struct); pnt++)
        {
          point = EvalFPoint(eval_struct, pnt);
          value = EvalFValue(eval_struct, pnt);
          if ((s_lower < point) && (point < s_upper))
          {
            tmp = fabs(value);
            if (tmp > f_prime_max)
            {
              f_prime_max = tmp;
            }
          }
        }
        tmp = fabs(Fprime_OF_S(s_upper, a, b));
        if (tmp > f_prime_max)
        {
          f_prime_max = tmp;
        }

        if (dir == 2)
        {
          /**************************************************************
          *                                                            *
          * Find the maximum value of h' over the interval, given that *
          *      the real roots of h'' have been previously found      *
          *                                                            *
          **************************************************************/

          h_prime_max = fabs(Hprime_OF_S(s_lower, a, b));
          for (pnt = 0; pnt < EvalNumHPoints(eval_struct); pnt++)
          {
            point = EvalHPoint(eval_struct, pnt);
            value = EvalHValue(eval_struct, pnt);
            if ((s_lower < point) && (point < s_upper))
            {
              tmp = fabs(value);
              if (tmp > h_prime_max)
              {
                h_prime_max = tmp;
              }
            }
          }
          tmp = fabs(Hprime_OF_S(s_upper, a, b));
          if (tmp > h_prime_max)
          {
            h_prime_max = tmp;
          }
        }

        psi_max = pfmax(fabs(plp[pi]), fabs(prp[pi])) * ds;
        if (psi_max != 0.0)
        {
          if (dir == 2)
          {
            tmp_total = f_prime_max * fabs(vp[vi]) / psi_max;
            tmp_gravity = h_prime_max * constant * fabs(bp[vi]) / psi_max;
            tmp_max = (f_prime_max * fabs(vp[vi])
                       + h_prime_max * constant * fabs(bp[vi]))
                      / psi_max;
          }
          else
          {
            tmp_max = (f_prime_max * fabs(vp[vi])) / psi_max;
          }

          switch (dir)
          {
            case 0:
              {
                if (tmp_max > max_xdir_value)
                {
                  max_xdir_value = tmp_max;
                }
                break;
              }

            case 1:
              {
                if (tmp_max > max_ydir_value)
                {
                  max_ydir_value = tmp_max;
                }
                break;
              }

            case 2:
              {
                if (tmp_max > max_zdir_value)
                {
                  max_zdir_value = tmp_max;
                }
                if (tmp_total > max_total_value)
                {
                  max_total_value = tmp_total;
                }
                if (tmp_gravity > max_gravity_value)
                {
                  max_gravity_value = tmp_gravity;
                }
                break;
              }
          }
        }
      });
    }
  }

  result_invoice = amps_NewInvoice("%d%d%d%d%d",
                                   &max_xdir_value,
                                   &max_ydir_value,
                                   &max_zdir_value,
                                   &max_total_value,
                                   &max_gravity_value);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);

  max_field_value = pfmax(max_xdir_value, pfmax(max_ydir_value, max_zdir_value));

#if 1
  /*****************************************************/
  /*                                                   */
  /* Print out some diagnostics on the computed values */
  /*                                                   */
  /*****************************************************/

  if (!amps_Rank(amps_CommWorld))
  {
    amps_Printf("Courant Numbers : [%e , %e, %e : (%e, %e)]\n",
                max_xdir_value,
                max_ydir_value,
                max_zdir_value,
                max_total_value,
                max_gravity_value);
    amps_Printf("Maximum Field Value = %e\n", max_field_value);
  }
#endif

  return max_field_value;
}
