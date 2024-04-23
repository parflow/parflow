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

/***************************************************************************
*
*
***************************************************************************/

#include "parflow.h"
#include "problem_eval.h"


/*-------------------------------------------------------------------------
 * compute_phase_maximum
 *-------------------------------------------------------------------------*/

double  ComputePhaseMaximum(
                            double phase_u_max,
                            double dx,
                            double phase_v_max,
                            double dy,
                            double phase_w_max,
                            double dz)
{
  double maximum, tmp;

  /* X direction */
  maximum = phase_u_max / dx;

  /* Y direction */
  tmp = phase_v_max / dy;
  if (tmp > maximum)
    maximum = tmp;

  /* Z direction */
  tmp = phase_w_max / dz;
  if (tmp > maximum)
    maximum = tmp;

  return maximum;
}

/*-------------------------------------------------------------------------
 * compute_total_maximum
 *-------------------------------------------------------------------------*/

double  ComputeTotalMaximum(
                            Problem *   problem,
                            EvalStruct *eval_struct,
                            double      s_lower,
                            double      s_upper,
                            double      total_u_max,
                            double      dx,
                            double      total_v_max,
                            double      dy,
                            double      total_w_max,
                            double      beta_max,
                            double      dz)
{
  PFModule     *phase_density = ProblemPhaseDensity(problem);

  double a, b, den0, den1, dtmp, g;

  double f_prime_max, h_prime_max;
  double point, value, tmp, maximum;
  int i;

  a = 1.0 / ProblemPhaseViscosity(problem, 0);
  b = 1.0 / ProblemPhaseViscosity(problem, 1);

  /* CSW  Hard-coded in an assumption here for constant density.
   *      Use dtmp as dummy argument here. */

  // Solution using a typedef: Define a pointer to a function which is taking
  // two floats and returns a float
  // typedef float (*pt2Func)(float, float);

  PFModuleInvokeType(PhaseDensityInvoke, phase_density, (0, NULL, NULL, &dtmp, &den0, CALCFCN));
  PFModuleInvokeType(PhaseDensityInvoke, phase_density, (1, NULL, NULL, &dtmp, &den1, CALCFCN));

  g = -ProblemGravity(problem);

  /**************************************************************
  *                                                            *
  * Find the maximum value of f' over the interval, given that *
  *      the real roots of f'' have been previously found      *
  *                                                            *
  **************************************************************/
  f_prime_max = fabs(Fprime_OF_S(s_lower, a, b));
  for (i = 0; i < EvalNumFPoints(eval_struct); i++)
  {
    point = EvalFPoint(eval_struct, i);
    value = EvalFValue(eval_struct, i);
    if ((point > s_lower) && (point < s_upper))
    {
      tmp = fabs(value);
      if (tmp > f_prime_max)
        f_prime_max = tmp;
    }
  }
  tmp = fabs(Fprime_OF_S(s_upper, a, b));
  if (tmp > f_prime_max)
    f_prime_max = tmp;

  /**************************************************************
  *                                                            *
  * Find the maximum value of h' over the interval, given that *
  *      the real roots of h'' have been previously found      *
  *                                                            *
  **************************************************************/
  h_prime_max = fabs(Hprime_OF_S(s_lower, a, b));
  for (i = 0; i < EvalNumHPoints(eval_struct); i++)
  {
    point = EvalHPoint(eval_struct, i);
    value = EvalHValue(eval_struct, i);
    if ((point > s_lower) && (point < s_upper))
    {
      tmp = fabs(value);
      if (tmp > h_prime_max)
        h_prime_max = tmp;
    }
  }
  tmp = fabs(Hprime_OF_S(s_upper, a, b));
  if (tmp > h_prime_max)
    h_prime_max = tmp;

  /**************************************************************
  *                                                            *
  *       Find the maximum for use in the dt computation       *
  *                                                            *
  **************************************************************/
  /* X direction */
  maximum = f_prime_max * (total_u_max / dx);

  /* Y direction */
  tmp = f_prime_max * (total_v_max / dy);
  if (tmp > maximum)
    maximum = tmp;

  /* Z direction */
  tmp = f_prime_max * (total_w_max / dz)
        + h_prime_max * fabs(g * (den0 - den1)) * (beta_max / dz);
  if (tmp > maximum)
    maximum = tmp;

  return maximum;
}
