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
 * File:        NewEndpts.c
 *
 * Written by:  Bill Bosl
 *              Lawrence Livermore National Lab
 *              phone:  (510) 423-2873
 *              e-mail: wjbosl@llnl.gov
 *
 * Functions contained in this file:
 *              NewEndpts ( ... )
 *
 * External functions called:
 *              ratqr ()        This is an Eispack routine written in
 *                               Fortran.
 *
 * Purpose:
 *               This function estimates the eigenvalues and condition
 *              number of a matrix A using a Lanczos-type algorithm.
 *              The key input parameters (alpha, beta) are generated
 *              in the calling routine, which implements a
 *              Chebyshev polynomial preconditioned conjugate gradient
 *              iterative solver. For more information, see:
 *
 *              Ashby, Manteuffel, Saylor, "A Taxonomy for Conjugate
 *              Gradient Methods", SIAM Journel of Numerical Analysis,
 *              Vol. 27, No. 6, 1542-1568, December 1990.
 *
 *****************************************************************************
 *
 *
 *  -----------------------------------------------------------------------------*/

#include "parflow.h"
#include <math.h>

void NewEndpts(
               double *alpha,
               double *beta,
               double *pp,
               int *   size_ptr,
               int     n,
               double *a_ptr,
               double *b_ptr,
               double *cond_ptr,
               double  ereps)
{
  /* Local variables */
  int i, size = *size_ptr;
  double mu1, mu2;
  double ninv, sqrtmep, meps;

  double tau, taun, eta;
  double cshinv;
  double c, d, g;
  double a, b, an, bn, gn;
  double condo, condn;
  double epz;
  double test;
  double cfo, cfn;

  /* Variables used to pass information to ratqr */
  double ep1 = 0;
  double      *diag, *e, *e2;
  int n_ev = 1, idef = 1, ierr;
  int type;
  int         *ind;
  double      *w, *bd;

  (void)pp;

  /* Allocate memory for local variable-size arrays */
  diag = (double*)malloc(size * sizeof(double));
  e = (double*)malloc(size * sizeof(double));
  e2 = (double*)malloc(size * sizeof(double));
  w = (double*)malloc(size * sizeof(double));
  bd = (double*)malloc(size * sizeof(double));
  ind = (int*)malloc(size * sizeof(int));

  /*  Fill diag, e, e2 for input to ratqr */
  diag[0] = (1.0 + beta[0]) / alpha[0];
  for (i = 1; i < size; i++)
  {
    diag[i] = (1.0 + beta[i]) / alpha[i];
    e[i] = -sqrt(beta[i - 1] / (alpha[i - 1] * alpha[i]));
    e2[i] = e[i] * e[i];
  }

  /* Estimate the eigenvalues of C(A)A using Eispack routine */
  type = 0;
  ratqr_(&size, &ep1, diag, e, e2, &n_ev, w, ind, bd, &type, &idef, &ierr);
  mu2 = w[0];
  type = 1;
  ratqr_(&size, &ep1, diag, e, e2, &n_ev, w, ind, bd, &type, &idef, &ierr);
  mu1 = w[0];

  /* Using mu1, mu2 estimate the min and max eigenvalues of A */

  /* Compute some intermediate quantities */
  a = *a_ptr;
  b = *b_ptr;
  c = 0.5 * (b - a);
  d = c + a;
  ninv = 1.0 / (double)n;

  /* Estimate machine epsilon */
  meps = 1.0;
  while (meps / 2 > 0)
  {
    meps = meps / 2;
  }
  sqrtmep = sqrt(meps);

  /* Check for small c relative to d */
  if (c <= d * sqrtmep)
  {
    /* tau = d^n */
    tau = pow(d, n);
    if (mu1 > 1.0)
      mu1 = 1.0;
    if (mu2 <= 1.0)
      mu2 = 1.0;
    an = d - pow(((1 - mu1) * tau), ninv);
    bn = d + pow(((mu2 - 1.0) * tau), ninv);
  }

  /* Otherwise, determine new endpoints */
  else
  {
    /* Compute polynomial deviation from 1 over [c,d] */
    g = d / c;
    tau = cosh(n * log(g + sqrt(g * g - 1.0)));

    /* Determine left endpoint */
    eta = (1.0 - mu1) * tau;
    if (eta > 1.0)
    {
      cshinv = log(eta + sqrt(eta * eta - 1.0));
      an = d - c * cosh(cshinv * ninv);
    }
    else            /* Use old endpoint */
    {
      an = a;
      mu1 = (tau - 1.0) / tau;
    }

    /* Determine right endpoint */
    eta = (mu2 - 1.0) * tau;
    if (eta > 1.0)
    {
      cshinv = log(eta + sqrt(eta * eta - 1.0));
      bn = d + c * cosh(cshinv * ninv);
    }
    else            /* Use old endpoint */
    {
      bn = b;
      mu2 = (tau + 1.0) / tau;
    }
  }

  /* Check for no change in the endpoints */
  if ((an == a) && (bn == b))
  {
    free(diag);
    free(e);
    free(e2);
    free(w);
    free(bd);
    free(ind);

    /* Revised estimate of the old condition number */
    *cond_ptr = mu2 / mu1;
    return;
  }
  else
  {
    gn = (bn + an) / (bn - an);
    taun = cosh(n * log(gn + sqrt(gn * gn - 1.0)));

    /* Revised estimate of the old condition number */
    condo = mu2 / mu1;

    /* Estimate of the new condition number */
    condn = (taun + 1.0) / (taun - 1.0);
    cfo = (sqrt(condo) - 1.0) / (sqrt(condo) + 1.0);
    cfn = (sqrt(condn) - 1.0) / (sqrt(condn) + 1.0);
    epz = pfmax(ereps, sqrtmep);
    test = log(epz) * (1.0 / log(cfo) - 1.0 / log(cfn));

    /* Resume iteration with current preconditioning matrix */
    if (test < 1.0)
    {
      *a_ptr = a;
      *b_ptr = b;
      *cond_ptr = condo;
    }

    else
    {
      /* Restart iteration with new endpoints */
      *a_ptr = an;
      *b_ptr = bn;
      mu1 = (taun - 1.0) / taun;
      mu2 = (taun + 1.0) / taun;
      *size_ptr = 0;
      *cond_ptr = condn;
    }
  }    /* End of "if ((an == a) && (bn == b)) ... else ..." clause.  */


  free(diag);
  free(e);
  free(e2);
  free(w);
  free(bd);
  free(ind);
}  /* End of function NewEndpts()  */


