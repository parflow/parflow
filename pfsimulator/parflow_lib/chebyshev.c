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
* Chebyshev iteration.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  /* InitInstanceXtra arguments */
  Grid     *grid;
  Matrix   *A;
  double    *temp_data;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Chebyshev:
 *   Solves A x = b with interval [ia, ib].
 *   RDF Assumes initial guess of 0.
 *--------------------------------------------------------------------------*/

void     Chebyshev(
                   Vector *x,
                   Vector *b,
                   double  tol,
                   int     zero,
                   double  ia,
                   double  ib,
                   int     num_iter)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Matrix    *A = (instance_xtra->A);

  Vector *r = NULL;
  Vector *del = NULL;

  double d, c22;
  double alpha, beta;

  int i = 0;

  (void)tol;
  (void)zero;

  /*-----------------------------------------------------------------------
   * Begin timing
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  r = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  del = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Start Chebyshev
   *-----------------------------------------------------------------------*/

  /* x = b */
  Copy(b, x);

  if ((i + 1) > num_iter)
  {
    return;
  }

  i++;

  d = 0.5 * (ib + ia);

  /* x = (1/d)*x */
  Scale((alpha = (1.0 / d)), x);

  if ((i + 1) > num_iter)
  {
    IncFLOPCount(3);
    return;
  }

  c22 = 0.25 * (ib - ia);
  c22 *= c22;

  /* alpha = 2 / d */
  alpha *= 2.0;

  /* del = x */
  Copy(x, del);

  while ((i + 1) <= num_iter)
  {
    i++;

    alpha = 1.0 / (d - c22 * alpha);

    beta = d * alpha - 1.0;

    /* r = b - A*x */
    Matvec(-1.0, A, x, 0.0, r);
    Axpy(1.0, b, r);

    /* del = alpha*r + beta*del */
    Scale(beta, del);
    Axpy(alpha, r, del);

    /* x = x + del */
    Axpy(1.0, del, x);
  }

  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(del);
  FreeVector(r);

  /*-----------------------------------------------------------------------
   * end Chebyshev
   *-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------
   * end timing
   *-----------------------------------------------------------------------*/

  IncFLOPCount(i * 5 + 7);
}


/*--------------------------------------------------------------------------
 * ChebyshevInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule     *ChebyshevInitInstanceXtra(
                                        Problem *    problem,
                                        Grid *       grid,
                                        ProblemData *problem_data,
                                        Matrix *     A,
                                        double *     temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  (void)problem;
  (void)problem_data;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    /* set new data */
    (instance_xtra->grid) = grid;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `A'
   *-----------------------------------------------------------------------*/

  if (A != NULL)
    (instance_xtra->A) = A;

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * ChebyshevFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  ChebyshevFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);


  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * ChebyshevNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *ChebyshevNewPublicXtra(char *name)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  (void)name;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * ChebyshevFreePublicXtra
 *--------------------------------------------------------------------------*/

void  ChebyshevFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * ChebyshevSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ChebyshevSizeOfTempData()
{
  int sz = 0;

  return sz;
}
