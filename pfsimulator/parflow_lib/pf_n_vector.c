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

#include "parflow.h"
#include "llnltyps.h"

#define ZERO 0.0
#define ONE  1.0

#if defined (PARFLOW_HAVE_SUNDIALS)
#ifdef __cplusplus
extern "C" {
#endif

/* Methods required for SUNDIALS N vectors */

/* Create new vector */
N_Vector PF_NVNewEmpty(SUNContext sunctx)
{
  N_Vector v = NULL;
  PF_N_Vector_Content content = NULL;

  /* Allocate memory for new N_Vector */
  v = N_VNewEmpty(sunctx);

  /* set function pointers for ops */
  v->ops->nvclone = PF_NVClone;
  v->ops->nvdestroy = PF_NVDestroy;
  v->ops->nvgetlength = PF_NVGetLength;

  v->ops->nvlinearsum = PFVLinearSumFcn;
  v->ops->nvconst = PFVConstInitFcn;
  v->ops->nvprod = PFVProdFcn;
  v->ops->nvdiv = PFVDivFcn;
  v->ops->nvscale = PFVScaleFcn;
  v->ops->nvabs = PFVAbsFcn;
  v->ops->nvinv = PFVInvFcn;
  v->ops->nvaddconst = PFVAddConstFcn;
  v->ops->nvdotprod = PFVDotProdFcn;
  v->ops->nvmaxnorm = PFVMaxNormFcn;
  v->ops->nvwrmsnorm = PFVWrmsNormFcn;
  v->ops->nvwrmsnormmask = NULL;
  v->ops->nvmin = PFVMinFcn;
  v->ops->nvwl2norm = PFVWL2NormFcn;
  v->ops->nvl1norm = PFVL1NormFcn;
  v->ops->nvcompare = PFVCompareFcn;
  v->ops->nvinvtest = PFVInvTestFcn;
  v->ops->nvconstrmask = NULL;
  v->ops->nvminquotient = NULL;

  /* Allocate memory for vector content (data) */
  content = (PF_N_Vector_Content)malloc(sizeof *content);
  /* set content data to NULL (empty) */
  content->data = NULL;
  content->owns_data = false;

  /* Attach pointers to complete new N_Vector */
  v->content = content;

  return(v);
}

/* Create new vector */
N_Vector PF_NVNew(SUNContext sunctx, Grid *grid, int num_ghost)
{
  N_Vector v = NULL;
  Vector *data = NULL;
  int nc = 1;

  /* create empty vector object */
  v = PF_NVNewEmpty(sunctx);

  /* create vector data and attach to content */
  /* Currently PF only supports nc=1 coefficient */
  data = NewVectorType(grid, nc, num_ghost, vector_cell_centered);
  N_VectorData(v) = data;
  N_VectorOwnsData(v) = true;

  return(v);
}

/* Create new vector from Parflow Vector */
N_Vector PF_NVNewFromVector(SUNContext sunctx, Vector *data)
{
  N_Vector v = NULL;

  /* create empty vector object */
  v = PF_NVNewEmpty(sunctx);

  /* Attach Parflow Vector to content */
  N_VectorData(v) = data;
  N_VectorOwnsData(v) = false;

  return(v);
}

/* Clone (shallow copy) vector from v.
 * Note: Data from v is not copied to w, just the memory is allocated.
 *       Sundials calls N_VScale to copy between N_Vectors.
 */
N_Vector PF_NVClone(N_Vector v)
{
  N_Vector w = NULL;
  Vector *wdata = NULL;
  Vector *vdata = N_VectorData(v);
  int nc = 1;         /* dummy variable - not used */

  /* Create vector object w from v. */
  w = PF_NVNewEmpty(v->sunctx);
  wdata = NewVectorType(VectorGrid(vdata), nc, VectorNumGhost(vdata), VectorType(vdata));

  /* assign wdata to w and return*/
  N_VectorData(w) = wdata;
  N_VectorOwnsData(w) = true;
  return(w);
}

/* Destroy N_Vector */
void PF_NVDestroy(N_Vector v)
{
  if (v == NULL)
    return;

  /* free data content */
  if (v->content != NULL)
  {
    if (N_VectorOwnsData(v) && N_VectorData(v) != NULL)
    {
      /* free data */
      FreeVector(N_VectorData(v));
      N_VectorData(v) = NULL;
    }
    free(v->content);
    v->content = NULL;
  }

  /* free ops */
  if (v->ops != NULL)
  {
    free(v->ops);
    v->ops = NULL;
  }

  /* free vector v */
  free(v);
  v = NULL;
}

/* Return global length of N_Vector */
long int PF_NVGetLength(N_Vector v)
{
  return VectorSize(N_VectorData(v));
}

/* Wrapper functions for vector utility functions */

void PFVLinearSumFcn(
/* LinearSum : z = a * x + b * y              */
                     double   a,
                     N_Vector xvec,
                     double   b,
                     N_Vector yvec,
                     N_Vector zvec)

{
  Vector *x = N_VectorData(xvec);
  Vector *y = N_VectorData(yvec);
  Vector *z = N_VectorData(zvec);

  PFVLinearSum(a, x, b, y, z);
}

void PFVConstInitFcn(
/* ConstInit : z = c   */
                     double   c,
                     N_Vector zvec)
{
  Vector *z = N_VectorData(zvec);

  PFVConstInit(c, z);
}

void PFVProdFcn(
/* Prod : z_i = x_i * y_i   */
                N_Vector xvec,
                N_Vector yvec,
                N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *y = N_VectorData(yvec);
  Vector *z = N_VectorData(zvec);

  PFVProd(x, y, z);
}

void PFVDivFcn(
/* Div : z_i = x_i / y_i   */
               N_Vector xvec,
               N_Vector yvec,
               N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *y = N_VectorData(yvec);
  Vector *z = N_VectorData(zvec);

  PFVDiv(x, y, z);
}

void PFVScaleFcn(
/* Scale : z = c * x   */
                 double   c,
                 N_Vector xvec,
                 N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  PFVScale(c, x, z);
}

void PFVAbsFcn(
/* Abs : z_i = |x_i|   */
               N_Vector xvec,
               N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  PFVAbs(x, z);
}

void PFVInvFcn(
/* Inv : z_i = 1 / x_i    */
               N_Vector xvec,
               N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  PFVInv(x, z);
}

void PFVAddConstFcn(
/* AddConst : z_i = x_i + b  */
                    N_Vector xvec,
                    double   b,
                    N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  PFVAddConst(x, b, z);
}

double PFVDotProdFcn(
/* DotProd = x dot y   */
                     N_Vector xvec,
                     N_Vector yvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *y = N_VectorData(yvec);

  return PFVDotProd(x, y);
}

double PFVMaxNormFcn(
/* MaxNorm = || x ||_{max}   */
                     N_Vector xvec)
{
  Vector *x = N_VectorData(xvec);

  return PFVMaxNorm(x);
}

double PFVWrmsNormFcn(
/* WrmsNorm = sqrt((sum_i (x_i * w_i)^2)/length)  */
                      N_Vector xvec,
                      N_Vector wvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *w = N_VectorData(wvec);

  return PFVWrmsNorm(x, w);
}

double PFVWL2NormFcn(
/* WL2Norm = sqrt(sum_i (x_i * w_i)^2)  */
                     N_Vector xvec,
                     N_Vector wvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *w = N_VectorData(wvec);

  return PFVWL2Norm(x, w);
}

double PFVL1NormFcn(
/* L1Norm = sum_i |x_i|  */
                    N_Vector xvec)
{
  Vector *x = N_VectorData(xvec);

  return PFVL1Norm(x);
}

double PFVMinFcn(
/* Min = min_i(x_i)   */
                 N_Vector xvec)
{
  Vector *x = N_VectorData(xvec);

  return PFVMin(x);
}

double PFVMaxFcn(
/* Max = max_i(x_i)   */
                 N_Vector xvec)
{
  Vector *x = N_VectorData(xvec);

  return PFVMax(x);
}

int PFVConstrProdPosFcn(
/* ConstrProdPos: Returns a boolean FALSE if some c[i]!=0.0  */
/*                and x[i]*c[i]<=0.0 */
                        N_Vector cvec,
                        N_Vector xvec)
{
  Vector *c = N_VectorData(cvec);
  Vector *x = N_VectorData(xvec);

  return PFVConstrProdPos(c, x);
}

void PFVCompareFcn(
/* Compare : z_i = (x_i > c)  */
                   double   c,
                   N_Vector xvec,
                   N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  PFVCompare(c, x, z);
}

int PFVInvTestFcn(
/* InvTest = (x_i != 0 forall i), z_i = 1 / x_i  */
                  N_Vector xvec,
                  N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  return PFVInvTest(x, z);
}

/* Constraint test functions required by Sundials if using constraints.
 * Adapted from SUNDIALS nvector -DOK
 */
bool PFVConstrMaskFcn(
                      N_Vector xvec,
                      N_Vector yvec,
                      N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *y = N_VectorData(yvec);
  Vector *z = N_VectorData(zvec);

  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *y_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  const double * __restrict__ yp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_y, ny_y, nz_y;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_y, i_z;
  double temp = ZERO;
  bool test = false;

  grid = VectorGrid(x);
  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
    x_sub = VectorSubvector(x, sg);
    y_sub = VectorSubvector(y, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_y = SubvectorNX(y_sub);
    ny_y = SubvectorNY(y_sub);
    nz_y = SubvectorNZ(y_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);
    yp = SubvectorElt(y_sub, ix, iy, iz);

    i_x = 0;
    i_y = 0;
    i_z = 0;
    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_y, nx_y, ny_y, nz_y, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      zp[i_z] = ZERO;
      test = (fabs(xp[i_x]) > RCONST(1.5) && yp[i_y] * xp[i_x] <= ZERO) ||
             (fabs(xp[i_x]) > RCONST(0.5) && yp[i_y] * xp[i_x] < ZERO);

      if (test)
      {
        temp = zp[i_z] = ONE;
      }
    });
  }
  IncFLOPCount(VectorSize(x));

  /* Return false if any constraint is violated */
  return (temp == ONE) ? false : true;
}

/* minimum of quotients.
 * Adapted from SUNDIALS nvector - DOK
 */
double PFVMinQuotientFcn(
                         N_Vector xvec,
                         N_Vector zvec)
{
  Vector *x = N_VectorData(xvec);
  Vector *z = N_VectorData(zvec);

  Grid       *grid = VectorGrid(x);
  Subgrid    *subgrid;

  Subvector  *x_sub;
  Subvector  *z_sub;

  const double * __restrict__ xp;
  double * __restrict__ zp;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_x, ny_x, nz_x;
  int nx_z, ny_z, nz_z;

  int sg, i, j, k, i_x, i_z;
  bool test = true;
  double min_val = SUN_BIG_REAL;

  amps_Invoice result_invoice;

  ForSubgridI(sg, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, sg);

    z_sub = VectorSubvector(z, sg);
    x_sub = VectorSubvector(x, sg);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_x = SubvectorNX(x_sub);
    ny_x = SubvectorNY(x_sub);
    nz_x = SubvectorNZ(x_sub);

    nx_z = SubvectorNX(z_sub);
    ny_z = SubvectorNY(z_sub);
    nz_z = SubvectorNZ(z_sub);

    zp = SubvectorElt(z_sub, ix, iy, iz);
    xp = SubvectorElt(x_sub, ix, iy, iz);

    i_x = 0;
    i_z = 0;
    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              i_x, nx_x, ny_x, nz_x, 1, 1, 1,
              i_z, nx_z, ny_z, nz_z, 1, 1, 1,
    {
      if (zp[i_z] == ZERO)
      {
        continue;
      }
      else
      {
        if (!test)
        {
          min_val = MIN(min_val, xp[i_x] / zp[i_z]);
        }
        else
        {
          min_val = xp[i_x] / zp[i_z];
          test = false;
        }
      }
    });
  }

  result_invoice = amps_NewInvoice("%d", min_val);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Min);
  amps_FreeInvoice(result_invoice);

  return(min_val);
}

#ifdef __cplusplus
}
#endif

#endif

