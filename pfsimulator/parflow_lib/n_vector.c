/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

#include "parflow.h"

static struct {
   Grid *grid;
   int   num_ghost;
} pf2kinsol_data;


void SetPf2KinsolData(
   Grid        *grid,
   int          num_ghost)
{
   pf2kinsol_data.grid = grid;
   pf2kinsol_data.num_ghost = num_ghost;
}

N_Vector N_VNew(
   int    N,
   void  *machEnv)
{
   Grid    *grid;
   int      num_ghost;

   (void) N;
   (void) machEnv;

   grid      = pf2kinsol_data.grid;
   num_ghost = pf2kinsol_data.num_ghost;
   return N_VNew_PF(grid,NUMDIMS) ;
}

N_Vector N_VNew_PF(
Grid *grid,
int numDims)
{
  N_Vector v;
  N_VectorContent content;
  int i;

  v = N_VNewEmpty_PF(numDims);

  content = NV_CONTENT_PF(v);


  for (i = 0; i < numDims; i++)
  {
    content->dims[i] =  NewVectorType(grid, 1, 1, vector_cell_centered);
  }

  return(v);

}

N_Vector N_VNewEmpty_PF(
int numDims)
{
  N_Vector v;
  N_Vector_Ops ops;
  N_VectorContent content;
  int i;

  v = (N_Vector) malloc(sizeof *v);
  ops = (N_Vector_Ops) malloc(sizeof(struct _generic_N_Vector_Ops));

  ops->nvclone           = N_VClone_PF;
  ops->nvcloneempty      = N_VCloneEmpty_PF;
  ops->nvdestroy         = N_VDestroy_PF;
  ops->nvspace           = N_VSpace_PF;
  ops->nvgetarraypointer = NULL;
  ops->nvsetarraypointer = NULL;
  ops->nvlinearsum       = N_VLinearSum_PF;
  ops->nvconst           = N_VConst_PF;
  ops->nvprod            = N_VProd_PF;
  ops->nvdiv             = N_VDiv_PF;
  ops->nvscale           = N_VScale_PF;
  ops->nvabs             = N_VAbs_PF;
  ops->nvinv             = N_VInv_PF;
  ops->nvaddconst        = NULL;
  ops->nvdotprod         = N_VDotProd_PF;
  ops->nvmaxnorm         = N_VMaxNorm_PF;
  ops->nvwrmsnormmask    = NULL;
  ops->nvwrmsnorm        = NULL;
  ops->nvmin             = N_VMin_PF;
  ops->nvwl2norm         = N_VWL2Norm_PF;
  ops->nvl1norm          = N_VL1Norm_PF;
  ops->nvcompare         = NULL;
  ops->nvinvtest         = NULL;
  ops->nvconstrmask      = N_VConstrMask_PF;
  ops->nvminquotient     = N_VMinQuotient_PF;

  content = (N_VectorContent) malloc(sizeof(struct _N_VectorContent));
  content->dims = (Vector **) malloc(numDims * sizeof(Vector *));

  content->numDims = numDims;

  for (i = 0; i < numDims; i++)
  {
    content->dims[i] = NULL;
  }

  v->content = content;
  v->ops = ops;

  return(v);
}


void N_VPrint(
   N_Vector x)
{
  Grid       *grid;
  N_VectorContent content;
  Vector     *v;
  Subgrid    *subgrid;

  Subvector  *v_sub;

  double     *vp;

  int         ix,   iy,   iz;
  int         nx,   ny,   nz;
  int         nx_x, ny_x, nz_x;
  
  int         sg, i, j, k, i_x;


  content = NV_CONTENT_PF(x);

  grid = VectorGrid(content->dims[0]);

  v = content->dims[0];


  ForSubgridI(sg, GridSubgrids(grid))
  {
     subgrid = GridSubgrid(grid, sg);

     v_sub = VectorSubvector(v, sg);

     ix = SubgridIX(subgrid);
     iy = SubgridIY(subgrid);
     iz = SubgridIZ(subgrid);

     nx = SubgridNX(subgrid);
     ny = SubgridNY(subgrid);
     nz = SubgridNZ(subgrid);

     nx_x = SubvectorNX(v_sub);
     ny_x = SubvectorNY(v_sub);
     nz_x = SubvectorNZ(v_sub);

     vp = SubvectorElt(v_sub, ix, iy, iz);

     i_x = 0;
     BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
	       i_x, nx_x, ny_x, nz_x, 1, 1, 1,
	       {
		  printf("%g\n", vp[i_x]);
                  fflush(NULL);
	       });
  }
  printf("\n");
  fflush(NULL);
}

/* N_VCloneEmpty_PF creates a new N_Vector of the same type as an 
 *    existing w vector; it does not allocate storage for the data array*/

N_Vector N_VCloneEmpty_PF(
N_Vector w )
{
  N_Vector clone;
  N_Vector_Ops ops;
  N_VectorContent clone_content,wcontent;
  int numDims;

  wcontent = NV_CONTENT_PF(w);

  /* Extract the content of the incoming N_Vector */
  numDims = wcontent->numDims;

  clone = (N_Vector) malloc(sizeof *w);

  ops = (N_Vector_Ops) malloc (sizeof(struct _generic_N_Vector_Ops));

  ops->nvclone           = w->ops->nvclone;
  ops->nvcloneempty      = w->ops->nvcloneempty;
  ops->nvdestroy         = w->ops->nvdestroy;
  ops->nvspace           = w->ops->nvspace;
  ops->nvgetarraypointer = w->ops->nvgetarraypointer;
  ops->nvsetarraypointer = w->ops->nvsetarraypointer;
  ops->nvlinearsum       = w->ops->nvlinearsum;
  ops->nvconst           = w->ops->nvconst;
  ops->nvprod            = w->ops->nvprod;
  ops->nvdiv             = w->ops->nvdiv;
  ops->nvscale           = w->ops->nvscale;
  ops->nvabs             = w->ops->nvabs;
  ops->nvinv             = w->ops->nvinv;
  ops->nvaddconst        = w->ops->nvaddconst;
  ops->nvdotprod         = w->ops->nvdotprod;
  ops->nvmaxnorm         = w->ops->nvmaxnorm;
  ops->nvwrmsnormmask    = w->ops->nvwrmsnormmask;
  ops->nvwrmsnorm        = w->ops->nvwrmsnorm;
  ops->nvmin             = w->ops->nvmin;
  ops->nvwl2norm         = w->ops->nvwl2norm;
  ops->nvl1norm          = w->ops->nvl1norm;
  ops->nvcompare         = w->ops->nvcompare;
  ops->nvinvtest         = w->ops->nvinvtest;
  ops->nvconstrmask      = w->ops->nvconstrmask;
  ops->nvminquotient     = w->ops->nvminquotient;

  clone_content = (N_VectorContent) malloc(sizeof(struct _N_VectorContent));

  clone_content->dims = NULL;

  clone_content->numDims = numDims;

  clone->content = clone_content;
  clone->ops = ops;

  return(clone);

}

/* N_VClone_PF returns a new Vector of the same form as the
 *    input N_Vector; it does not copy the vector but allocates the storage*/

N_Vector N_VClone_PF(
N_Vector w)
{
  N_Vector clone;
  Grid *grid;
  N_VectorContent clone_content,w_content;
  int i,numDims;

  w_content = NV_CONTENT_PF(w);
  grid = VectorGrid(w_content->dims[0]);

  clone = N_VCloneEmpty_PF(w);
  clone_content = NV_CONTENT_PF(clone);
  numDims = clone_content->numDims;

  clone_content->dims = (Vector **) malloc(numDims * sizeof(Vector *));

  for (i = 0; i < numDims; i++)
  {
    clone_content->dims[i] = NewVectorType(grid, 1, 1, vector_cell_centered);
  }

  return(clone);

}
  

/* Destroy storage of N_Vector */
void N_VDestroy_PF(
N_Vector v)
{
  N_VectorContent content;
  Vector *dim;
  int i, numDims;

  content = NV_CONTENT_PF(v);

  numDims = content->numDims;
  for (i = 0; i < numDims; i++)
  {
    dim = content->dims[i];
    FreeVector(dim);
  }
  free(content->dims);

  free(content);
  free(v);
}

/* Returns space requirements for one N_Vector */
void N_VSpace_PF(
N_Vector v,
long int *lrw, 
long int *liw)
{
  N_VectorContent content;
  Vector *dim;
  int nprocs;
  int i, numDims;

  content = NV_CONTENT_PF(v);
  *lrw = 0;

  numDims = content->numDims;
  for (i = 0; i < numDims; i++)
  {
    dim = content->dims[i];
    *lrw +=SizeOfVector(dim);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  *liw = 0;
}
   

/* Perform operation z = ax + by */
void N_VLinearSum_PF(
double a, 
N_Vector x,
double b,
N_Vector y, 
N_Vector z)
{
  Vector *xx, *yy, *zz;
  N_VectorContent xcont, ycont, zcont;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    yy = ycont->dims[i];
    zz = zcont->dims[i];
    PFVLinearSum(a, xx, b, yy, zz);
  }

}

/* Sets all components of z to c */
void N_VConst_PF(
double c,
N_Vector z)
{
  N_VectorContent content;
  Vector *zz;
  int i, numDims;

  content= NV_CONTENT_PF(z);

  numDims = content->numDims;
  for (i = 0; i < numDims; i++)
  {
    zz = content->dims[i];
    PFVConstInit(c, zz);
  }

}

/* Sets z to the component-wise product of inputs x and y */
void N_VProd_PF(
N_Vector x, 
N_Vector y, 
N_Vector z)
{
  N_VectorContent xcont, ycont, zcont;
  Vector *xx, *yy, *zz;
  int i,numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
     xx = xcont->dims[i];
     yy = ycont->dims[i];
     zz = zcont->dims[i];
     PFVProd(xx, yy, zz);
  }

}

/* Sets z to be the component-wise ratio of the inputs x and y */
void N_VDiv_PF(
N_Vector x, 
N_Vector y, 
N_Vector z)
{
  N_VectorContent xcont, ycont, zcont;
  Vector *xx, *yy, *zz;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);
  zcont = NV_CONTENT_PF(z);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    yy = ycont->dims[i];
    zz = zcont->dims[i];
    PFVDiv(xx, yy, zz);
  }

}

/* Scales x by the scalar c and returns the result in z */
void N_VScale_PF(
double c,
N_Vector x, 
N_Vector z)
{
  N_VectorContent xcont, zcont;
  Vector *xx, *zz;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  zcont = NV_CONTENT_PF(z);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    zz = zcont->dims[i];
    PFVScale(c, xx, zz);
  }

}

/* Sets the components of y to be the absolute values of the components of x */
void N_VAbs_PF(
N_Vector x, 
N_Vector y)
{
  N_VectorContent xcont, ycont;
  Vector *xx, *yy;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    yy = ycont->dims[i];
    PFVAbs(xx, yy);
  }

}

/* Sets the components of y to be the inverse of the components of x;
 *    does not check for div by zero */
void N_VInv_PF(
N_Vector x, 
N_Vector y)
{
  N_VectorContent xcont, ycont;
  Vector *xx, *yy;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    yy = ycont->dims[i];
    PFVInvTest(xx, yy);
  }

}

/* Returns the value of the ordinary dot product of x and y */
double N_VDotProd_PF(
N_Vector x, 
N_Vector y)
{
  N_VectorContent xcont, ycont;
  Vector *xx, *yy;
  double dprod;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);
  ycont = NV_CONTENT_PF(y);

  dprod = 0.0;

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    yy = ycont->dims[i];
    dprod += PFVDotProd(xx, yy);
  }

  return(dprod);
}

/* Returns the maximum norm of x */
double N_VMaxNorm_PF(
N_Vector x)
{
  N_VectorContent xcont;
  Vector *xx;
  double dimMaxnorm;
  double maxnorm = 0.0;
  int i,numDims;

  xcont = NV_CONTENT_PF(x);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    dimMaxnorm = PFVMaxNorm(xx);
    if (dimMaxnorm > maxnorm) maxnorm = dimMaxnorm;
  }

  return(maxnorm);
}

/* Returns the smallest element of x */
double N_VMin_PF(
N_Vector x)
{
  N_VectorContent xcont;
  Vector *xx;
  double dimMin;
  double min = BIG_REAL;
  int i,numDims;

  xcont = NV_CONTENT_PF(x);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    dimMin= PFVMin(xx);
    if (dimMin < min) min = dimMin;
  }

  return(min);
}

/* Returns the weighted Euclidean nrom of x with weights w */
double N_VWL2Norm_PF(
N_Vector x, 
N_Vector w)
{
  N_VectorContent xcont, wcont;
  Vector *xx, *ww;
  double sum = 0.0;
  double l2normm;
  int i,numDims;

  xcont = NV_CONTENT_PF(x);
  wcont = NV_CONTENT_PF(w);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    ww = wcont->dims[i];
    /* The modified PFVWL2Norm function returns only the sum, NOT sqrt(sum) */
    sum += PFVWL2Norm(xx, ww);
  }

  l2normm = sqrt(sum);

  return(l2normm);
}

/* Returns the L1 norm of x */
double N_VL1Norm_PF(
N_Vector x)
{
  N_VectorContent xcont;
  Vector *xx;
  double sum = 0.0;
  int i, numDims;

  xcont = NV_CONTENT_PF(x);

  numDims = xcont->numDims;
  for (i = 0; i < numDims; i++)
  {
    xx = xcont->dims[i];
    sum += PFVL1Norm(xx);
  }

  return(sum);
}


/* Performs a constraint test (old kinsol) */
booleantype N_VConstrProdPos_PF(
N_Vector c,
N_Vector x)
{
  N_VectorContent ccont, xcont;
  Vector *cc, *xx;
  int dimTest;
  int  test = 1;
  int i, numDims;

  ccont = NV_CONTENT_PF(c);
  xcont = NV_CONTENT_PF(x);

  numDims = ccont->numDims;
  for (i = 0; i < numDims; i++)
  {
    cc = ccont->dims[i];
    xx = xcont->dims[i];
    dimTest = PFVConstrProdPos(cc, xx);
    if (dimTest < test) test  = dimTest;
  }

  if (test == 0)
    return(FALSE);
  else
    return(TRUE);
}


/* Performs a constraint test (see manual) */
booleantype N_VConstrMask_PF(
N_Vector c, 
N_Vector x, 
N_Vector m)
{
  N_VectorContent ccont, xcont, mcont;
  Vector *cc, *xx, *mm;
  int dimTest;
  int  test = 1;
  int i, numDims;

  ccont = NV_CONTENT_PF(c);
  xcont = NV_CONTENT_PF(x);
  mcont = NV_CONTENT_PF(m);

  numDims = ccont->numDims;
  for (i = 0; i < numDims; i++)
  {
    cc = ccont->dims[i];
    xx = xcont->dims[i];
    mm = mcont->dims[i];
    dimTest = PFVConstrMask(cc, xx, mm);
    if (dimTest < test) test  = dimTest;
  }

  if (test == 0)
    return(FALSE);
  else
    return(TRUE);
}

/* This routine returns the minimum of the quotients obtained 
 *    by termswise dividing num by denom; zero elements in denom
 *       will be skipped; if no quotients are found, thenthe large 
 *          value BIG_REAL in sundialstypes.h is returned */
double N_VMinQuotient_PF(
N_Vector num,
N_Vector denom)
{
  N_VectorContent ncont, dcont;
  Vector *n, *d;
  double dimMin;
  double min = BIG_REAL;
  int i, numDims;

  ncont = NV_CONTENT_PF(num);
  dcont = NV_CONTENT_PF(denom);

  numDims = ncont->numDims;
  for (i = 0; i < numDims; i++)
  {
    n = ncont->dims[i];
    d = dcont->dims[i];
    dimMin = PFVMinQuotient(n, d);
    if (dimMin < min) min = dimMin;
  }

  return(min);
}


