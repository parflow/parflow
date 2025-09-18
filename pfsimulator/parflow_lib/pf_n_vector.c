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
#ifndef _PF_N_VECTOR_HEADER
#define _PF_N_VECTOR_HEADER

#include "parflow.h"

#if defined (PARFLOW_HAVE_SUNDIALS)
#ifdef __cplusplus
extern "C" {
#endif

/* Create new vector */
PF_N_Vector PF_NVNewEmpty(SUNContext sunctx)
{
  PF_N_Vector v               = NULL;
  PF_N_Vector_Content content = NULL;
  PF_N_Vector_Ops ops         = NULL;

  /* Allocate memory for new PF_N_Vector */
  v = (PF_N_Vector)malloc(sizeof *v);

  /* Allocate memory for ops */
  ops = (PF_N_Vector_Ops)malloc(sizeof *ops);
  /* set function pointers for ops */
  ops->nvlinearsum    = PFVLinearSum;
  ops->nvconst        = PFVConstInit;
  ops->nvprod         = PFVProd;
  ops->nvdiv          = PFVDiv;
  ops->nvscale        = PFVScale;
  ops->nvabs          = PFVAbs;
  ops->nvinv          = PFVInv;
  ops->nvaddconst     = PFVAddConst;
  ops->nvdotprod      = PFVDotProd;
  ops->nvmaxnorm      = PFVMaxNorm;
  ops->nvwrmsnorm     = PFVWrmsNorm;
  ops->nvwrmsnormmask = NULL;
  ops->nvmin          = PFVMin;
  ops->nvwl2norm      = PFVWL2Norm;
  ops->nvl1norm       = PFVL1Norm;
  ops->nvcompare      = PFVCompare;
  ops->nvinvtest      = PFVInvTest;
  ops->nvconstrmask   = NULL;
  ops->nvminquotient  = NULL;

  /* Allocate memory for vector content (data) */
  content = (PF_N_Vector_Content)malloc(sizeof *content);
  /* set content data to NULL (empty) */
  content->data = NULL;

  /* Attach pointers to complete new PF_N_Vector */
  v->content = content;
  v->ops = ops;
  v->sunctx = sunctx; 

  return (v);
}

/* Create new vector */
PF_N_Vector PF_NVNew(SUNContext sunctx)
{
  PF_N_Vector v = NULL;
  Vector *data  = NULL;
  int dummy     = 0;

  /* create empty vector object */
  v = PF_NVNewEmpty(sunctx);
  
  /* create vector data and attach to content */
  /* Existing function takes two dummy variables */
  data = N_VNew(dummy, (void *)&dummy);
  v->content->data = data;
  
  return(v);
}

/* Clone vector from v */
PF_N_Vector PF_NVClone(PF_N_Vector v)
{
  PF_N_Vector w  = NULL;
  Vector *wdata  = NULL;
  Vector *vdata  = v->content->data;
  int nc         = 1; /* dummy variable - not used */

  /* Create vector object w from v. */
  w = PF_NVNewEmpty(sunctx);
  wdata = NewVectorType(VectorGrid(v), nc, VectorNumGhost(v), VectorType(v));
  /* Copy vector data from v to w */
  PFVCopy(vdata, wdata);
  
  /* assign wdata to w and return*/
  w->content->data = wdata;
  
  return(w);
}

/* Destroy PF_N_Vector */
void PF_NVDestroy(PF_N_Vector v)
{
  if(v == NULL) return;

  /* free data content */
  if(v->content != NULL)
  {
    if(v->content->data != NULL)
    {
      /* free data */
      N_VFree(v->content->data);
      v->content->data = NULL;
    }
    free(v->content);
    v->content = NULL;
  }

  /* free ops */
  if(v->ops != NULL)
  {
    free(v->ops);
    v->ops = NULL;
  }

  /* free vector v */
  free(v);
  v = NULL;
}

/* Return global length of PF_N_Vector */
int PF_NVGetLength(PF_N_Vector v)
{
  return VectorSize(v->content->data);
}

///* Perform constraint test. This is used for constraint checking in kinsols */
//bool PF_NVConstrMask(PF_N_Vector c, PF_N_Vector x, PF_N_Vector m)
//{  
//  return true;
//}

#ifdef __cplusplus
}
#endif

#endif

#endif

