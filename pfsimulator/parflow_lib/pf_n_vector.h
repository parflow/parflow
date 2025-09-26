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

#include "vector.h"
#include "llnltyps.h"

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

#if defined (PARFLOW_HAVE_SUNDIALS)
#include "sundials/sundials_core.h"

/* Content field for SUNDIALS' N_Vector object. 
 * We could use Vector * directly without this 
 * wrapper struct. - DOK
*/
struct PF_N_Vector_Content_struct {
    Vector *data;
    bool   owns_data;
 };

/* forward reference for pointers to structs */
typedef struct PF_N_Vector_Content_struct* PF_N_Vector_Content;

/* N_Vector Accessor Macros */
/* Macros to interact with SUNDIALS N_Vector */
#define N_VectorContent(n_vector)		((PF_N_Vector_Content)((n_vector)->content))
#define N_VectorData(n_vector)			(N_VectorContent(n_vector)->data)
#define N_VectorOwnsData(n_vector)		(N_VectorContent(n_vector)->owns_data)

/* N_Vector.c protos for External SUNDIALS */
#ifdef __cplusplus
extern "C" {
#endif
N_Vector PF_NVNewEmpty(SUNContext sunctx);
N_Vector PF_NVNew(SUNContext sunctx, Grid *grid, int num_ghost);
N_Vector PF_NVNewFromVector(SUNContext sunctx, Vector *data);
N_Vector PF_NVClone(N_Vector v);
void PF_NVDestroy(N_Vector v);
int PF_NVGetLength(N_Vector v);

void PFVLinearSumFcn(double a, N_Vector x, double b, N_Vector y, N_Vector z);
void PFVConstInitFcn(double c, N_Vector z);
void PFVProdFcn(N_Vector x, N_Vector y, N_Vector z);
void PFVDivFcn(N_Vector x, N_Vector y, N_Vector z);
void PFVScaleFcn(double c, N_Vector x, N_Vector z);
void PFVAbsFcn(N_Vector x, N_Vector z);
void PFVInvFcn(N_Vector x, N_Vector z);
void PFVAddConstFcn(N_Vector x, double b, N_Vector z);
double PFVDotProdFcn(N_Vector x, N_Vector y);
double PFVMaxNormFcn(N_Vector x);
double PFVWrmsNormFcn(N_Vector x, N_Vector w);
double PFVWL2NormFcn(N_Vector x, N_Vector w);
double PFVL1NormFcn(N_Vector x);
double PFVMinFcn(N_Vector x);
double PFVMaxFcn(N_Vector x);
int PFVConstrProdPosFcn(N_Vector c, N_Vector x);
void PFVCompareFcn(double c, N_Vector x, N_Vector z);
int PFVInvTestFcn(N_Vector x, N_Vector z);
double PFVMinQuotientFcn(N_Vector xvec, N_Vector zvec);
bool PFVConstrMaskFcn(N_Vector xvec, N_Vector yvec, N_Vector zvec);

#ifdef __cplusplus
}
#endif

#endif
#endif
