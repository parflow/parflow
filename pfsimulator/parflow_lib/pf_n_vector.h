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

#if defined (PARFLOW_HAVE_SUNDIALS)
//#include "/usr/gapps/thcs/apps/toss_4_x86_64_ib/sundials/7.4.0/include/sundials/sundials_context.h"
//#include "kinsol_dependences.h"

typedef struct PF_N_Vector_Content_struct {
    Vector *PF_vector;
 } *PF_N_Vector_Content;

typedef struct PF_N_Vector_Ops_struct {
    /* Constructors and destructors */
    Vector (*nvclone)(Vector);
    void (*nvdestroy)(Vector);
    /* standard vector operations */
    void (*nvlinearsum)(double a, Vector *x, double b,
                                  Vector *y, Vector *z);
    void (*nvconst)(double c, Vector *z);
    void (*nvprod)(Vector *x, Vector *y, Vector *z);
    void (*nvdiv)(Vector *x, Vector *y, Vector *z);
    void (*nvscale)(double c, Vector *x, Vector *z);
    void (*nvabs)(Vector *x, Vector *z);
    void (*nvinv)(Vector *x, Vector *z);
    void (*nvaddconst)(Vector *x, double b, Vector *z);
    double (*nvdotprod)(Vector *x, Vector *y);
    double (*nvmaxnorm)(Vector *x);
    double (*nvwrmsnorm)(Vector *x, Vector *w);
    double (*nvwrmsnormmask)(Vector *x, Vector *w, Vector *id);
    double (*nvmin)(Vector *x);
    double (*nvwl2norm)(Vector *x, Vector *w);
    double (*nvl1norm)(Vector *x);
    void (*nvcompare)(double c, Vector *x, Vector *z);
    int (*nvinvtest)(Vector *x, Vector *z);
    int (*nvconstrmask)(Vector *c, Vector *x, Vector *m);
    double (*nvminquotient)(Vector *num, Vector *denom);
} *PF_N_Vector_Ops;

typedef struct PF_N_Vector_struct {
    PF_N_Vector_Content data;
    PF_N_Vector_Ops     ops;
    SUNContext 		sunctx;
} *PF_N_Vector;

#endif
#endif
