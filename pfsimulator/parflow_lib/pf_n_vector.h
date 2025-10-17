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
#include "sundials/sundials_core.h"

/* Content field for the ParFlow SUNDIALS' N_Vector object */
struct PF_N_Vector_Content_struct {
  Vector *data;
  bool owns_data;
};

/* forward reference for pointers to structs */
typedef struct PF_N_Vector_Content_struct* PF_N_Vector_Content;

/* N_Vector Accessor Macros */
/* Macros to interact with SUNDIALS N_Vector */
#define N_VectorContent(n_vector)               ((PF_N_Vector_Content)((n_vector)->content))
#define N_VectorData(n_vector)                  (N_VectorContent(n_vector)->data)
#define N_VectorOwnsData(n_vector)              (N_VectorContent(n_vector)->owns_data)

#endif
#endif
