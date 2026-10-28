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

#ifndef _PF_HYPRE
#define _PF_HYPRE

#ifdef HAVE_HYPRE
#include "hypre_dependences.h"

/**
 * Copy a ParFlow vector to a Hypre vector.
 *
 * @param pf_vector the source vector
 * @param hypre_vector the destination vector
 */
void CopyParFlowVectorToHypreVector(
                                    Vector *            pf_vector,
                                    HYPRE_StructVector* hypre_vector
                                    );

/**
 * Copy a Hypre vector to a ParFlow vector.
 *
 * @param hypre_vector the source vector
 * @param pf_vector the destination vector
 */
void CopyHypreVectorToParflowVector(
                                    HYPRE_StructVector* hypre_vector,
                                    Vector *            pf_vector
                                    );


void HypreAssembleGrid(
                       Grid*             pf_grid,
                       HYPRE_StructGrid* hypre_grid,
                       double*           dxyz
                       );

/**
 * Create and initialize Hypre structures.
 *
 * Sets up the Hypre data structures for the grid, stencil, matrix and vectors.
 * This setup is common to all the Hypre solvers ParFlow is using so it has
 * been factored out to a common utility function.   Pointers to
 * the Hypre data structures are returned.
 *
 * @param pf_matrix Source matrix
 * @param hypre_grid Constructed hypre grid
 * @param hypre_stencil Constructed hypre stencil
 * @param hypre_mat Constructed hypre matrix
 * @param hypre_rhs Constructed rhs vector
 * @parame hypre_soln Constructed solution vector
 */
void HypreInitialize(Matrix*              pf_matrix,
                     HYPRE_StructGrid*    hypre_grid,
                     HYPRE_StructStencil* hypre_stencil,
                     HYPRE_StructMatrix*  hypre_mat,
                     HYPRE_StructVector*  hypre_rhs,
                     HYPRE_StructVector*  hypre_soln
                     );

/**
 * Assemble the Hypre matrix from B and C ParFlow matrices.Matrix values are
 * inserted by box and stencil entry.
 *
 * It issues one HYPRE_StructMatrixSetBoxValues2 call per stencil entry
 * per subgrid (stencil_size calls total per subgrid, instead of one
 * HYPRE_StructMatrixSetValues call per grid cell). The overland-flow
 * top-surface correction is still done cell-wise, but only over the
 * 2-D top layer (nx*ny cells).
 *
 * @param pf_Bmat The B matrix
 * @param pf_Cmat The C matrix
 * @param hyre_mat The filled in Hypre matrix
 * @param problem_data ParFlow problem data
 */
void HypreAssembleMatrix(
                                   Matrix *            pf_Bmat,
                                   Matrix *            pf_Cmat,
                                   HYPRE_StructMatrix* hypre_mat,
                                   ProblemData *       problem_data
                                   );

#endif

#endif
