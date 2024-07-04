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
* Header info for Matrix data structures
*
*****************************************************************************/

#ifndef _MATRIX_HEADER
#define _MATRIX_HEADER

#include "grid.h"

#ifdef HAVE_SAMRAI
#include "SAMRAI/xfer/RefineAlgorithm.h"
#include "SAMRAI/xfer/RefineSchedule.h"
#endif

enum matrix_type {
  matrix_cell_centered,
  matrix_non_samrai
};


/*--------------------------------------------------------------------------
 * Stencil
 *--------------------------------------------------------------------------*/

typedef int StencilElt[3];

typedef struct {
  StencilElt  *shape;    /* Description of a stencil's shape */
  int size;              /* Number of stencil coefficients */
} Stencil;

/*--------------------------------------------------------------------------
 * Submatrix
 *--------------------------------------------------------------------------*/

typedef struct {
  double*    data;             /* Pointer to Matrix data */
  int allocated;               /* Is data was allocated */

  int*       data_index;       /* array of indices corresponding to
                                * stencil coefficient starting positions
                                * in data array */

  int data_size;               /* Size of data */

  Subregion* data_space;
} Submatrix;

/*--------------------------------------------------------------------------
 * Matrix
 *--------------------------------------------------------------------------*/

typedef struct {
  Submatrix       **submatrices;   /* Array of pointers to submatrices */

  Grid             *grid;          /* Matrix domain */
  SubregionArray   *range;         /* Matrix range */

  SubregionArray   *data_space;    /* Description of Matrix data */

  Stencil          *stencil;       /* Matrix stencil */

  int              *data_stencil;       /* Stencil for the stored data */
  int data_stencil_size;                /* Stencil size for the stored data */

  int symmetric;                   /* Is matrix symmetric? */

  int size;                        /* Total number of nonzero coefficients */

  CommPkg          *comm_pkg;      /* Information on how to update boundary */

  enum matrix_type type;

#ifdef HAVE_SAMRAI
  int samrai_id;                /* SAMRAI ID for this vector */
  // SGS FIXME This is very hacky and should be removed
  int table_index;                /* index into table of variables */

  SAMRAI::tbox::Pointer < SAMRAI::xfer::RefineAlgorithm > boundary_fill_refine_algorithm;
  SAMRAI::tbox::Pointer < SAMRAI::xfer::RefineSchedule > boundary_fill_schedule;
#endif
} Matrix;


/*--------------------------------------------------------------------------
 * Accessor functions for the Stencil structure
 *--------------------------------------------------------------------------*/

#define StencilShape(sg) ((sg)->shape)
#define StencilSize(sg)  ((sg)->size)

/*--------------------------------------------------------------------------
 * Accessor functions for the Submatrix structure
 *--------------------------------------------------------------------------*/

#define SubmatrixData(submatrix) ((submatrix)->data)
#define SubmatrixStencilData(submatrix, s) \
  (((submatrix)->data) + ((submatrix)->data_index[s]))

#define SubmatrixDataSpace(submatrix)  ((submatrix)->data_space)

#define SubmatrixIX(submatrix)   (SubregionIX(SubmatrixDataSpace(submatrix)))
#define SubmatrixIY(submatrix)   (SubregionIY(SubmatrixDataSpace(submatrix)))
#define SubmatrixIZ(submatrix)   (SubregionIZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixNX(submatrix)   (SubregionNX(SubmatrixDataSpace(submatrix)))
#define SubmatrixNY(submatrix)   (SubregionNY(SubmatrixDataSpace(submatrix)))
#define SubmatrixNZ(submatrix)   (SubregionNZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixSX(submatrix)   (SubregionSX(SubmatrixDataSpace(submatrix)))
#define SubmatrixSY(submatrix)   (SubregionSY(SubmatrixDataSpace(submatrix)))
#define SubmatrixSZ(submatrix)   (SubregionSZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixEltIndex(submatrix, x, y, z)                   \
  (((x) - SubmatrixIX(submatrix)) / SubmatrixSX(submatrix) +    \
   (((y) - SubmatrixIY(submatrix)) / SubmatrixSY(submatrix) +   \
    (((z) - SubmatrixIZ(submatrix)) / SubmatrixSZ(submatrix)) * \
    SubmatrixNY(submatrix)) *                                   \
   SubmatrixNX(submatrix))

#define SubmatrixElt(submatrix, s, x, y, z) \
  (SubmatrixStencilData(submatrix, s) + SubmatrixEltIndex(submatrix, x, y, z))

#define SubmatrixSize(submatrix) SubmatrixNX((submatrix)) * SubmatrixNY((submatrix)) * \
  SubmatrixNZ((submatrix))


/*--------------------------------------------------------------------------
 * Accessor functions for the Matrix structure
 *--------------------------------------------------------------------------*/

#define MatrixSubmatrix(matrix, n) ((matrix)->submatrices[(n)])

#define MatrixGrid(matrix)        ((matrix)->grid)
#define MatrixRange(matrix)       ((matrix)->range)

#define MatrixDataSpace(matrix)   ((matrix)->data_space)

#define MatrixStencil(matrix)     ((matrix)->stencil)

#define MatrixDataStencil(matrix)     ((matrix)->data_stencil)
#define MatrixDataStencilSize(matrix) ((matrix)->data_stencil_size)

#define MatrixSymmetric(matrix)   ((matrix)->symmetric)

#define MatrixSize(matrix)        ((matrix)->size)

#define MatrixCommPkg(matrix)     ((matrix)->comm_pkg)


#endif
