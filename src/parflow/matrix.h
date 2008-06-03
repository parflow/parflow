/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for Matrix data structures
 *
 *****************************************************************************/

#ifndef _MATRIX_HEADER
#define _MATRIX_HEADER

#include "grid.h"


/*--------------------------------------------------------------------------
 * Stencil
 *--------------------------------------------------------------------------*/

typedef int StencilElt[3];

typedef struct
{
   StencilElt  *shape;   /* Description of a stencil's shape */
   int          size;    /* Number of stencil coefficients */

} Stencil;

/*--------------------------------------------------------------------------
 * Submatrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   double    *data;            /* Pointer to Matrix data */
   int       *data_index;      /* array of indices corresponding to
				  stencil coefficient starting positions
				  in data array */

   Subregion *data_space;

} Submatrix;

/*--------------------------------------------------------------------------
 * Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   Submatrix       **submatrices;  /* Array of pointers to submatrices */

   double           *data;         /* Pointer to Matrix data */
   int               data_size;    /* Size of data */

   Grid             *grid;         /* Matrix domain */
   SubregionArray   *range;        /* Matrix range */

   SubregionArray   *data_space;   /* Description of Matrix data */

   Stencil          *stencil;      /* Matrix stencil */

   int              *data_stencil;      /* Stencil for the stored data */
   int               data_stencil_size; /* Stencil size for the stored data */

   int               symmetric;    /* Is matrix symmetric? */

   int               size;         /* Total number of nonzero coefficients */

   CommPkg          *comm_pkg;     /* Information on how to update boundary */

} Matrix;


/*--------------------------------------------------------------------------
 * Accessor functions for the Stencil structure
 *--------------------------------------------------------------------------*/

#define StencilShape(sg) ((sg) -> shape)
#define StencilSize(sg)  ((sg) -> size)

/*--------------------------------------------------------------------------
 * Accessor functions for the Submatrix structure
 *--------------------------------------------------------------------------*/

#define SubmatrixData(submatrix) ((submatrix) -> data)
#define SubmatrixStencilData(submatrix, s) \
(((submatrix) -> data) + ((submatrix) -> data_index[s]))

#define SubmatrixDataSpace(submatrix)  ((submatrix) -> data_space)

#define SubmatrixIX(submatrix)   (SubregionIX(SubmatrixDataSpace(submatrix)))
#define SubmatrixIY(submatrix)   (SubregionIY(SubmatrixDataSpace(submatrix)))
#define SubmatrixIZ(submatrix)   (SubregionIZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixNX(submatrix)   (SubregionNX(SubmatrixDataSpace(submatrix)))
#define SubmatrixNY(submatrix)   (SubregionNY(SubmatrixDataSpace(submatrix)))
#define SubmatrixNZ(submatrix)   (SubregionNZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixSX(submatrix)   (SubregionSX(SubmatrixDataSpace(submatrix)))
#define SubmatrixSY(submatrix)   (SubregionSY(SubmatrixDataSpace(submatrix)))
#define SubmatrixSZ(submatrix)   (SubregionSZ(SubmatrixDataSpace(submatrix)))

#define SubmatrixEltIndex(submatrix, x, y, z) \
(((x) - SubmatrixIX(submatrix))/SubmatrixSX(submatrix) + \
 (((y) - SubmatrixIY(submatrix))/SubmatrixSY(submatrix) + \
  (((z) - SubmatrixIZ(submatrix))/SubmatrixSZ(submatrix)) * \
  SubmatrixNY(submatrix)) * \
 SubmatrixNX(submatrix))

#define SubmatrixElt(submatrix, s, x, y, z) \
(SubmatrixStencilData(submatrix, s) + SubmatrixEltIndex(submatrix, x, y, z))

/*--------------------------------------------------------------------------
 * Accessor functions for the Matrix structure
 *--------------------------------------------------------------------------*/

#define MatrixSubmatrix(matrix,n) ((matrix) -> submatrices[(n)])

#define MatrixData(matrix)        ((matrix) -> data)

#define MatrixGrid(matrix)        ((matrix) -> grid)
#define MatrixRange(matrix)       ((matrix) -> range)

#define MatrixDataSpace(matrix)   ((matrix)-> data_space)

#define MatrixStencil(matrix)     ((matrix) -> stencil)

#define MatrixDataStencil(matrix)     ((matrix) -> data_stencil)
#define MatrixDataStencilSize(matrix) ((matrix) -> data_stencil_size)

#define MatrixSymmetric(matrix)   ((matrix) -> symmetric)

#define MatrixSize(matrix)        ((matrix) -> size)

#define MatrixCommPkg(matrix)     ((matrix) -> comm_pkg)


#endif
