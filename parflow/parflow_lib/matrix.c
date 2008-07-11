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
 * Constructors and destructors for matrix structure.
 *
 *****************************************************************************/

#include "parflow.h"

#include "matrix.h"


/*--------------------------------------------------------------------------
 * NewStencil
 *--------------------------------------------------------------------------*/

Stencil  *NewStencil(shape, sz)
int  shape[][3];
int  sz;
{
   Stencil     *new;
   StencilElt  *new_shape;

   int          i, j;


   new = talloc(Stencil, 1);
   StencilShape(new) = talloc(StencilElt, sz);

   new_shape = StencilShape(new);
   for (i = 0; i < sz; i++)
      for (j = 0; j < 3; j++)
         new_shape[i][j] = shape[i][j];

   StencilSize(new) = sz;

   return new;
}

/*--------------------------------------------------------------------------
 * NewMatrixUpdatePkg:
 *   RDF hack
 *--------------------------------------------------------------------------*/

CommPkg   *NewMatrixUpdatePkg(matrix, ghost)
Matrix    *matrix;
Stencil   *ghost;
{
   CommPkg     *new;

   Submatrix   *submatrix;

   Region      *send_reg, *recv_reg;

   int          ix, iy, iz;
   int          sx, sy, sz;

   int          n;


   submatrix = MatrixSubmatrix(matrix, 0);

   ix = SubmatrixIX(submatrix);
   iy = SubmatrixIY(submatrix);
   iz = SubmatrixIZ(submatrix);
   sx = SubmatrixSX(submatrix);
   sy = SubmatrixSY(submatrix);
   sz = SubmatrixSZ(submatrix);

   n = StencilSize(MatrixStencil(matrix));
   if (MatrixSymmetric(matrix))
      n = (n+1)/2;

   CommRegFromStencil(&send_reg, &recv_reg, MatrixGrid(matrix), ghost);
   ProjectRegion(send_reg, sx, sy, sz, ix, iy, iz);
   ProjectRegion(recv_reg, sx, sy, sz, ix, iy, iz);
   new = NewCommPkg(send_reg, recv_reg,
		    MatrixDataSpace(matrix), n, MatrixData(matrix));

   FreeRegion(send_reg);
   FreeRegion(recv_reg);

   return new;
}


/*--------------------------------------------------------------------------
 * InitMatrixUpdate
 *--------------------------------------------------------------------------*/

CommHandle  *InitMatrixUpdate(matrix)
Matrix      *matrix;
{
   return  InitCommunication(MatrixCommPkg(matrix));
}


/*--------------------------------------------------------------------------
 * FinalizeMatrixUpdate
 *--------------------------------------------------------------------------*/

void         FinalizeMatrixUpdate(handle)
CommHandle  *handle;
{
   FinalizeCommunication(handle);
}


/*--------------------------------------------------------------------------
 * NewMatrix
 *--------------------------------------------------------------------------*/

Matrix          *NewMatrix(grid, range, stencil, symmetry, ghost)
Grid            *grid;
SubregionArray  *range;
Stencil         *stencil;
int              symmetry;
Stencil         *ghost;
{
   Matrix         *new;
   Submatrix      *new_sub;

   StencilElt     *shape;

   Subregion      *subregion;
   Subregion      *new_subregion;

   double         *data;
   int            *data_index;

   int             data_size;

   int             xl, xu, yl, yu, zl, zu;

   int             i, j, k, n;
   int             nx, ny, nz;

   int            *symmetric_coeff;

   int            *data_stencil;
   int             data_stencil_size;


   shape = StencilShape(stencil);

   /*-----------------------------------------------------------------------
    * Set up symmetric_coeff.
    * This array is used to set up the data_index array in
    * the Submatrix structure.
    *-----------------------------------------------------------------------*/

   symmetric_coeff = ctalloc(int, StencilSize(stencil));

   if (symmetry)
   {
      for (k = 0; k < StencilSize(stencil); k++)
      {
	 n = 0;
	 if (shape[k][2] != 0)
	    n += ((shape[k][2]) < 0 ? -1 : 1) * 4;
	 if (shape[k][1] != 0)
	    n += ((shape[k][1]) < 0 ? -1 : 1) * 2;
	 if (shape[k][0] != 0)
	    n += ((shape[k][0]) < 0 ? -1 : 1) * 1;

	 /* if lower triangle, set index for symmetric coefficient */
	 if (n < 0)
	 {
	    for (n = 0; n < StencilSize(stencil); n++)
	    {
	       if (shape[n][0] == -shape[k][0] &&
		   shape[n][1] == -shape[k][1] &&
		   shape[n][0] == -shape[k][0])
	       {
		  symmetric_coeff[k] = n;
	       }
	    }
	 }
      }
   }

   /*-----------------------------------------------------------------------
    * Set up the stencil for the stored data.
    *-----------------------------------------------------------------------*/

   data_stencil_size = 0;
   for (k = 0; k < StencilSize(stencil); k++)
   {
      if (!symmetric_coeff[k])
	 data_stencil_size++;
   }

   data_stencil = ctalloc(int, 3*data_stencil_size);

   i = 0;
   for (k = 0; k < StencilSize(stencil); k++)
   {
      if (!symmetric_coeff[k])
      {
	 for (j = 0; j < 3; j++)
	 {
	    data_stencil[i] = shape[k][j];
	    i++;
	 }
      }
   }

   /*-----------------------------------------------------------------------
    * If (range == NULL), the range is the same as the domain
    *-----------------------------------------------------------------------*/

   if (range == NULL)
      range = (SubregionArray *) GridSubgrids(grid);

   /*-----------------------------------------------------------------------
    * If (ghost), compute ghost-layer size
    *-----------------------------------------------------------------------*/

   xl = xu = yl = yu = zl = zu = 0;
   if (ghost)
   {
      for (j = 0; j < StencilSize(ghost); j++)
      {
	 n = StencilShape(ghost)[j][0];
	 if (n < 0)
	    xl = max(xl, -n);
	 else
	    xu = max(xu,  n);

	 n = StencilShape(ghost)[j][1];
	 if (n < 0)
	    yl = max(yl, -n);
	 else
	    yu = max(yu,  n);

	 n = StencilShape(ghost)[j][2];
	 if (n < 0)
	    zl = max(zl, -n);
	 else
	    zu = max(zu,  n);
      }
   }

   /*-----------------------------------------------------------------------
    * Set up Matrix shell
    *-----------------------------------------------------------------------*/

   new = ctalloc(Matrix, 1);

   (new -> submatrices) = ctalloc(Submatrix *, GridNumSubgrids(grid));

   data_size = 0;
   MatrixDataSpace(new) = NewSubregionArray();
   ForSubgridI(i, GridSubgrids(grid))
   {
      new_sub = ctalloc(Submatrix, 1);

      data_index = ctalloc(int, StencilSize(stencil));

      subregion = SubregionArraySubregion(range, i);

      new_subregion = DuplicateSubregion(GridSubgrid(grid, i));
      if (SubgridNX(new_subregion) &&
	  SubgridNY(new_subregion) &&
	  SubgridNZ(new_subregion))
      {
	 SubgridIX(new_subregion) -= xl,
	 SubgridIY(new_subregion) -= yl,
	 SubgridIZ(new_subregion) -= zl,
	 SubgridNX(new_subregion) += xl + xu,
	 SubgridNY(new_subregion) += yl + yu,
	 SubgridNZ(new_subregion) += zl + zu,

	 ProjectSubgrid(new_subregion,
			SubregionSX(subregion),
			SubregionSY(subregion),
			SubregionSZ(subregion),
			SubregionIX(subregion),
			SubregionIY(subregion),
			SubregionIZ(subregion));
      }
      SubmatrixDataSpace(new_sub) = new_subregion;
      AppendSubregion(new_subregion, MatrixDataSpace(new));

      /*--------------------------------------------------------------------
       * Compute data_index array
       *--------------------------------------------------------------------*/

      nx = SubmatrixNX(new_sub);
      ny = SubmatrixNY(new_sub);
      nz = SubmatrixNZ(new_sub);
      n  = nx * ny * nz; 

      /* set pointers for upper triangle coefficients */
      for (k = 0; k < StencilSize(stencil); k++)
	 if (!symmetric_coeff[k])
	 {
	    data_index[k] = data_size;

	    data_size += n;
	 }

      /* set pointers for lower triangle coefficients */
      for (k = 0; k < StencilSize(stencil); k++)
	 if (symmetric_coeff[k])
	 {
	    data_index[k] =
	       data_index[symmetric_coeff[k]] +
	       (shape[k][2]*ny + shape[k][1])*nx + shape[k][0];
	 }

      (new_sub -> data_index) = data_index;

      MatrixSubmatrix(new, i) = new_sub;
   }

   (new -> data_size) = data_size;

   MatrixGrid(new)  = grid;
   MatrixRange(new) = range;

   MatrixStencil(new) = stencil;
   MatrixDataStencil(new) = data_stencil;
   MatrixDataStencilSize(new) = data_stencil_size;
   MatrixSymmetric(new) = symmetry;

   /* Compute total number of coefficients in matrix */
   MatrixSize(new) = GridSize(grid) * StencilSize(stencil);

   /*-----------------------------------------------------------------------
    * Set up Matrix data
    *-----------------------------------------------------------------------*/

   data = amps_CTAlloc(double, (data_size));
   MatrixData(new) = data;
   
   for (i = 0; i < GridNumSubgrids(grid); i++)
      SubmatrixData(MatrixSubmatrix(new, i)) = data;

   if (ghost)
      MatrixCommPkg(new) = NewMatrixUpdatePkg(new, ghost);
   else
      MatrixCommPkg(new) = NULL;

   /*-----------------------------------------------------------------------
    * Free up memory and return
    *-----------------------------------------------------------------------*/

   tfree(symmetric_coeff);

   return new;
}


/*--------------------------------------------------------------------------
 * FreeStencil
 *--------------------------------------------------------------------------*/

void      FreeStencil(stencil)
Stencil  *stencil;
{
   if (stencil)
   {
      tfree(StencilShape(stencil));
      tfree(stencil);
   }
}

/*--------------------------------------------------------------------------
 * FreeMatrix
 *--------------------------------------------------------------------------*/

void FreeMatrix(matrix)
Matrix *matrix;
{
   Submatrix  *submatrix;

   int         i;


   amps_TFree(matrix -> data);

   for(i = 0; i < GridNumSubgrids(MatrixGrid(matrix)); i++)
   {
      submatrix = MatrixSubmatrix(matrix, i);

      tfree(submatrix -> data_index);
      tfree(submatrix);
   }

   FreeStencil(MatrixStencil(matrix));
   tfree(MatrixDataStencil(matrix));

   FreeSubregionArray(MatrixDataSpace(matrix));

   if (MatrixCommPkg(matrix))
      FreeCommPkg(MatrixCommPkg(matrix));

   tfree(matrix -> submatrices);
   tfree(matrix);
}


/*--------------------------------------------------------------------------
 * InitMatrix
 *--------------------------------------------------------------------------*/

void    InitMatrix(A, value)
Matrix *A;
double  value;
{
   Grid       *grid = MatrixGrid(A);

   Submatrix  *A_sub;
   double     *Ap;
   int         im;

   SubgridArray  *subgrids;
   Subgrid       *subgrid;

   Stencil       *stencil;

   int         is, s;

   int         ix,   iy,   iz;
   int         nx,   ny,   nz;
   int         nx_m, ny_m, nz_m;

   int         i, j, k;


   subgrids = GridSubgrids(grid);
   ForSubgridI(is, subgrids)
   {
      subgrid = SubgridArraySubgrid(subgrids, is);

      A_sub = MatrixSubmatrix(A, is);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      nx_m = SubmatrixNX(A_sub);
      ny_m = SubmatrixNY(A_sub);
      nz_m = SubmatrixNZ(A_sub);

      stencil = MatrixStencil(A);
      for (s = 0; s < StencilSize(stencil); s++)
      {
	 Ap = SubmatrixElt(A_sub, s, ix, iy, iz);
	       
	 im  = 0;
	 BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
		   im, nx_m, ny_m, nz_m, 1, 1, 1,
		   {
		      Ap[im]  = value;
		   });
      }
   }
}
