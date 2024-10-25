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
* Constructors and destructors for matrix structure.
*
*****************************************************************************/

#include "parflow.h"
#include "matrix.h"

#ifdef HAVE_SAMRAI
#include "SAMRAI/hier/PatchDescriptor.h"
#include "SAMRAI/pdat/CellVariable.h"
#include "SAMRAI/pdat/SideVariable.h"

using namespace SAMRAI;

static int samrai_matrix_ids[4][2048];

#endif


/*--------------------------------------------------------------------------
 * NewStencil
 *--------------------------------------------------------------------------*/

Stencil  *NewStencil(
                     int shape[][3],
                     int sz)
{
  Stencil     *new_stencil;
  StencilElt  *new_shape;

  int i, j;


  new_stencil = talloc(Stencil, 1);
  StencilShape(new_stencil) = talloc(StencilElt, sz);

  new_shape = StencilShape(new_stencil);
  for (i = 0; i < sz; i++)
    for (j = 0; j < 3; j++)
      new_shape[i][j] = shape[i][j];

  StencilSize(new_stencil) = sz;

  return new_stencil;
}

/*--------------------------------------------------------------------------
 * NewMatrixUpdatePkg:
 *   RDF hack
 *--------------------------------------------------------------------------*/

CommPkg   *NewMatrixUpdatePkg(
                              Matrix * matrix,
                              Stencil *ghost)
{
  CommPkg     *new_commpkg = NULL;

  Grid* grid = MatrixGrid(matrix);

  if (GridNumSubgrids(grid) > 1)
  {
    PARFLOW_ERROR("NewMatrixUpdatePkg can't be used with number subgrids > 1");
  }
  else
  {
    Region      *send_reg, *recv_reg;

    Submatrix * submatrix = MatrixSubmatrix(matrix, 0);

    int ix = SubmatrixIX(submatrix);
    int iy = SubmatrixIY(submatrix);
    int iz = SubmatrixIZ(submatrix);
    int sx = SubmatrixSX(submatrix);
    int sy = SubmatrixSY(submatrix);
    int sz = SubmatrixSZ(submatrix);

    int n = StencilSize(MatrixStencil(matrix));
    if (MatrixSymmetric(matrix))
      n = (n + 1) / 2;

    CommRegFromStencil(&send_reg, &recv_reg, MatrixGrid(matrix), ghost);
    ProjectRegion(send_reg, sx, sy, sz, ix, iy, iz);
    ProjectRegion(recv_reg, sx, sy, sz, ix, iy, iz);


    new_commpkg = NewCommPkg(send_reg, recv_reg,
                             MatrixDataSpace(matrix), n, SubmatrixData(submatrix));

    FreeRegion(send_reg);
    FreeRegion(recv_reg);
  }

  return new_commpkg;
}


/*--------------------------------------------------------------------------
 * InitMatrixUpdate
 *--------------------------------------------------------------------------*/

CommHandle  *InitMatrixUpdate(
                              Matrix *matrix)
{
  CommHandle *return_handle = NULL;

  enum ParflowGridType grid_type = invalid_grid_type;

#ifdef HAVE_SAMRAI
  switch (matrix->type)
  {
    case matrix_cell_centered:
    {
      grid_type = flow_3D_grid_type;
      break;
    }

    case matrix_non_samrai:
    {
      grid_type = invalid_grid_type;
      break;
    }
  }
#endif

  if (grid_type == invalid_grid_type)
  {
    return_handle = InitCommunication(MatrixCommPkg(matrix));
  }
  else
  {
#ifdef HAVE_SAMRAI
    tbox::Dimension dim(GlobalsParflowSimulation->getDim(grid_type));
    if (matrix->boundary_fill_refine_algorithm.isNull())
    {
      tbox::Pointer < hier::PatchHierarchy > hierarchy(GlobalsParflowSimulation->getPatchHierarchy(grid_type));
      const int level_number = 0;

      matrix->boundary_fill_refine_algorithm = new xfer::RefineAlgorithm(dim);

      matrix->boundary_fill_refine_algorithm->registerRefine(
                                                             matrix->samrai_id,
                                                             matrix->samrai_id,
                                                             matrix->samrai_id,
                                                             tbox::Pointer < xfer::RefineOperator > (NULL));

      tbox::Pointer < hier::PatchLevel > level =
        hierarchy->getPatchLevel(level_number);

      matrix->boundary_fill_schedule = matrix->boundary_fill_refine_algorithm
                                       ->createSchedule(level,
                                                        level_number - 1,
                                                        hierarchy,
                                                        NULL);
    }
    const double time = 1.0;
    matrix->boundary_fill_schedule->fillData(time);
#endif
  }

  return return_handle;
}


/*--------------------------------------------------------------------------
 * FinalizeMatrixUpdate
 *--------------------------------------------------------------------------*/

void         FinalizeMatrixUpdate(
                                  CommHandle *handle)
{
  if (handle)
  {
    FinalizeCommunication(handle);
  }
}


/*--------------------------------------------------------------------------
 * NewMatrix
 *--------------------------------------------------------------------------*/

Matrix          *NewMatrix(
                           Grid *          grid,
                           SubregionArray *range,
                           Stencil *       stencil,
                           int             symmetry,
                           Stencil *       ghost)
{
  return NewMatrixType(grid, range, stencil, symmetry, ghost, matrix_non_samrai);
}

Matrix          *NewMatrixType(
                               Grid *           grid,
                               SubregionArray * range,
                               Stencil *        stencil,
                               int              symmetry,
                               Stencil *        ghost,
                               enum matrix_type type)
{
  Matrix         *new_matrix;
  Submatrix      *new_sub;

  StencilElt     *shape;

  Subregion      *subregion;
  Subregion      *new_subregion;

  double         *data;
  int            *data_index;

  int data_size;

  int xl, xu, yl, yu, zl, zu;

  int i, j, k, n;
  int nx, ny, nz;

  int            *symmetric_coeff;

  int            *data_stencil;
  int data_stencil_size;


  new_matrix = ctalloc(Matrix, 1);

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

  data_stencil = ctalloc(int, 3 * data_stencil_size);

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
    range = (SubregionArray*)GridSubgrids(grid);

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
        xl = pfmax(xl, -n);
      else
        xu = pfmax(xu, n);

      n = StencilShape(ghost)[j][1];
      if (n < 0)
        yl = pfmax(yl, -n);
      else
        yu = pfmax(yu, n);

      n = StencilShape(ghost)[j][2];
      if (n < 0)
        zl = pfmax(zl, -n);
      else
        zu = pfmax(zu, n);
    }
  }

#ifdef HAVE_SAMRAI
  enum ParflowGridType grid_type = invalid_grid_type;
  switch (type)
  {
    case matrix_cell_centered:
    {
      grid_type = flow_3D_grid_type;
      break;
    }

    case matrix_non_samrai:
    {
      grid_type = invalid_grid_type;
      break;
    }
  }

  tbox::Dimension dim(GlobalsParflowSimulation->getDim(grid_type));

  hier::IntVector ghosts(dim, xl);

  int index = 0;
  if (grid_type > 0)
  {
    for (int i = 0; i < 2048; i++)
    {
      if (samrai_matrix_ids[grid_type][i] == 0)
      {
        index = i;
        break;
      }
    }
  }

  std::string variable_name =
    "Matrix_" + tbox::Utilities::intToString(grid_type, 1) + "_" +
    tbox::Utilities::intToString(index, 4);

  tbox::Pointer < hier::Variable > variable;
#else
  type = matrix_non_samrai;
#endif

  new_matrix->type = type;


  // SAMRAI doesn't allow ghost width to be different along
  // the edges to get max.   This could be fixed with
  // a more specific SAMRAI data type for PF.
  switch (type)
  {
    case matrix_cell_centered:
    {
      xl = pfmax(xl, xu);
      xl = pfmax(xl, yl);
      xl = pfmax(xl, yu);
      xl = pfmax(xl, zl);
      xl = pfmax(xl, zu);
      xu = xl;
      yl = yu = xl;
      zl = zu = xl;
      break;
    }

    case matrix_non_samrai:
    {
      break;
    }
  }

  /*-----------------------------------------------------------------------
   * Set up Matrix shell
   *-----------------------------------------------------------------------*/



  (new_matrix->submatrices) = ctalloc(Submatrix *, GridNumSubgrids(grid));

  MatrixDataSpace(new_matrix) = NewSubregionArray();
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
    AppendSubregion(new_subregion, MatrixDataSpace(new_matrix));

    /*--------------------------------------------------------------------
     * Compute data_index array
     *--------------------------------------------------------------------*/

    nx = SubmatrixNX(new_sub);
    ny = SubmatrixNY(new_sub);
    nz = SubmatrixNZ(new_sub);
    n = nx * ny * nz;

    data_size = 0;
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
          (shape[k][2] * ny + shape[k][1]) * nx + shape[k][0];
      }

    (new_sub->data_index) = data_index;
    (new_sub->data_size) = data_size;

    MatrixSubmatrix(new_matrix, i) = new_sub;

    switch (type)
    {
#ifdef HAVE_SAMRAI
      case matrix_cell_centered:
      {
        // SAMRAI allocates
        break;
      }
#endif
      case matrix_non_samrai:
      {
        Submatrix *submatrix = MatrixSubmatrix(new_matrix, i);
        data = ctalloc_amps(double, submatrix->data_size);
        submatrix->allocated = TRUE;
        SubmatrixData(submatrix) = data;

        break;
      }

      default:
      {
        // SGS add abort
        fprintf(stderr, "Invalid matrix type\n");
      }
    }
  }

  MatrixGrid(new_matrix) = grid;
  MatrixRange(new_matrix) = range;

  MatrixStencil(new_matrix) = stencil;
  MatrixDataStencil(new_matrix) = data_stencil;
  MatrixDataStencilSize(new_matrix) = data_stencil_size;
  MatrixSymmetric(new_matrix) = symmetry;

  /* Compute total number of coefficients in matrix */
  MatrixSize(new_matrix) = GridSize(grid) * StencilSize(stencil);

  /*-----------------------------------------------------------------------
   * Set up Matrix data
   *-----------------------------------------------------------------------*/

  switch (type)
  {
#ifdef HAVE_SAMRAI
    case matrix_cell_centered:
    {
      variable = new pdat::CellVariable < double > (dim,
                                                    variable_name,
                                                    MatrixDataStencilSize(new_matrix));
      break;
    }
#endif
    case matrix_non_samrai:
    {
      // Allocated previously
      break;
    }

    default:
    {
      // SGS add abort
      fprintf(stderr, "Invalid matrix type\n");
    }
  }

  /* For SAMRAI vectors set the data pointer */
  switch (type)
  {
#ifdef HAVE_SAMRAI
    case matrix_cell_centered:
    {
      tbox::Pointer < hier::PatchHierarchy > hierarchy(GlobalsParflowSimulation->getPatchHierarchy(grid_type));
      tbox::Pointer < hier::PatchLevel > level(hierarchy->getPatchLevel(0));

      tbox::Pointer < hier::PatchDescriptor > patch_descriptor(hierarchy->getPatchDescriptor());

      new_matrix->samrai_id = patch_descriptor->definePatchDataComponent(
                                                                         variable_name,
                                                                         variable->getPatchDataFactory()->cloneFactory(ghosts));


      samrai_matrix_ids[grid_type][index] = new_matrix->samrai_id;
      new_matrix->table_index = index;

      std::cout << "samrai_id " << new_matrix->samrai_id << std::endl;

      level->allocatePatchData(new_matrix->samrai_id);

      int i = 0;
      for (hier::PatchLevel::Iterator patch_iterator(level);
           patch_iterator;
           patch_iterator++, i++)
      {
        hier::Patch *patch = *patch_iterator;

        const hier::Box patch_box = patch->getBox();

        std::cout << "In matrix box " << patch_box << std::endl;

        tbox::Pointer < pdat::CellData < double >> patch_data(
                                                              patch->getPatchData(new_matrix->samrai_id));
        Submatrix *submatrix = MatrixSubmatrix(new_matrix, i);
        SubmatrixData(submatrix) = patch_data->getPointer(0);
        patch_data->fillAll(0);

        MatrixCommPkg(new_matrix) = NULL;
      }
      break;
    }
#endif
    case matrix_non_samrai:
      if (ghost)
        MatrixCommPkg(new_matrix) = NewMatrixUpdatePkg(new_matrix, ghost);

      break;

    default:
      PARFLOW_ERROR("invalid matrix type");
      break;
  }

  /*-----------------------------------------------------------------------
   * Free up memory and return
   *-----------------------------------------------------------------------*/

  tfree(symmetric_coeff);

  return new_matrix;
}


/*--------------------------------------------------------------------------
 * FreeStencil
 *--------------------------------------------------------------------------*/

void      FreeStencil(
                      Stencil *stencil)
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

void FreeMatrix(
                Matrix *matrix)
{
  Submatrix  *submatrix;

  int i;

  switch (matrix->type)
  {
#ifdef HAVE_SAMRAI
    case matrix_cell_centered:
    {
      std::cout << "freeing samrai_id " << matrix->samrai_id << std::endl;

      ParflowGridType grid_type = flow_3D_grid_type;
      if (!matrix->boundary_fill_refine_algorithm.isNull())
      {
        matrix->boundary_fill_refine_algorithm.setNull();
        matrix->boundary_fill_schedule.setNull();
      }

      tbox::Pointer < hier::PatchHierarchy > hierarchy(GlobalsParflowSimulation->getPatchHierarchy(grid_type));
      tbox::Pointer < hier::PatchLevel > level(hierarchy->getPatchLevel(0));

      level->deallocatePatchData(matrix->samrai_id);

      tbox::Pointer < hier::PatchDescriptor > patch_descriptor(hierarchy->getPatchDescriptor());
      patch_descriptor->removePatchDataComponent(matrix->samrai_id);

      samrai_matrix_ids[grid_type][matrix->table_index] = 0;
      break;
    }
#endif
    case matrix_non_samrai:
    {
      break;
    }

    default:
      fprintf(stderr, "Invalid matrix type\n");
  }

  for (i = 0; i < GridNumSubgrids(MatrixGrid(matrix)); i++)
  {
    submatrix = MatrixSubmatrix(matrix, i);

    if (submatrix->allocated)
    {
      tfree_amps(submatrix->data);
    }

    tfree(submatrix->data_index);
    tfree(submatrix);
  }

  FreeStencil(MatrixStencil(matrix));
  tfree(MatrixDataStencil(matrix));

  FreeSubregionArray(MatrixDataSpace(matrix));

  if (MatrixCommPkg(matrix))
    FreeCommPkg(MatrixCommPkg(matrix));

  tfree(matrix->submatrices);
  tfree(matrix);
}


/*--------------------------------------------------------------------------
 * InitMatrix
 *--------------------------------------------------------------------------*/

void    InitMatrix(
                   Matrix *A,
                   double  value)
{
  Grid       *grid = MatrixGrid(A);

  Submatrix  *A_sub;
  double     *Ap;
  int im;

  SubgridArray  *subgrids;
  Subgrid       *subgrid;

  Stencil       *stencil;

  int is, s;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_m, ny_m, nz_m;

  int i, j, k;


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

      im = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                im, nx_m, ny_m, nz_m, 1, 1, 1,
      {
        Ap[im] = value;
      });
    }
  }
}
