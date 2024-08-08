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
* Routines for setting up communication packages.
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * ProjectRegion:
 *   RDF temporary until rewrite of CommRegFromStencil, SubtractSubregions,
 *   IntersectSubregions, ...
 *--------------------------------------------------------------------------*/

void     ProjectRegion(
                       Region *region,
                       int     sx,
                       int     sy,
                       int     sz,
                       int     ix,
                       int     iy,
                       int     iz)
{
  SubregionArray  *sr_array;

  Subregion       *subregion;

  int i, j;


  ForSubregionArrayI(i, region)
  {
    sr_array = RegionSubregionArray(region, i);

    ForSubregionI(j, sr_array)
    {
      subregion = SubregionArraySubregion(sr_array, j);
      if (!ProjectSubgrid(subregion, sx, sy, sz, ix, iy, iz))
      {
        DeleteSubregion(sr_array, j);
        j--;
      }
    }
  }

  return;
}

/*--------------------------------------------------------------------------
 * ProjectRBPoint:
 *   RDF temporary until rewrite of CommRegFromStencil, SubtractSubregions,
 *   IntersectSubregions, ...
 *--------------------------------------------------------------------------*/

Region  *ProjectRBPoint(
                        Region *region,
                        int     rb[4][3])
{
  Region          *new_region;

  Region          *tmp_reg;

  int i, j;


  new_region = NewRegion(RegionSize(region));

  for (i = 0; i < 4; i++)
  {
    tmp_reg = DuplicateRegion(region);
    ProjectRegion(tmp_reg, 2, 2, 2, rb[i][0], rb[i][1], rb[i][2]);
    ForSubregionArrayI(j, tmp_reg)
    {
      AppendSubregionArray(RegionSubregionArray(tmp_reg, j),
                           RegionSubregionArray(new_region, j));
      SubregionArraySize(RegionSubregionArray(tmp_reg, j)) = 0;
    }
    FreeRegion(tmp_reg);
  }

  return new_region;
}

/*--------------------------------------------------------------------------
 * CreateComputePkgs
 *--------------------------------------------------------------------------*/

void  CreateComputePkgs(
                        Grid *grid)
{
  SubgridArray  *subgrids = GridSubgrids(grid);

  Region   *send_reg;
  Region   *recv_reg;
  Region   *dep_reg;
  Region   *ind_reg;

  Region   *proj_send_reg;
  Region   *proj_recv_reg;
  Region   *proj_dep_reg;
  Region   *proj_ind_reg;

  int red[4][3] = { { 1, 0, 0 },
                    { 0, 1, 0 },
                    { 0, 0, 1 },
                    { 1, 1, 1 } };
  int black[4][3] = { { 0, 0, 0 },
                      { 1, 1, 0 },
                      { 1, 0, 1 },
                      { 0, 1, 1 } };

  Stencil  *update_all_stencil;
  Stencil  *update_all2_stencil;
  Stencil  *update_godunov_stencil;
  Stencil  *update_velz_stencil;
  Stencil  *update_pgs1_stencil;
  Stencil  *update_pgs2_stencil;
  Stencil  *update_pgs3_stencil;
  Stencil  *update_pgs4_stencil;

  int update_all_shape[][3] = { { -1, 0, 0 },
                                { 1, 0, 0 },
                                { 0, -1, 0 },
                                { 0, 1, 0 },
                                { 0, 0, -1 },
                                { 0, 0, 1 } };
  int update_all2_shape[][3] = { { -1, 0, 0 },
                                 { 1, 0, 0 },
                                 { -2, 0, 0 },
                                 { 2, 0, 0 },
                                 { 0, -1, 0 },
                                 { 0, 1, 0 },
                                 { 0, -2, 0 },
                                 { 0, 2, 0 },
                                 { 0, 0, -1 },
                                 { 0, 0, 1 },
                                 { 0, 0, -2 },
                                 { 0, 0, 2 } };
  int update_velz_shape[][3] = { { -1, 0, 0 },
                                 { 1, 0, 0 },
                                 { 0, -1, 0 },
                                 { 0, 1, 0 },
                                 { 0, 0, -1 },
                                 { 0, 0, 1 },
                                 { 0, 0, 2 },
                                 { 0, 0, -2 } };

  StencilElt *update_godunov_shape;
  StencilElt *update_pgs1_shape;
  StencilElt *update_pgs2_shape;
  StencilElt *update_pgs3_shape;
  StencilElt *update_pgs4_shape;
  int godunov_count, is, s_elt_index, i, j, k;
  int pgs_count;
  int s_elt;
  int start[3], stop[3], inc[3];


  /*------------------------------------------------------
   * Set up the update_godunov_shape using the
   * update_all_shape.
   *------------------------------------------------------*/

  update_godunov_shape = talloc(StencilElt, 135);

  /* set up center 27 point stencil coefficients */
  godunov_count = 0;
  for (k = -1; k != 2; k++)
    for (j = -1; j != 2; j++)
      for (i = -1; i != 2; i++)
      {
        update_godunov_shape[godunov_count][0] = i;
        update_godunov_shape[godunov_count][1] = j;
        update_godunov_shape[godunov_count][2] = k;
        godunov_count++;
      }

  /* set up remaining stencil coefficients */
  for (is = 0; is < 6; is++)
  {
    for (s_elt_index = 0; s_elt_index < 3; s_elt_index++)
    {
      if ((s_elt = update_all_shape[is][s_elt_index]))
      {
        start[s_elt_index] = 2 * s_elt;
        inc[s_elt_index] = s_elt;
        stop[s_elt_index] = 4 * s_elt;
      }
      else
      {
        start[s_elt_index] = -1;
        inc[s_elt_index] = 1;
        stop[s_elt_index] = 2;
      }
    }

    for (k = start[2]; k != stop[2]; k += inc[2])
      for (j = start[1]; j != stop[1]; j += inc[1])
        for (i = start[0]; i != stop[0]; i += inc[0])
        {
          update_godunov_shape[godunov_count][0] = i;
          update_godunov_shape[godunov_count][1] = j;
          update_godunov_shape[godunov_count][2] = k;
          godunov_count++;
        }
  }

  /*------------------------------------------------------
   * Set up the update_pgs#_shape using the
   * update_all_shape, where # = 1,2,3,4
   *------------------------------------------------------*/

  /* update_pgs1_shape */
  update_pgs1_shape = talloc(StencilElt, 27);
  pgs_count = 0;
  for (k = -1; k < 2; k++)
    for (j = -1; j < 2; j++)
      for (i = -1; i < 2; i++)
      {
        update_pgs1_shape[pgs_count][0] = i;
        update_pgs1_shape[pgs_count][1] = j;
        update_pgs1_shape[pgs_count][2] = k;
        pgs_count++;
      }

  /* update_pgs2_shape */
  update_pgs2_shape = talloc(StencilElt, 125);
  pgs_count = 0;
  for (k = -2; k < 3; k++)
    for (j = -2; j < 3; j++)
      for (i = -2; i < 3; i++)
      {
        update_pgs2_shape[pgs_count][0] = i;
        update_pgs2_shape[pgs_count][1] = j;
        update_pgs2_shape[pgs_count][2] = k;
        pgs_count++;
      }

  /* update_pgs3_shape */
  update_pgs3_shape = talloc(StencilElt, 343);
  pgs_count = 0;
  for (k = -3; k < 4; k++)
    for (j = -3; j < 4; j++)
      for (i = -3; i < 4; i++)
      {
        update_pgs3_shape[pgs_count][0] = i;
        update_pgs3_shape[pgs_count][1] = j;
        update_pgs3_shape[pgs_count][2] = k;
        pgs_count++;
      }

  /* update_pgs4_shape */
  update_pgs4_shape = talloc(StencilElt, 729);
  pgs_count = 0;
  for (k = -4; k < 5; k++)
    for (j = -4; j < 5; j++)
      for (i = -4; i < 5; i++)
      {
        update_pgs4_shape[pgs_count][0] = i;
        update_pgs4_shape[pgs_count][1] = j;
        update_pgs4_shape[pgs_count][2] = k;
        pgs_count++;
      }

  /*------------------------------------------------------
   * Set up the stencils describing the compute patterns
   * in the code.
   *------------------------------------------------------*/

  update_all_stencil = NewStencil(update_all_shape, 6);
  update_all2_stencil = NewStencil(update_all2_shape, 12);
  update_godunov_stencil = NewStencil(update_godunov_shape, 135);
  update_velz_stencil = NewStencil(update_velz_shape, 8);
  update_pgs1_stencil = NewStencil(update_pgs1_shape, 27);
  update_pgs2_stencil = NewStencil(update_pgs2_shape, 125);
  update_pgs3_stencil = NewStencil(update_pgs3_shape, 343);
  update_pgs4_stencil = NewStencil(update_pgs4_shape, 729);

  /*------------------------------------------------------
   * Malloc ComputePkg arrays.
   *------------------------------------------------------*/

  GridComputePkgs(grid) = talloc(ComputePkg *, NumUpdateModes);

  /*------------------------------------------------------
   * Define VectorUpdateAll mode.
   *------------------------------------------------------*/

  CommRegFromStencil(&send_reg, &recv_reg, grid, update_all_stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_all_stencil);

  GridComputePkg(grid, VectorUpdateAll) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /*------------------------------------------------------
   * Define VectorUpdateAll2 mode.
   *------------------------------------------------------*/

  CommRegFromStencil(&send_reg, &recv_reg, grid, update_all2_stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_all2_stencil);

  GridComputePkg(grid, VectorUpdateAll2) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /*------------------------------------------------------
   * Define VectorUpdateRPoint and VectorUpdateBPoint modes.
   *------------------------------------------------------*/

  CommRegFromStencil(&send_reg, &recv_reg, grid, update_all_stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_all_stencil);

  /* RDF temporary until rewrite of CommRegFromStencil */
  proj_send_reg = ProjectRBPoint(send_reg, black);
  proj_recv_reg = ProjectRBPoint(recv_reg, black);
  proj_dep_reg = ProjectRBPoint(dep_reg, red);
  proj_ind_reg = ProjectRBPoint(ind_reg, red);

  GridComputePkg(grid, VectorUpdateRPoint) =
    NewComputePkg(proj_send_reg, proj_recv_reg, proj_dep_reg, proj_ind_reg);

  /* RDF temporary until rewrite of CommRegFromStencil */
  proj_send_reg = ProjectRBPoint(send_reg, red);
  proj_recv_reg = ProjectRBPoint(recv_reg, red);
  proj_dep_reg = ProjectRBPoint(dep_reg, black);
  proj_ind_reg = ProjectRBPoint(ind_reg, black);

  GridComputePkg(grid, VectorUpdateBPoint) =
    NewComputePkg(proj_send_reg, proj_recv_reg, proj_dep_reg, proj_ind_reg);

  FreeRegion(send_reg);
  FreeRegion(recv_reg);
  FreeRegion(dep_reg);
  FreeRegion(ind_reg);

  /*------------------------------------------------------
   * Define VectorUpdateGodunov mode.
   *------------------------------------------------------*/

  CommRegFromStencil(&send_reg, &recv_reg, grid, update_godunov_stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_godunov_stencil);

  GridComputePkg(grid, VectorUpdateGodunov) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /*------------------------------------------------------
   * Define VectorUpdateVelZ mode.
   *------------------------------------------------------*/

  CommRegFromStencil(&send_reg, &recv_reg, grid, update_velz_stencil);

  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_velz_stencil);

  GridComputePkg(grid, VectorUpdateVelZ) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /*------------------------------------------------------
   * Define VectorUpdatePGS# modes.
   *------------------------------------------------------*/

  /* PGS1 */
  CommRegFromStencil(&send_reg, &recv_reg, grid, update_pgs1_stencil);
  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_pgs1_stencil);
  GridComputePkg(grid, VectorUpdatePGS1) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /* PGS2 */
  CommRegFromStencil(&send_reg, &recv_reg, grid, update_pgs2_stencil);
  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_pgs2_stencil);
  GridComputePkg(grid, VectorUpdatePGS2) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /* PGS3 */
  CommRegFromStencil(&send_reg, &recv_reg, grid, update_pgs3_stencil);
  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_pgs3_stencil);
  GridComputePkg(grid, VectorUpdatePGS3) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);

  /* PGS4 */
  CommRegFromStencil(&send_reg, &recv_reg, grid, update_pgs4_stencil);
  ComputeRegFromStencil(&dep_reg, &ind_reg,
                        subgrids, send_reg, recv_reg, update_pgs4_stencil);
  GridComputePkg(grid, VectorUpdatePGS4) =
    NewComputePkg(send_reg, recv_reg, dep_reg, ind_reg);


  /*------------------------------------------------------
   * Free up the stencils.
   *------------------------------------------------------*/

  tfree(update_godunov_shape);
  tfree(update_pgs1_shape);
  tfree(update_pgs2_shape);
  tfree(update_pgs3_shape);
  tfree(update_pgs4_shape);

  FreeStencil(update_all_stencil);
  FreeStencil(update_all2_stencil);
  FreeStencil(update_godunov_stencil);
  FreeStencil(update_velz_stencil);
  FreeStencil(update_pgs1_stencil);
  FreeStencil(update_pgs2_stencil);
  FreeStencil(update_pgs3_stencil);
  FreeStencil(update_pgs4_stencil);
}



/*--------------------------------------------------------------------------
 * FreeComputePkgs
 *--------------------------------------------------------------------------*/

void  FreeComputePkgs(
                      Grid *grid)
{
  int i;


  for (i = 0; i < NumUpdateModes; i++)
    FreeComputePkg(GridComputePkg(grid, i));
  tfree(GridComputePkgs(grid));
}
