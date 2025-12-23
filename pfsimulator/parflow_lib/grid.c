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
* Member functions for Grid class.
*
*****************************************************************************/

#include "parflow.h"
#include "grid.h"

#include <math.h>

/*--------------------------------------------------------------------------
 * NewGrid
 *--------------------------------------------------------------------------*/

Grid  *NewGrid(
               SubgridArray *subgrids,
               SubgridArray *all_subgrids)
{
  Grid    *new_grid;

  Subgrid *s;

  int i, size;

  int ix = INT_MAX;
  int iy = INT_MAX;
  int iz = INT_MAX;
  int nx = INT_MIN;
  int ny = INT_MIN;
  int nz = INT_MIN;

  new_grid = talloc(Grid, 1);

  new_grid->subgrids = subgrids;
  new_grid->all_subgrids = all_subgrids;

  size = 0;
  for (i = 0; i < SubgridArraySize(all_subgrids); i++)
  {
    s = SubgridArraySubgrid(all_subgrids, i);
    size += (s->nx) * (s->ny) * (s->nz);

    if (s->ix < ix)
    {
      ix = s->ix;
    }

    if (s->iy < iy)
    {
      iy = s->iy;
    }

    if (s->iz < iz)
    {
      iz = s->iz;
    }

    if ((s->ix + s->nx) > nx)
    {
      nx = s->ix + s->nx;
    }

    if ((s->iy + s->ny) > ny)
    {
      ny = s->iy + s->ny;
    }

    if ((s->iz + s->nz) > nz)
    {
      nz = s->iz + s->nz;
    }
  }

  new_grid->background = NewSubgrid(ix, iy, iz, nx, ny, nz, 1, 1, 1, 0);

  new_grid->size = size;

  new_grid->compute_pkgs = NULL;

  return new_grid;
}


/*--------------------------------------------------------------------------
 * FreeGrid
 *--------------------------------------------------------------------------*/

void  FreeGrid(
               Grid *grid)
{
  if (grid)
  {
    if (grid->background)
    {
      FreeSubgrid(grid->background);
    }

    FreeSubgridArray(GridAllSubgrids(grid));

    /* these subgrid arrays point to subgrids in all_subgrids */
    SubgridArraySize(GridSubgrids(grid)) = 0;
    FreeSubgridArray(GridSubgrids(grid));

    if (GridComputePkgs(grid))
      FreeComputePkgs(grid);

    tfree(grid);
  }
}


/*--------------------------------------------------------------------------
 * ProjectSubgrid:
 *   Projects a subgrid onto an "index region".  This index region is the
 *   collection of index space indices with strides (sx, sy, sz) that
 *   contains the index (ix, iy, iz).  The base index space is determined
 *   by the subgrid passed in.
 *
 *   Returns 0 for an empty projection, and 1 for a nonempty projection.
 *--------------------------------------------------------------------------*/

int         ProjectSubgrid(
                           Subgrid *subgrid,
                           int      sx,
                           int      sy,
                           int      sz,
                           int      ix,
                           int      iy,
                           int      iz)
{
  int il, iu;


  /*------------------------------------------------------
   * set the strides
   *------------------------------------------------------*/

  subgrid->sx = sx;
  subgrid->sy = sy;
  subgrid->sz = sz;

  /*------------------------------------------------------
   * project in x
   *------------------------------------------------------*/

  il = SubgridIX(subgrid) + ix;
  iu = il + SubgridNX(subgrid);

  il = ((int)((il + (sx - 1)) / sx)) * sx - ix;
  iu = ((int)((iu + (sx - 1)) / sx)) * sx - ix;

  subgrid->ix = il;
  subgrid->nx = (iu - il) / sx;

  /*------------------------------------------------------
   * project in y
   *------------------------------------------------------*/

  il = SubgridIY(subgrid) + iy;
  iu = il + SubgridNY(subgrid);

  il = ((int)((il + (sy - 1)) / sy)) * sy - iy;
  iu = ((int)((iu + (sy - 1)) / sy)) * sy - iy;

  subgrid->iy = il;
  subgrid->ny = (iu - il) / sy;

  /*------------------------------------------------------
   * project in z
   *------------------------------------------------------*/

  il = SubgridIZ(subgrid) + iz;
  iu = il + SubgridNZ(subgrid);

  il = ((int)((il + (sz - 1)) / sz)) * sz - iz;
  iu = ((int)((iu + (sz - 1)) / sz)) * sz - iz;

  subgrid->iz = il;
  subgrid->nz = (iu - il) / sz;

  /*------------------------------------------------------
   * return
   *------------------------------------------------------*/

  if (((subgrid->nx) > 0) &&
      ((subgrid->ny) > 0) &&
      ((subgrid->nz) > 0))
  {
    return 1;
  }
  else
  {
    subgrid->nx = 0;
    subgrid->ny = 0;
    subgrid->nz = 0;

    return 0;
  }
}


/*--------------------------------------------------------------------------
 * ConvertToSubgrid:
 *   Converts a subregion to a subgrid, if possible.
 *
 *   Note: The subregion passed in is modified.
 *--------------------------------------------------------------------------*/

Subgrid    *ConvertToSubgrid(
                             Subregion *subregion)
{
  int sx, sy, sz;
  int ex, ey, ez;

  sx = SubregionSX(subregion);
  sy = SubregionSY(subregion);
  sz = SubregionSZ(subregion);

  /* check that (x, y, z) coords are divisible by strides */
  if ((SubregionIX(subregion) % sx) ||
      (SubregionIY(subregion) % sy) ||
      (SubregionIZ(subregion) % sz))
  {
    return NULL;
  }

  /* get Exp2 of (sx, sy, sz); return NULL if not a power of 2 */
  if ((ex = Exp2(sx)) < 0)
    return NULL;
  if ((ey = Exp2(sy)) < 0)
    return NULL;
  if ((ez = Exp2(sz)) < 0)
    return NULL;

  /* modify the subregion */
  SubregionIX(subregion) /= sx;
  SubregionIY(subregion) /= sy;
  SubregionIZ(subregion) /= sz;
  SubregionSX(subregion) = 1;
  SubregionSY(subregion) = 1;
  SubregionSZ(subregion) = 1;
  SubregionRX(subregion) -= ex;
  SubregionRY(subregion) -= ey;
  SubregionRZ(subregion) -= ez;
  SubregionLevel(subregion) -= ex + ey + ez;

  return((Subgrid*)subregion);
}


/*--------------------------------------------------------------------------
 * ExtractSubgrid: RDF todo
 *   Extract a subgrid with resolution (rx, ry, rz).
 *   Assumes that (rx, ry, rz) are not finer than subgrid's resolution.
 *--------------------------------------------------------------------------*/

Subgrid  *ExtractSubgrid(
                         int      rx,
                         int      ry,
                         int      rz,
                         Subgrid *subgrid)
{
  Subgrid  *new_subgrid;

  int ix, iy, iz;
  int nx, ny, nz;

  int d, t;


  if ((d = SubgridRX(subgrid) - rx) > 0)
  {
    d = (int)pow(2.0, (double)d);

    t = SubgridIX(subgrid);
    ix = ((int)((t + (d - 1)) / d));

    t += SubgridNX(subgrid);
    nx = ((int)((t + (d - 1)) / d)) - ix;
  }
  else
  {
    ix = SubgridIX(subgrid);
    nx = SubgridNX(subgrid);
  }

  if ((d = SubgridRY(subgrid) - ry) > 0)
  {
    d = (int)pow(2.0, (double)d);

    t = SubgridIY(subgrid);
    iy = ((int)((t + (d - 1)) / d));

    t += SubgridNY(subgrid);
    ny = ((int)((t + (d - 1)) / d)) - iy;
  }
  else
  {
    iy = SubgridIY(subgrid);
    ny = SubgridNY(subgrid);
  }

  if ((d = SubgridRZ(subgrid) - rz) > 0)
  {
    d = (int)pow(2.0, (double)d);

    t = SubgridIZ(subgrid);
    iz = ((int)((t + (d - 1)) / d));

    t += SubgridNZ(subgrid);
    nz = ((int)((t + (d - 1)) / d)) - iz;
  }
  else
  {
    iz = SubgridIZ(subgrid);
    nz = SubgridNZ(subgrid);
  }

  new_subgrid = NewSubgrid(ix, iy, iz, nx, ny, nz, rx, ry, rz,
                           SubgridProcess(subgrid));

  return new_subgrid;
}


/*--------------------------------------------------------------------------
 * IntersectSubgrids: RDF todo
 *   For subgrids S_1 = (xl_1, yl_1, zl_1) X (xu_1, yu_1, zu_1) and
 *                S_2 = (xl_2, yl_2, zl_2) X (xu_2, yu_2, zu_2) in the same
 *   index space (i.e. same resolution background grid),
 *   S = S_1 * S_2 = (pfmax(xl_i), pfmax(yl_i), pfmax(zl_i)) X
 *                   (pfmin(xu_i), pfmin(yu_i), pfmin(zu_i))
 *
 *   This routine assumes that rs_i >= rs_j, for all s = x, y, z, where
 *   (i,j) = (1,2) or (2,1).  i.e. one of the subgrids passed in has
 *   resolution either finer than or equal to the other subgrid (in all
 *   three directions).
 *--------------------------------------------------------------------------*/

Subgrid  *IntersectSubgrids(
                            Subgrid *subgrid1,
                            Subgrid *subgrid2)
{
  Subgrid  *new_subgrid, *old;


  if (SubgridLevel(subgrid2) > SubgridLevel(subgrid1))
  {
    old = subgrid1;
    new_subgrid = ExtractSubgrid(SubgridRX(subgrid1),
                                 SubgridRY(subgrid1),
                                 SubgridRZ(subgrid1),
                                 subgrid2);
  }
  else
  {
    old = subgrid2;
    new_subgrid = ExtractSubgrid(SubgridRX(subgrid2),
                                 SubgridRY(subgrid2),
                                 SubgridRZ(subgrid2),
                                 subgrid1);
  }

  /* find x bounds */
  SubgridNX(new_subgrid) = pfmin(SubgridIX(new_subgrid) + SubgridNX(new_subgrid),
                                 SubgridIX(old) + SubgridNX(old));
  SubgridIX(new_subgrid) = pfmax(SubgridIX(new_subgrid), SubgridIX(old));
  if (SubgridNX(new_subgrid) > SubgridIX(new_subgrid))
    SubgridNX(new_subgrid) -= SubgridIX(new_subgrid);
  else
    goto empty;

  /* find y bounds */
  SubgridNY(new_subgrid) = pfmin(SubgridIY(new_subgrid) + SubgridNY(new_subgrid),
                                 SubgridIY(old) + SubgridNY(old));
  SubgridIY(new_subgrid) = pfmax(SubgridIY(new_subgrid), SubgridIY(old));
  if (SubgridNY(new_subgrid) > SubgridIY(new_subgrid))
    SubgridNY(new_subgrid) -= SubgridIY(new_subgrid);
  else
    goto empty;

  /* find z bounds */
  SubgridNZ(new_subgrid) = pfmin(SubgridIZ(new_subgrid) + SubgridNZ(new_subgrid),
                                 SubgridIZ(old) + SubgridNZ(old));
  SubgridIZ(new_subgrid) = pfmax(SubgridIZ(new_subgrid), SubgridIZ(old));
  if (SubgridNZ(new_subgrid) > SubgridIZ(new_subgrid))
    SubgridNZ(new_subgrid) -= SubgridIZ(new_subgrid);
  else
    goto empty;

  /* return intersection */
  return new_subgrid;

empty:
  FreeSubgrid(new_subgrid);
  return NULL;
}


/*--------------------------------------------------------------------------
 * SubtractSubgrids: RDF todo
 *   Compute S_1 - S_2.
 *
 *   This routine assumes that rs_2 >= rs_1, for all s = x, y, z.
 *   i.e. S_2 has resolution either finer than or equal to S_1 (in all
 *   three directions).
 *--------------------------------------------------------------------------*/

SubgridArray  *SubtractSubgrids(
                                Subgrid *subgrid1,
                                Subgrid *subgrid2)
{
  SubgridArray  *new_a;

  Subgrid       *s, *new_s;

  int d, t;
  int cx0, cx1, cy0, cy1, cz0, cz1;


  s = DuplicateSubgrid(subgrid1);

  /*------------------------------------------------------
   * get cut points in x
   *------------------------------------------------------*/

  if ((d = (SubgridRX(subgrid2) - SubgridRX(subgrid1))))
  {
    d = (int)pow(2.0, (double)d);
    t = SubgridIX(subgrid2);
    cx0 = ((int)((t + (d - 1)) / d));

    t += SubgridNX(subgrid2);
    cx1 = ((int)((t + (d - 1)) / d));
  }
  else
  {
    cx0 = SubgridIX(subgrid2);
    cx1 = cx0 + SubgridNX(subgrid2);
  }
  if ((cx0 >= SubgridIX(s) + SubgridNX(s)) || (cx1 <= SubgridIX(s)))
    goto all;

  /*------------------------------------------------------
   * get cut points in y
   *------------------------------------------------------*/

  if ((d = (SubgridRY(subgrid2) - SubgridRY(subgrid1))))
  {
    d = (int)pow(2.0, (double)d);
    t = SubgridIY(subgrid2);
    cy0 = ((int)((t + (d - 1)) / d));

    t += SubgridNY(subgrid2);
    cy1 = ((int)((t + (d - 1)) / d));
  }
  else
  {
    cy0 = SubgridIY(subgrid2);
    cy1 = cy0 + SubgridNY(subgrid2);
  }
  if ((cy0 >= SubgridIY(s) + SubgridNY(s)) || (cy1 <= SubgridIY(s)))
    goto all;

  /*------------------------------------------------------
   * get cut points in z
   *------------------------------------------------------*/

  if ((d = (SubgridRZ(subgrid2) - SubgridRZ(subgrid1))))
  {
    d = (int)pow(2.0, (double)d);
    t = SubgridIZ(subgrid2);
    cz0 = ((int)((t + (d - 1)) / d));

    t += SubgridNZ(subgrid2);
    cz1 = ((int)((t + (d - 1)) / d));
  }
  else
  {
    cz0 = SubgridIZ(subgrid2);
    cz1 = cz0 + SubgridNZ(subgrid2);
  }
  if ((cz0 >= SubgridIZ(s) + SubgridNZ(s)) || (cz1 <= SubgridIZ(s)))
    goto all;

  /*------------------------------------------------------
   * create SubgridArray
   *------------------------------------------------------*/

  new_a = NewSubgridArray();

  /*------------------------------------------------------
   * cut s in x
   *------------------------------------------------------*/

  if ((cx0 > SubgridIX(s)) && (cx0 < SubgridIX(s) + SubgridNX(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNX(new_s) = cx0 - SubgridIX(s);
    AppendSubgrid(new_s, new_a);

    SubgridNX(s) = SubgridIX(s) + SubgridNX(s) - cx0;
    SubgridIX(s) = cx0;
  }
  if ((cx1 > SubgridIX(s)) && (cx1 < SubgridIX(s) + SubgridNX(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNX(new_s) = SubgridIX(s) + SubgridNX(s) - cx1;
    SubgridIX(new_s) = cx1;
    AppendSubgrid(new_s, new_a);

    SubgridNX(s) = cx1 - SubgridIX(s);
  }

  /*------------------------------------------------------
   * cut s in y
   *------------------------------------------------------*/

  if ((cy0 > SubgridIY(s)) && (cy0 < SubgridIY(s) + SubgridNY(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNY(new_s) = cy0 - SubgridIY(new_s);
    AppendSubgrid(new_s, new_a);

    SubgridNY(s) = SubgridIY(s) + SubgridNY(s) - cy0;
    SubgridIY(s) = cy0;
  }
  if ((cy1 > SubgridIY(s)) && (cy1 < SubgridIY(s) + SubgridNY(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNY(new_s) = SubgridIY(s) + SubgridNY(s) - cy1;
    SubgridIY(new_s) = cy1;
    AppendSubgrid(new_s, new_a);

    SubgridNY(s) = cy1 - SubgridIY(s);
  }

  /*------------------------------------------------------
   * cut s in z
   *------------------------------------------------------*/

  if ((cz0 > SubgridIZ(s)) && (cz0 < SubgridIZ(s) + SubgridNZ(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNZ(new_s) = cz0 - SubgridIZ(new_s);
    AppendSubgrid(new_s, new_a);

    SubgridNZ(s) = SubgridIZ(s) + SubgridNZ(s) - cz0;
    SubgridIZ(s) = cz0;
  }
  if ((cz1 > SubgridIZ(s)) && (cz1 < SubgridIZ(s) + SubgridNZ(s)))
  {
    new_s = DuplicateSubgrid(s);
    SubgridNZ(new_s) = SubgridIZ(s) + SubgridNZ(s) - cz1;
    SubgridIZ(new_s) = cz1;
    AppendSubgrid(new_s, new_a);

    SubgridNZ(s) = cz1 - SubgridIZ(s);
  }

  FreeSubgrid(s);
  return new_a;

all:
  new_a = NewSubgridArray();
  AppendSubgrid(s, new_a);
  return new_a;
}


/*--------------------------------------------------------------------------
 * UnionSubgridArray: RDF todo
 *   Compute the union of all S_i.
 *
 * Note: This routine ignores process numbers.  All process numbers in
 * the resulting union are currently set to 0.
 *--------------------------------------------------------------------------*/

SubgridArray  *UnionSubgridArray(
                                 SubgridArray *subgrids)
{
  SubgridArray  *new_sa;
  SubgridArray  *old_sa;
  SubgridArray  *tmp_sa0, *tmp_sa1;

  Subgrid       *subgrid;

  int           *level_array;
  int           *tmp_level_array;
  int num_levels;

  int           *block_index[3];
  int block_sz[3];
  int factor[3];
  int           *block;
  int index;

  int ibox[3][2];
  int l, si, sj, d, i, j, k;

  int rx, ry, rz;

  int join;

  int i_tmp0 = 0, i_tmp1 = 0;
  double d_tmp;


  /*---------------------------------------------------------
   * Set up sorted level_array
   *---------------------------------------------------------*/

  num_levels = 0;
  level_array = NULL;
  ForSubgridI(si, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, si);

    for (i = 0; i < num_levels; i++)
      if (SubgridLevel(subgrid) == level_array[i])
        break;
    if (i == num_levels)
    {
      num_levels++;

      tmp_level_array = level_array;
      level_array = talloc(int, num_levels);

      for (i = 0; i < (num_levels - 1); i++)
        if (SubgridLevel(subgrid) < tmp_level_array[i])
          break;
        else
          level_array[i] = tmp_level_array[i];
      level_array[i++] = SubgridLevel(subgrid);
      for (; i < num_levels; i++)
        level_array[i] = tmp_level_array[i - 1];

      tfree(tmp_level_array);
    }
  }

  /*---------------------------------------------------------
   * Set up new_sa
   *---------------------------------------------------------*/

  new_sa = NewSubgridArray();

  for (l = num_levels; l--;)
  {
    old_sa = new_sa;

    /*------------------------------------------------------
     * Put current level subgrids into new_sa and set rx, ry, rz
     *------------------------------------------------------*/

    new_sa = NewSubgridArray();

    ForSubgridI(si, subgrids)
    {
      subgrid = SubgridArraySubgrid(subgrids, si);

      if (SubgridLevel(subgrid) == level_array[l])
        AppendSubgrid(DuplicateSubgrid(subgrid), new_sa);
    }

    subgrid = SubgridArraySubgrid(new_sa, 0);
    rx = SubgridRX(subgrid);
    ry = SubgridRY(subgrid);
    rz = SubgridRZ(subgrid);

    /*------------------------------------------------------
     * Subtract old_sa from new_sa
     *------------------------------------------------------*/

    ForSubgridI(si, old_sa)
    {
      tmp_sa0 = NewSubgridArray();

      ForSubgridI(sj, new_sa)
      {
        tmp_sa1 = SubtractSubgrids(SubgridArraySubgrid(new_sa, sj),
                                   SubgridArraySubgrid(old_sa, si));

        AppendSubgridArray(tmp_sa1, tmp_sa0);
        SubgridArraySize(tmp_sa1) = 0;
        FreeSubgridArray(tmp_sa1);
      }

      FreeSubgridArray(new_sa);
      new_sa = tmp_sa0;
    }

    /*------------------------------------------------------
     * Set up block_index array with new_sa subgrid indices
     *------------------------------------------------------*/

    for (d = 0; d < 3; d++)
    {
      block_index[d] = talloc(int, 2 * SubgridArraySize(new_sa));
      block_sz[d] = 0;
    }

    ForSubgridI(si, new_sa)
    {
      subgrid = SubgridArraySubgrid(new_sa, si);

      ibox[0][0] = SubgridIX(subgrid);
      ibox[0][1] = ibox[0][0] + SubgridNX(subgrid);
      ibox[1][0] = SubgridIY(subgrid);
      ibox[1][1] = ibox[1][0] + SubgridNY(subgrid);
      ibox[2][0] = SubgridIZ(subgrid);
      ibox[2][1] = ibox[2][0] + SubgridNZ(subgrid);

      for (d = 0; d < 3; d++)
      {
        for (i = 0; i < 2; i++)
        {
          for (j = 0; j < block_sz[d]; j++)
            if (ibox[d][i] == block_index[d][j])
              break;
          if (j == block_sz[d])
          {
            block_index[d][j] = ibox[d][i];
            block_sz[d]++;
          }
        }
      }
    }
    for (d = 0; d < 3; d++)
      block_sz[d]--;

    /*------------------------------------------------------
     * Set factor values
     *------------------------------------------------------*/

    factor[0] = 1;
    factor[1] = (block_sz[0] + 1);
    factor[2] = (block_sz[1] + 1) * factor[1];

    /*------------------------------------------------------
     * Sort block_index array (simple bubble sort)
     *------------------------------------------------------*/

    for (d = 0; d < 3; d++)
    {
      for (i = block_sz[d]; i > 0; i--)
        for (j = 0; j < i; j++)
          if (block_index[d][j] > block_index[d][j + 1])
          {
            d_tmp = block_index[d][j];
            block_index[d][j] = block_index[d][j + 1];
            block_index[d][j + 1] = (int)d_tmp;
          }
    }

    /*------------------------------------------------------
     * Set up 3-D block array
     *------------------------------------------------------*/

    block = ctalloc(int, (block_sz[0] * block_sz[1] * block_sz[2]));

    ForSubgridI(si, new_sa)
    {
      subgrid = SubgridArraySubgrid(new_sa, si);

      ibox[0][0] = SubgridIX(subgrid);
      ibox[0][1] = ibox[0][0] + SubgridNX(subgrid);
      ibox[1][0] = SubgridIY(subgrid);
      ibox[1][1] = ibox[1][0] + SubgridNY(subgrid);
      ibox[2][0] = SubgridIZ(subgrid);
      ibox[2][1] = ibox[2][0] + SubgridNZ(subgrid);

      for (d = 0; d < 3; d++)
      {
        j = 0;
        for (i = 0; i < 2; i++)
        {
          while (ibox[d][i] > block_index[d][j])
            j++;
          ibox[d][i] = j;
        }
      }

      for (k = ibox[2][0]; k < ibox[2][1]; k++)
        for (j = ibox[1][0]; j < ibox[1][1]; j++)
          for (i = ibox[0][0]; i < ibox[0][1]; i++)
          {
            index = ((k) * block_sz[1] + j) * block_sz[0] + i;

            block[index] = factor[2] + factor[1] + factor[0];
          }
    }

    /*------------------------------------------------------
     * Join block array in x
     *------------------------------------------------------*/

    for (k = 0; k < block_sz[2]; k++)
      for (j = 0; j < block_sz[1]; j++)
      {
        join = 0;
        for (i = 0; i < block_sz[0]; i++)
        {
          index = ((k) * block_sz[1] + j) * block_sz[0] + i;

          if ((join) && (block[index] == i_tmp1))
          {
            block[index] = 0;
            block[i_tmp0] += factor[0];
          }
          else
          {
            if (block[index])
            {
              i_tmp0 = index;
              i_tmp1 = block[index];
              join = 1;
            }
            else
              join = 0;
          }
        }
      }

    /*------------------------------------------------------
     * Join block array in y
     *------------------------------------------------------*/

    for (k = 0; k < block_sz[2]; k++)
      for (i = 0; i < block_sz[0]; i++)
      {
        join = 0;
        for (j = 0; j < block_sz[1]; j++)
        {
          index = ((k) * block_sz[1] + j) * block_sz[0] + i;

          if ((join) && (block[index] == i_tmp1))
          {
            block[index] = 0;
            block[i_tmp0] += factor[1];
          }
          else
          {
            if (block[index])
            {
              i_tmp0 = index;
              i_tmp1 = block[index];
              join = 1;
            }
            else
              join = 0;
          }
        }
      }

    /*------------------------------------------------------
     * Join block array in z
     *------------------------------------------------------*/

    for (i = 0; i < block_sz[0]; i++)
      for (j = 0; j < block_sz[1]; j++)
      {
        join = 0;
        for (k = 0; k < block_sz[2]; k++)
        {
          index = ((k) * block_sz[1] + j) * block_sz[0] + i;

          if ((join) && (block[index] == i_tmp1))
          {
            block[index] = 0;
            block[i_tmp0] += factor[2];
          }
          else
          {
            if (block[index])
            {
              i_tmp0 = index;
              i_tmp1 = block[index];
              join = 1;
            }
            else
              join = 0;
          }
        }
      }

    /*------------------------------------------------------
     * Set up new_sa representing the union
     *------------------------------------------------------*/

    FreeSubgridArray(new_sa);
    new_sa = NewSubgridArray();

    index = 0;
    for (k = 0; k < block_sz[2]; k++)
      for (j = 0; j < block_sz[1]; j++)
        for (i = 0; i < block_sz[0]; i++)
        {
          if (block[index])
          {
            i_tmp0 = block[index];

            ibox[0][0] = block_index[0][i];
            ibox[1][0] = block_index[1][j];
            ibox[2][0] = block_index[2][k];
            ibox[0][1] =
              block_index[0][i + ((i_tmp0 % factor[1]))];
            ibox[1][1] =
              block_index[1][j + ((i_tmp0 % factor[2]) / factor[1])];
            ibox[2][1] =
              block_index[2][k + ((i_tmp0) / factor[2])];

            subgrid =
              NewSubgrid(ibox[0][0],
                         ibox[1][0],
                         ibox[2][0],
                         ibox[0][1] - ibox[0][0],
                         ibox[1][1] - ibox[1][0],
                         ibox[2][1] - ibox[2][0],
                         rx, ry, rz, 0);

            AppendSubgrid(subgrid, new_sa);
          }

          index++;
        }

    ForSubgridI(si, old_sa)
    AppendSubgrid(SubgridArraySubgrid(old_sa, si), new_sa);

    /*------------------------------------------------------
     * Free up old_sa, block_index and block
     *------------------------------------------------------*/

    SubgridArraySize(old_sa) = 0;
    FreeSubgridArray(old_sa);

    for (d = 0; d < 3; d++)
      tfree(block_index[d]);

    tfree(block);
  }

  /*---------------------------------------------------------
   * Free up level_array and return new_sa
   *---------------------------------------------------------*/

  tfree(level_array);

  return new_sa;
}

