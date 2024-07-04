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
* Routines for manipulating charvector structures.
*
*
*****************************************************************************/

#include "parflow.h"
#include "char_vector.h"

/*--------------------------------------------------------------------------
 * NewCharVectorUpdatePkg:
 *   `send_region' and `recv_region' are "regions" of `grid'.
 *
 * ===============  Don't use this yet; NewCommPkg routine that
 * ===============  accepts char input is needed first.
 *
 *--------------------------------------------------------------------------*/

CommPkg  *NewCharVectorUpdatePkg(
                                 CharVector *charvector,
                                 int         update_mode)
{
  CommPkg     *new_comm_pkg;
  ComputePkg  *compute_pkg;


  compute_pkg = GridComputePkg(CharVectorGrid(charvector), update_mode);

  new_comm_pkg = NewCommPkg(ComputePkgSendRegion(compute_pkg),
                            ComputePkgRecvRegion(compute_pkg),
                            CharVectorDataSpace(charvector),
                            CharVectorNC(charvector),
                            (double*)CharVectorData(charvector));

  return new_comm_pkg;
}

/*--------------------------------------------------------------------------
 * InitCharVectorUpdate
 *--------------------------------------------------------------------------*/

CommHandle  *InitCharVectorUpdate(
                                  CharVector *charvector,
                                  int         update_mode)
{
  return InitCommunication(CharVectorCommPkg(charvector, update_mode));
}


/*--------------------------------------------------------------------------
 * FinalizeCharVectorUpdate
 *--------------------------------------------------------------------------*/

void         FinalizeCharVectorUpdate(CommHandle *handle)
{
  FinalizeCommunication(handle);
}


/*--------------------------------------------------------------------------
 * NewTempCharVector
 *--------------------------------------------------------------------------*/

CharVector  *NewTempCharVector(
                               Grid *grid,
                               int   nc,
                               int   num_ghost)
{
  CharVector    *new_char_vector;
  Subcharvector *new_sub;

  Subgrid   *subgrid;

  int data_size;
  int i, j, n;


  new_char_vector = ctalloc(CharVector, 1);

  (new_char_vector->subcharvectors) = ctalloc(Subcharvector *, GridNumSubgrids(grid));

  data_size = 0;
  CharVectorDataSpace(new_char_vector) = NewSubgridArray();
  ForSubgridI(i, GridSubgrids(grid))
  {
    new_sub = ctalloc(Subcharvector, 1);

    subgrid = GridSubgrid(grid, i);

    (new_sub->data_index) = talloc(int, nc);

    SubcharvectorDataSpace(new_sub) =
      NewSubgrid(SubgridIX(subgrid) - num_ghost,
                 SubgridIY(subgrid) - num_ghost,
                 SubgridIZ(subgrid) - num_ghost,
                 SubgridNX(subgrid) + 2 * num_ghost,
                 SubgridNY(subgrid) + 2 * num_ghost,
                 SubgridNZ(subgrid) + 2 * num_ghost,
                 SubgridRX(subgrid),
                 SubgridRY(subgrid),
                 SubgridRZ(subgrid),
                 SubgridProcess(subgrid));
    AppendSubgrid(SubcharvectorDataSpace(new_sub),
                  CharVectorDataSpace(new_char_vector));

    SubcharvectorNC(new_sub) = nc;

    n = SubcharvectorNX(new_sub) * SubcharvectorNY(new_sub) * SubcharvectorNZ(new_sub);
    for (j = 0; j < nc; j++)
    {
      (new_sub->data_index[j]) = data_size;

      data_size += n;
    }

    CharVectorSubcharvector(new_char_vector, i) = new_sub;
  }

  (new_char_vector->data_size) = data_size;

  CharVectorGrid(new_char_vector) = grid;

  CharVectorNC(new_char_vector) = nc;

  CharVectorSize(new_char_vector) = GridSize(grid) * nc;

  return new_char_vector;
}


/*--------------------------------------------------------------------------
 * SetTempCharVectorData
 *--------------------------------------------------------------------------*/

void     SetTempCharVectorData(
                               CharVector *charvector,
                               char *      data)
{
  Grid       *grid = CharVectorGrid(charvector);

  int i;


  /* if necessary, free old CommPkg's */
  if (CharVectorData(charvector))
    for (i = 0; i < NumUpdateModes; i++)
      FreeCommPkg(CharVectorCommPkg(charvector, i));

  CharVectorData(charvector) = data;

  ForSubgridI(i, GridSubgrids(grid))
  SubcharvectorData(CharVectorSubcharvector(charvector, i)) = CharVectorData(charvector);

  for (i = 0; i < NumUpdateModes; i++)
    CharVectorCommPkg(charvector, i) = NewCharVectorUpdatePkg(charvector, i);
}


/*--------------------------------------------------------------------------
 * NewCharVector
 *--------------------------------------------------------------------------*/

CharVector  *NewCharVector(
                           Grid *grid,
                           int   nc,
                           int   num_ghost)
{
  CharVector  *new_char_vector;
  char  *data;


  new_char_vector = NewTempCharVector(grid, nc, num_ghost);
  data = ctalloc_amps(char, SizeOfCharVector(new_char_vector));
  SetTempCharVectorData(new_char_vector, data);

  return new_char_vector;
}


/*--------------------------------------------------------------------------
 * FreeTempCharVector
 *--------------------------------------------------------------------------*/

void FreeTempCharVector(
                        CharVector *charvector)
{
  int i;


  for (i = 0; i < NumUpdateModes; i++)
    FreeCommPkg(CharVectorCommPkg(charvector, i));

  ForSubgridI(i, GridSubgrids(CharVectorGrid(charvector)))
  {
    tfree(CharVectorSubcharvector(charvector, i)->data_index);
    tfree(CharVectorSubcharvector(charvector, i));
  }

  FreeSubgridArray(CharVectorDataSpace(charvector));

  tfree(charvector->subcharvectors);
  tfree(charvector);
}


/*--------------------------------------------------------------------------
 * FreeCharVector
 *--------------------------------------------------------------------------*/

void     FreeCharVector(
                        CharVector *charvector)
{
  tfree_amps(CharVectorData(charvector));
  FreeTempCharVector(charvector);
}


/*--------------------------------------------------------------------------
 * InitCharVector
 *--------------------------------------------------------------------------*/

void InitCharVector(CharVector *v, char value)
{
  Grid       *grid = CharVectorGrid(v);

  Subcharvector  *v_sub;
  char     *vp;

  Subgrid    *subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s, a;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = CharVectorSubcharvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubcharvectorNX(v_sub);
    ny_v = SubcharvectorNY(v_sub);
    nz_v = SubcharvectorNZ(v_sub);

    for (a = 0; a < SubcharvectorNC(v_sub); a++)
    {
      vp = SubcharvectorElt(v_sub, a, ix, iy, iz);

      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        vp[iv] = value;
      });
    }
  }
}

/*--------------------------------------------------------------------------
 * InitCharVectorAll
 *--------------------------------------------------------------------------*/

void InitCharVectorAll(CharVector *v, char value)
{
  Grid       *grid = CharVectorGrid(v);

  Subcharvector  *v_sub;
  char     *vp;

  int ix_v, iy_v, iz_v;
  int nx_v, ny_v, nz_v;

  int i_s, a;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    v_sub = CharVectorSubcharvector(v, i_s);

    ix_v = SubcharvectorIX(v_sub);
    iy_v = SubcharvectorIY(v_sub);
    iz_v = SubcharvectorIZ(v_sub);

    nx_v = SubcharvectorNX(v_sub);
    ny_v = SubcharvectorNY(v_sub);
    nz_v = SubcharvectorNZ(v_sub);

    for (a = 0; a < SubcharvectorNC(v_sub); a++)
    {
      vp = SubcharvectorData(v_sub);

      iv = 0;
      BoxLoopI1(i, j, k, ix_v, iy_v, iz_v, nx_v, ny_v, nz_v,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        vp[iv] = value;
      });
    }
  }
}


/*--------------------------------------------------------------------------
 * InitCharVectorInc
 *--------------------------------------------------------------------------*/


void InitCharVectorInc(CharVector *v, char value, int inc)
{
  Grid       *grid = CharVectorGrid(v);

  Subcharvector  *v_sub;
  char     *vp;

  Subgrid    *subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s, a;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = CharVectorSubcharvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubcharvectorNX(v_sub);
    ny_v = SubcharvectorNY(v_sub);
    nz_v = SubcharvectorNZ(v_sub);

    for (a = 0; a < SubcharvectorNC(v_sub); a++)
    {
      vp = SubcharvectorElt(v_sub, a, ix, iy, iz);

      iv = 0;
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        vp[iv] = (char)(value + (i + j + k) * inc);
      });
    }
  }
}




