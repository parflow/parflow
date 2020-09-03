/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
* Routines for manipulating vector structures.
*
*
*****************************************************************************/

#include "parflow.h"
#include "vector.h"


/*--------------------------------------------------------------------------
 * NewVectorCommPkg:
 *--------------------------------------------------------------------------*/

CommPkg  *NewVectorCommPkg(vector, compute_pkg)
Vector * vector;
ComputePkg  *compute_pkg;
{
  CommPkg     *new;


  new = NewCommPkg(ComputePkgSendRegion(compute_pkg),
                   ComputePkgRecvRegion(compute_pkg),
                   VectorDataSpace(vector), 1, VectorData(vector));

  return new;
}

/*--------------------------------------------------------------------------
 * InitVectorUpdate
 *--------------------------------------------------------------------------*/

CommHandle  *InitVectorUpdate(vector, update_mode)
Vector * vector;
int update_mode;
{
#ifdef SHMEM_OBJECTS
  return (void*)1;
#else
#ifdef NO_VECTOR_UPDATE
  return (void*)1;
#else
  return InitCommunication(VectorCommPkg(vector, update_mode));
#endif
#endif
}


/*--------------------------------------------------------------------------
 * FinalizeVectorUpdate
 *--------------------------------------------------------------------------*/

void         FinalizeVectorUpdate(handle)
CommHandle * handle;
{
#ifdef SHMEM_OBJECTS
  amps_Sync(amps_CommWorld);
#else
#ifdef NO_VECTOR_UPDATE
#else
  FinalizeCommunication(handle);
#endif
#endif
}


/*--------------------------------------------------------------------------
 * NewTempVector
 *--------------------------------------------------------------------------*/

Vector  *NewTempVector(grid, nc, num_ghost)
Grid * grid;
int nc;
int num_ghost;
{
  Vector    *new;
  Subvector *new_sub;

  Subgrid   *subgrid;

  int data_size;
  int i, n;


  new = ctalloc(Vector, 1);

  (new->subvectors) = ctalloc(Subvector *, GridNumSubgrids(grid));

  data_size = 0;

  VectorDataSpace(new) = NewSubgridArray();
  ForSubgridI(i, GridSubgrids(grid))
  {
    new_sub = ctalloc(Subvector, 1);

    subgrid = GridSubgrid(grid, i);

    SubvectorDataSpace(new_sub) =
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
    AppendSubgrid(SubvectorDataSpace(new_sub),
                  VectorDataSpace(new));

    n = SubvectorNX(new_sub) * SubvectorNY(new_sub) * SubvectorNZ(new_sub);

    data_size = n;

    VectorSubvector(new, i) = new_sub;
  }

  (new->data_size) = data_size;

  VectorGrid(new) = grid;

  VectorSize(new) = GridSize(grid);

  return new;
}


/*--------------------------------------------------------------------------
 * SetTempVectorData
 *--------------------------------------------------------------------------*/

void     SetTempVectorData(vector, data)
Vector * vector;
double  *data;
{
  Grid       *grid = VectorGrid(vector);

  int i;

#ifdef SHMEM_OBJECTS
  amps_Invoice invoice;

  /* Warning Passing pointer as a double !!!! */
  invoice = amps_NewInvoice("%i", &data);
#endif


  /* if necessary, free old CommPkg's */
  if (VectorData(vector))
    for (i = 0; i < NumUpdateModes; i++)
      FreeCommPkg(VectorCommPkg(vector, i));

#ifdef SHMEM_OBJECTS
  amps_BCast(amps_CommWorld, 0, invoice);

  if (amps_Rank(amps_CommWorld))
  {
    data += vector->shmem_offset;
  }
#endif

  VectorData(vector) = data;

  ForSubgridI(i, GridSubgrids(grid))
  SubvectorData(VectorSubvector(vector, i)) = VectorData(vector);

  for (i = 0; i < NumUpdateModes; i++)
  {
    VectorCommPkg(vector, i) =
      NewVectorCommPkg(vector, GridComputePkg(grid, i));
  }
}


/*--------------------------------------------------------------------------
 * NewVector
 *--------------------------------------------------------------------------*/

Vector  *NewVector(grid, nc, num_ghost)
Grid * grid;
int nc;
int num_ghost;
{
  Vector  *new;
  double  *data;

  new = NewTempVector(grid, nc, num_ghost);

#ifdef SHMEM_OBJECTS
  /* Node 0 allocates */
  if (!amps_Rank(amps_CommWorld))
    data = ctalloc_amps(double, SizeOfVector(new));
#else
  data = ctalloc_amps(double, SizeOfVector(new));
#endif

  SetTempVectorData(new, data);

  return new;
}


/*--------------------------------------------------------------------------
 * FreeTempVector
 *--------------------------------------------------------------------------*/

void FreeTempVector(vector)
Vector * vector;
{
  int i;


  for (i = 0; i < NumUpdateModes; i++)
    FreeCommPkg(VectorCommPkg(vector, i));

  ForSubgridI(i, GridSubgrids(VectorGrid(vector)))
  {
    tfree(VectorSubvector(vector, i));
  }

  FreeSubgridArray(VectorDataSpace(vector));

  tfree(vector->subvectors);
  tfree(vector);
}


/*--------------------------------------------------------------------------
 * FreeVector
 *--------------------------------------------------------------------------*/

void     FreeVector(vector)
Vector * vector;
{
#ifndef SHMEM_OBJECTS
  tfree_amps(VectorData(vector));
#endif

  FreeTempVector(vector);
}


/*--------------------------------------------------------------------------
 * InitVector
 *--------------------------------------------------------------------------*/

void    InitVector(v, value)
Vector * v;
double value;
{
  Grid       *grid = VectorGrid(v);

  Subvector  *v_sub;
  double     *vp;

  Subgrid    *subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    vp = SubvectorElt(v_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vp[iv] = value;
    });
  }
}

/*--------------------------------------------------------------------------
 * InitVectorAll
 *--------------------------------------------------------------------------*/

void    InitVectorAll(v, value)
Vector * v;
double value;
{
  Grid       *grid = VectorGrid(v);

  Subvector  *v_sub;
  double     *vp;

  Subgrid    *subgrid;

  int ix_v, iy_v, iz_v;
  int nx_v, ny_v, nz_v;

  int i_s;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(v, i_s);

    ix_v = SubvectorIX(v_sub);
    iy_v = SubvectorIY(v_sub);
    iz_v = SubvectorIZ(v_sub);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    vp = SubvectorData(v_sub);

    iv = 0;
    BoxLoopI1(i, j, k, ix_v, iy_v, iz_v, nx_v, ny_v, nz_v,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vp[iv] = value;
    });
  }

#ifdef SHMEM_OBJECTS
  amps_Sync(amps_CommWorld);
#endif
}


/*--------------------------------------------------------------------------
 * InitVectorInc
 *--------------------------------------------------------------------------*/


void    InitVectorInc(v, value, inc)
Vector * v;
double value;
double inc;
{
  Grid       *grid = VectorGrid(v);

  Subvector  *v_sub;
  double     *vp;

  Subgrid    *subgrid;


  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    vp = SubvectorElt(v_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vp[iv] = value + (i + j + k) * inc;
    });
  }
}


/*--------------------------------------------------------------------------
 * InitVectorRandom
 *--------------------------------------------------------------------------*/

void    InitVectorRandom(v, seed)
Vector * v;
long seed;
{
  Grid       *grid = VectorGrid(v);

  Subvector  *v_sub;
  double     *vp;

  Subgrid    *subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s;
  int i, j, k, iv;

  void   srand48();
  double drand48();


  srand48(seed);

  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    vp = SubvectorElt(v_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vp[iv] = drand48();
    });
  }
}

