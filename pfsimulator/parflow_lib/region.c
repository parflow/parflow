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
* Member functions for Region class.
*
*****************************************************************************/
#include "parflow.h"
#include "grid.h"

#include <math.h>
#include <string.h>

/*--------------------------------------------------------------------------
 * NewSubregion
 *--------------------------------------------------------------------------*/

Subregion  *NewSubregion(
                         int ix,
                         int iy,
                         int iz,
                         int nx,
                         int ny,
                         int nz,
                         int sx,
                         int sy,
                         int sz,
                         int rx,
                         int ry, int
                         rz,
                         int process)
{
  Subregion *new_subregion;

  new_subregion = talloc(Subregion, 1);

  (new_subregion->ix) = ix;
  (new_subregion->iy) = iy;
  (new_subregion->iz) = iz;

  (new_subregion->nx) = nx;
  (new_subregion->ny) = ny;
  (new_subregion->nz) = nz;

  (new_subregion->sx) = sx;
  (new_subregion->sy) = sy;
  (new_subregion->sz) = sz;

  (new_subregion->rx) = rx;
  (new_subregion->ry) = ry;
  (new_subregion->rz) = rz;
  (new_subregion->level) = rx + ry + rz;

  (new_subregion->process) = process;

  return new_subregion;
}


/*--------------------------------------------------------------------------
 * NewSubregionArray
 *--------------------------------------------------------------------------*/

SubregionArray  *NewSubregionArray()
{
  SubregionArray *new_subregion_array;


  new_subregion_array = talloc(SubregionArray, 1);

  (new_subregion_array->subregions) = NULL;
  (new_subregion_array->size) = 0;

  return new_subregion_array;
}

/*--------------------------------------------------------------------------
 * NewRegion
 *--------------------------------------------------------------------------*/

Region  *NewRegion(
                   int size)
{
  Region  *new_region;

  int i;


  new_region = talloc(Region, 1);
  memset(new_region, 0, sizeof(Region));

  (new_region->subregion_arrays) = talloc(SubregionArray *, size);
  memset(new_region->subregion_arrays, 0, size * sizeof(SubregionArray *));

  for (i = 0; i < size; i++)
    RegionSubregionArray(new_region, i) = NewSubregionArray();
  (new_region->size) = size;

  return new_region;
}


/*--------------------------------------------------------------------------
 * FreeSubregion
 *--------------------------------------------------------------------------*/

void        FreeSubregion(Subregion *subregion)
{
  tfree(subregion);
}


/*--------------------------------------------------------------------------
 * FreeSubregionArray
 *--------------------------------------------------------------------------*/

void             FreeSubregionArray(
                                    SubregionArray *subregion_array)
{
  int i;


  ForSubregionI(i, subregion_array)
  FreeSubregion(SubregionArraySubregion(subregion_array, i));

  tfree(subregion_array->subregions);

  tfree(subregion_array);
}


/*--------------------------------------------------------------------------
 * FreeRegion
 *--------------------------------------------------------------------------*/

void     FreeRegion(
                    Region *region)
{
  int i;


  ForSubregionArrayI(i, region)
  FreeSubregionArray(RegionSubregionArray(region, i));

  tfree(region->subregion_arrays);

  tfree(region);
}


/*--------------------------------------------------------------------------
 * DuplicateSubregion:
 *   Return a duplicate subregion.
 *--------------------------------------------------------------------------*/

Subregion  *DuplicateSubregion(
                               Subregion *subregion)
{
  Subregion *new_subregion;


  new_subregion = NewSubregion(SubregionIX(subregion),
                               SubregionIY(subregion),
                               SubregionIZ(subregion),
                               SubregionNX(subregion),
                               SubregionNY(subregion),
                               SubregionNZ(subregion),
                               SubregionSX(subregion),
                               SubregionSY(subregion),
                               SubregionSZ(subregion),
                               SubregionRX(subregion),
                               SubregionRY(subregion),
                               SubregionRZ(subregion),
                               SubregionProcess(subregion));

  return new_subregion;
}


/*--------------------------------------------------------------------------
 * DuplicateSubregionArray:
 *   Return a duplicate subregion_array.
 *--------------------------------------------------------------------------*/

SubregionArray  *DuplicateSubregionArray(
                                         SubregionArray *subregion_array)
{
  SubregionArray *new_subregion_array;
  Subregion     **new_s;
  int new_sz;

  Subregion     **old_s;
  int i, data_sz;


  new_subregion_array = NewSubregionArray();
  new_s = NULL;
  new_sz = SubregionArraySize(subregion_array);

  if (new_sz)
  {
    data_sz = ((((new_sz - 1) / SubregionArrayBlocksize) + 1) *
               SubregionArrayBlocksize);
    new_s = talloc(Subregion *, data_sz);
    memset(new_s, 0, data_sz * sizeof(Subregion *));

    old_s = SubregionArraySubregions(subregion_array);

    for (i = 0; i < new_sz; i++)
      new_s[i] = DuplicateSubregion(old_s[i]);
  }

  SubregionArraySubregions(new_subregion_array) = new_s;
  SubregionArraySize(new_subregion_array) = new_sz;

  return new_subregion_array;
}


/*--------------------------------------------------------------------------
 * DuplicateRegion:
 *   Return a duplicate region.
 *--------------------------------------------------------------------------*/

Region  *DuplicateRegion(
                         Region *region)
{
  Region          *new_region;
  SubregionArray **new_sr_arrays;
  int new_sz;

  SubregionArray **old_sr_arrays;
  int i;


  new_sz = RegionSize(region);
  new_region = NewRegion(new_sz);

  if (new_sz)
  {
    new_sr_arrays = RegionSubregionArrays(new_region);
    old_sr_arrays = RegionSubregionArrays(region);

    for (i = 0; i < new_sz; i++)
    {
      FreeSubregionArray(new_sr_arrays[i]);
      new_sr_arrays[i] = DuplicateSubregionArray(old_sr_arrays[i]);
    }
  }

  return new_region;
}


/*--------------------------------------------------------------------------
 * AppendSubregion:
 *   Append subregion to the end of sr_array.
 *   The sr_array may be empty.
 *--------------------------------------------------------------------------*/

void             AppendSubregion(
                                 Subregion *     subregion,
                                 SubregionArray *sr_array)
{
  int sr_array_sz = SubregionArraySize(sr_array);

  Subregion  **old_s, **new_s;

  int i;


  if (!(sr_array_sz % SubregionArrayBlocksize))
  {
    new_s = talloc(Subregion *, sr_array_sz + SubregionArrayBlocksize);
    memset(new_s, 0, (sr_array_sz + SubregionArrayBlocksize) * sizeof(Subregion *));
    old_s = (sr_array->subregions);

    for (i = 0; i < sr_array_sz; i++)
    {
      new_s[i] = old_s[i];
    }

    (sr_array->subregions) = new_s;

    tfree(old_s);
  }

  SubregionArraySubregion(sr_array, sr_array_sz) = subregion;
  SubregionArraySize(sr_array)++;
  /*tfree(subregion);*/
}


/*--------------------------------------------------------------------------
 * DeleteSubregion:
 *   Delete subregion from sr_array.
 *--------------------------------------------------------------------------*/

void             DeleteSubregion(
                                 SubregionArray *sr_array,
                                 int             index)
{
  Subregion  **subregions;

  int i;


  subregions = SubregionArraySubregions(sr_array);

  FreeSubregion(subregions[index]);
  for (i = index; i < SubregionArraySize(sr_array) - 1; i++)
    subregions[i] = subregions[i + 1];

  SubregionArraySize(sr_array)--;
}


/*--------------------------------------------------------------------------
 * AppendSubregionArray:
 *   Append sr_array_0 to the end of sr_array_1.
 *   The sr_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void             AppendSubregionArray(
                                      SubregionArray *sr_array_0,
                                      SubregionArray *sr_array_1)
{
  int i;


  ForSubregionI(i, sr_array_0)
  AppendSubregion(SubregionArraySubregion(sr_array_0, i), sr_array_1);
}


/*--------------------------------------------------------------------------
 * IntersectSubregions: RDF todo
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * SubtractSubregions: RDF todo
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * UnionSubregionArray: RDF todo
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * CommRegFromStencil: RDF todo
 *--------------------------------------------------------------------------*/
