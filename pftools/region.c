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

#include <math.h>
#include "pfload_file.h"
#include "region.h"

SubregionArray  *NewSubregionArray();


/*--------------------------------------------------------------------------
 * NewSubregion
 *--------------------------------------------------------------------------*/

Subregion  *NewSubregion(
                         int ix,
                         int iy, int iz,
                         int nx, int ny, int nz,
                         int sx, int sy, int sz,
                         int rx, int ry, int rz,
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

SGSRegion  *NewRegion(
                      int size)
{
  SGSRegion  *new_region;

  int i;


  new_region = ctalloc(SGSRegion, 1);

  if (size)
    (new_region->subregion_arrays) = ctalloc(SubregionArray *, size);

  for (i = 0; i < size; i++)
    RegionSubregionArray(new_region, i) = NewSubregionArray();
  (new_region->size) = size;

  return new_region;
}


/*--------------------------------------------------------------------------
 * FreeSubregion
 *--------------------------------------------------------------------------*/

void  FreeSubregion(
                    Subregion *subregion)
{
  free(subregion);
}


/*--------------------------------------------------------------------------
 * FreeSubregionArray
 *--------------------------------------------------------------------------*/

void  FreeSubregionArray(
                         SubregionArray *subregion_array)
{
  int i;


  ForSubregionI(i, subregion_array)
  FreeSubregion(SubregionArraySubregion(subregion_array, i));

  free(subregion_array->subregions);
  free(subregion_array);
}


/*--------------------------------------------------------------------------
 * FreeRegion
 *--------------------------------------------------------------------------*/

void  FreeRegion(
                 SGSRegion *region)
{
  int i;


  ForSubregionArrayI(i, region)
  FreeSubregionArray(RegionSubregionArray(region, i));

  free(region->subregion_arrays);
  free(region);
}


/*--------------------------------------------------------------------------
 * AppendSubregion:
 *   Append subregion to the end of subregion_array.
 *   The subregion_array may be empty.
 *--------------------------------------------------------------------------*/

void  AppendSubregion(
                      Subregion *      subregion,
                      SubregionArray **subregion_array)
{
  SubregionArray  *s_array = *subregion_array;
  int s_array_sz = SubregionArraySize(s_array);

  Subregion      **old_s, **new_s;

  int i;


  if (!(s_array_sz % SubregionArrayBlocksize))
  {
    new_s = ctalloc(Subregion *, s_array_sz + SubregionArrayBlocksize);
    old_s = (s_array->subregions);

    for (i = 0; i < s_array_sz; i++)
      new_s[i] = old_s[i];

    (s_array->subregions) = new_s;

    if (s_array_sz)
      free(old_s);
  }

  SubregionArraySubregion(s_array, s_array_sz) = subregion;
  SubregionArraySize(s_array)++;
}


/*--------------------------------------------------------------------------
 * AppendSubregionArray:
 *   Append subregion_array_0 to the end of subregion_array_1.
 *   The subregion_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void  AppendSubregionArray(
                           SubregionArray * subregion_array_0,
                           SubregionArray **subregion_array_1)
{
  int i;


  ForSubregionI(i, subregion_array_0)
  AppendSubregion(SubregionArraySubregion(subregion_array_0, i),
                  subregion_array_1);
}

