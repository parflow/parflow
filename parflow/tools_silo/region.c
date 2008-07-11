/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <math.h>
#include "pfload_file.h"
#include "region.h"


/*--------------------------------------------------------------------------
 * NewSubregion
 *--------------------------------------------------------------------------*/

Subregion  *NewSubregion(ix, iy, iz,
			 nx, ny, nz,
			 sx, sy, sz,
			 rx, ry, rz,
			 process)
int       ix, iy, iz;
int       nx, ny, nz;
int       sx, sy, sz;
int       rx, ry, rz;
int       process;
{
   Subregion *new;


   new = talloc(Subregion, 1);

   (new -> ix)       = ix;
   (new -> iy)       = iy;
   (new -> iz)       = iz;

   (new -> nx)       = nx;
   (new -> ny)       = ny;
   (new -> nz)       = nz;

   (new -> sx)       = sx;
   (new -> sy)       = sy;
   (new -> sz)       = sz;

   (new -> rx)       = rx;
   (new -> ry)       = ry;
   (new -> rz)       = rz;
   (new -> level)    = rx + ry + rz;

   (new -> process)  = process;

   return new;
}


/*--------------------------------------------------------------------------
 * NewSubregionArray
 *--------------------------------------------------------------------------*/

SubregionArray  *NewSubregionArray()
{
   SubregionArray *new;


   new = talloc(SubregionArray, 1);

   (new -> subregions) = NULL;
   (new -> size)     = 0;

   return new;
}


/*--------------------------------------------------------------------------
 * NewRegion
 *--------------------------------------------------------------------------*/

SGSRegion  *NewRegion(size)
int  size;
{
   SGSRegion  *new;

   int      i;


   new = ctalloc(SGSRegion, 1);

   if (size)
      (new -> subregion_arrays) = ctalloc(SubregionArray *, size);

   for (i = 0; i < size; i++)
      RegionSubregionArray(new, i) = NewSubregionArray();
   (new -> size)           = size;

   return new;
}


/*--------------------------------------------------------------------------
 * FreeSubregion
 *--------------------------------------------------------------------------*/

void  FreeSubregion(subregion)
Subregion  *subregion;
{
   free(subregion);
}


/*--------------------------------------------------------------------------
 * FreeSubregionArray
 *--------------------------------------------------------------------------*/

void  FreeSubregionArray(subregion_array)
SubregionArray  *subregion_array;
{
   int  i;


   ForSubregionI(i, subregion_array)
      FreeSubregion(SubregionArraySubregion(subregion_array, i));

   free(subregion_array -> subregions);
   free(subregion_array);
}


/*--------------------------------------------------------------------------
 * FreeRegion
 *--------------------------------------------------------------------------*/

void  FreeRegion(region)
SGSRegion  *region;
{
   int  i;


   ForSubregionArrayI(i, region)
      FreeSubregionArray(RegionSubregionArray(region, i));

   free(region -> subregion_arrays);
   free(region);
}


/*--------------------------------------------------------------------------
 * AppendSubregion:
 *   Append subregion to the end of subregion_array.
 *   The subregion_array may be empty.
 *--------------------------------------------------------------------------*/

void  AppendSubregion(subregion, subregion_array)
Subregion       *subregion;
SubregionArray **subregion_array;
{
   SubregionArray  *s_array    = *subregion_array;
   int              s_array_sz = SubregionArraySize(s_array);

   Subregion      **old_s, **new_s;

   int              i;


   if (!(s_array_sz % SubregionArrayBlocksize))
   {
      new_s = ctalloc(Subregion *, s_array_sz + SubregionArrayBlocksize);
      old_s = (s_array -> subregions);

      for (i = 0; i < s_array_sz; i++)
	 new_s[i] = old_s[i];

      (s_array -> subregions) = new_s;

      if (s_array_sz)
	 free(old_s);
   }

   SubregionArraySubregion(s_array, s_array_sz) = subregion;
   SubregionArraySize(s_array) ++;
}


/*--------------------------------------------------------------------------
 * AppendSubregionArray:
 *   Append subregion_array_0 to the end of subregion_array_1.
 *   The subregion_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void  AppendSubregionArray(subregion_array_0, subregion_array_1)
SubregionArray  *subregion_array_0;
SubregionArray **subregion_array_1;
{
   int  i;


   ForSubregionI(i, subregion_array_0)
      AppendSubregion(SubregionArraySubregion(subregion_array_0, i),
		      subregion_array_1);
}

