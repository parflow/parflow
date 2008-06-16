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
 * Member functions for Region class.
 *
 *****************************************************************************/

#include <math.h>
#include "parflow.h"
#include "grid.h"


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

Region  *NewRegion(size)
int      size;
{
   Region  *new;

   int      i;


   new = ctalloc(Region, 1);

   (new -> subregion_arrays) = ctalloc(SubregionArray *, size);

   for (i = 0; i < size; i++)
      RegionSubregionArray(new, i) = NewSubregionArray();
   (new -> size)           = size;

   return new;
}


/*--------------------------------------------------------------------------
 * FreeSubregion
 *--------------------------------------------------------------------------*/

void        FreeSubregion(subregion)
Subregion  *subregion;
{
   tfree(subregion);
}


/*--------------------------------------------------------------------------
 * FreeSubregionArray
 *--------------------------------------------------------------------------*/

void             FreeSubregionArray(subregion_array)
SubregionArray  *subregion_array;
{
   int  i;


   ForSubregionI(i, subregion_array)
      FreeSubregion(SubregionArraySubregion(subregion_array, i));

   tfree(subregion_array -> subregions);

   tfree(subregion_array);
}


/*--------------------------------------------------------------------------
 * FreeRegion
 *--------------------------------------------------------------------------*/

void     FreeRegion(region)
Region  *region;
{
   int  i;


   ForSubregionArrayI(i, region)
      FreeSubregionArray(RegionSubregionArray(region, i));

   tfree(region -> subregion_arrays);

   tfree(region);
}


/*--------------------------------------------------------------------------
 * DuplicateSubregion:
 *   Return a duplicate subregion.
 *--------------------------------------------------------------------------*/

Subregion  *DuplicateSubregion(subregion)
Subregion  *subregion;
{
   Subregion *new;


   new = NewSubregion(SubregionIX(subregion),
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

   return new;
}


/*--------------------------------------------------------------------------
 * DuplicateSubregionArray:
 *   Return a duplicate subregion_array.
 *--------------------------------------------------------------------------*/

SubregionArray  *DuplicateSubregionArray(subregion_array)
SubregionArray  *subregion_array;
{
   SubregionArray *new;
   Subregion     **new_s;
   int             new_sz;

   Subregion     **old_s;
   int             i, data_sz;


   new = NewSubregionArray();
   new_s = NULL;
   new_sz = SubregionArraySize(subregion_array);

   if (new_sz)
   {
      data_sz = ((((new_sz - 1) / SubregionArrayBlocksize) + 1) *
		 SubregionArrayBlocksize);
      new_s = ctalloc(Subregion *, data_sz);

      old_s = SubregionArraySubregions(subregion_array);

      for (i = 0; i < new_sz; i++)
	 new_s[i] = DuplicateSubregion(old_s[i]);
   }

   SubregionArraySubregions(new) = new_s;
   SubregionArraySize(new)       = new_sz;

   return new;
}


/*--------------------------------------------------------------------------
 * DuplicateRegion:
 *   Return a duplicate region.
 *--------------------------------------------------------------------------*/

Region  *DuplicateRegion(region)
Region  *region;
{
   Region          *new;
   SubregionArray **new_sr_arrays;
   int              new_sz;

   SubregionArray **old_sr_arrays;
   int             i;


   new_sz = RegionSize(region);
   new = NewRegion(new_sz);

   if (new_sz)
   {
      new_sr_arrays = RegionSubregionArrays(new);
      old_sr_arrays = RegionSubregionArrays(region);

      for (i = 0; i < new_sz; i++)
      {
	 FreeSubregionArray(new_sr_arrays[i]);
	 new_sr_arrays[i] = DuplicateSubregionArray(old_sr_arrays[i]);
      }
   }

   return new;
}


/*--------------------------------------------------------------------------
 * AppendSubregion:
 *   Append subregion to the end of sr_array.
 *   The sr_array may be empty.
 *--------------------------------------------------------------------------*/

void             AppendSubregion(subregion, sr_array)
Subregion       *subregion;
SubregionArray  *sr_array;
{
   int          sr_array_sz = SubregionArraySize(sr_array);

   Subregion  **old_s, **new_s;

   int          i;


   if (!(sr_array_sz % SubregionArrayBlocksize))
   {
      new_s = ctalloc(Subregion *, sr_array_sz + SubregionArrayBlocksize);
      old_s = (sr_array -> subregions);

      for (i = 0; i < sr_array_sz; i++) {
	 new_s[i] = old_s[i];
      }

      (sr_array -> subregions) = new_s;

      tfree(old_s);
   }

   SubregionArraySubregion(sr_array, sr_array_sz) = subregion;
   SubregionArraySize(sr_array) ++;
   /*tfree(subregion);*/
}


/*--------------------------------------------------------------------------
 * DeleteSubregion:
 *   Delete subregion from sr_array.
 *--------------------------------------------------------------------------*/

void             DeleteSubregion(sr_array, index)
SubregionArray  *sr_array;
int              index;
{
   Subregion  **subregions;

   int          i;


   subregions = SubregionArraySubregions(sr_array);

   FreeSubregion(subregions[index]);
   for (i = index; i < SubregionArraySize(sr_array) - 1; i++)
      subregions[i] = subregions[i+1];

   SubregionArraySize(sr_array) --;
}


/*--------------------------------------------------------------------------
 * AppendSubregionArray:
 *   Append sr_array_0 to the end of sr_array_1.
 *   The sr_array_1 may be empty.
 *--------------------------------------------------------------------------*/

void             AppendSubregionArray(sr_array_0, sr_array_1)
SubregionArray  *sr_array_0;
SubregionArray  *sr_array_1;
{
   int  i;


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

