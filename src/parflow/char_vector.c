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
 * Routines for manipulating charvector structures.
 *
 *
 *****************************************************************************/

#include "parflow.h"
#include "char_vector.h"

/*--------------------------------------------------------------------------
 * NewCharVectorUpdatePkg:
 *   `send_region' and `recv_region' are "regions" of `grid'.

===============  Don't use this yet; NewCommPkg routine that
===============  accepts char input is needed first.

 *--------------------------------------------------------------------------*/

CommPkg  *NewCharVectorUpdatePkg(charvector, update_mode)
CharVector   *charvector;
int       update_mode;
{
   CommPkg     *new;
   ComputePkg  *compute_pkg;


   compute_pkg = GridComputePkg(CharVectorGrid(charvector), update_mode);

   new = NewCommPkg(ComputePkgSendRegion(compute_pkg),
		    ComputePkgRecvRegion(compute_pkg),
		    CharVectorDataSpace(charvector),
		    CharVectorNC(charvector),
		    (double*) CharVectorData(charvector)); 

   return new;
}

/*--------------------------------------------------------------------------
 * InitCharVectorUpdate
 *--------------------------------------------------------------------------*/

CommHandle  *InitCharVectorUpdate(charvector, update_mode)
CharVector      *charvector;
int          update_mode;
{
   return  InitCommunication(CharVectorCommPkg(charvector, update_mode));
}


/*--------------------------------------------------------------------------
 * FinalizeCharVectorUpdate
 *--------------------------------------------------------------------------*/

void         FinalizeCharVectorUpdate(handle)
CommHandle  *handle;
{
   FinalizeCommunication(handle);
}


/*--------------------------------------------------------------------------
 * NewTempCharVector
 *--------------------------------------------------------------------------*/

CharVector  *NewTempCharVector(grid, nc, num_ghost)
Grid    *grid;
int      nc;
int      num_ghost;
{
   CharVector    *new;
   Subcharvector *new_sub;

   Subgrid   *subgrid;

   int        data_size;
   int        i, j, n;


   new = ctalloc(CharVector, 1);

   (new -> subcharvectors) = ctalloc(Subcharvector *, GridNumSubgrids(grid));

   data_size = 0;
   CharVectorDataSpace(new) = NewSubgridArray();
   ForSubgridI(i, GridSubgrids(grid))
   {
      new_sub = ctalloc(Subcharvector, 1);

      subgrid = GridSubgrid(grid ,i);

      (new_sub -> data_index) = talloc(int, nc);

      SubcharvectorDataSpace(new_sub) = 
	 NewSubgrid(SubgridIX(subgrid) - num_ghost,
		    SubgridIY(subgrid) - num_ghost,
		    SubgridIZ(subgrid) - num_ghost,
		    SubgridNX(subgrid) + 2*num_ghost,
		    SubgridNY(subgrid) + 2*num_ghost,
		    SubgridNZ(subgrid) + 2*num_ghost,
		    SubgridRX(subgrid),
		    SubgridRY(subgrid),
		    SubgridRZ(subgrid),
		    SubgridProcess(subgrid));
      AppendSubgrid(SubcharvectorDataSpace(new_sub),
		    CharVectorDataSpace(new));

      SubcharvectorNC(new_sub) = nc;
       
      n = SubcharvectorNX(new_sub) * SubcharvectorNY(new_sub) * SubcharvectorNZ(new_sub);
      for (j = 0; j < nc; j++)
      {
	 (new_sub -> data_index[j]) = data_size;

	 data_size += n;
      }

      CharVectorSubcharvector(new, i) = new_sub;
   }

   (new -> data_size) = data_size;

   CharVectorGrid(new) = grid;

   CharVectorNC(new) = nc;

   CharVectorSize(new) = GridSize(grid) * nc;

   return new;
}


/*--------------------------------------------------------------------------
 * SetTempCharVectorData
 *--------------------------------------------------------------------------*/

void     SetTempCharVectorData(charvector, data)
CharVector  *charvector;
char  *data;
{
   Grid       *grid = CharVectorGrid(charvector);

   int         i;


   /* if necessary, free old CommPkg's */
   if (CharVectorData(charvector))
      for(i = 0; i < NumUpdateModes; i++)
	 FreeCommPkg(CharVectorCommPkg(charvector, i));

   CharVectorData(charvector) = data;

   ForSubgridI(i, GridSubgrids(grid))
      SubcharvectorData(CharVectorSubcharvector(charvector, i)) = CharVectorData(charvector);

   for(i = 0; i < NumUpdateModes; i++)
      CharVectorCommPkg(charvector, i) = NewCharVectorUpdatePkg(charvector, i);
}


/*--------------------------------------------------------------------------
 * NewCharVector
 *--------------------------------------------------------------------------*/

CharVector  *NewCharVector(grid, nc, num_ghost)
Grid    *grid;
int      nc;
int      num_ghost;
{
    CharVector  *new;
    char  *data;


    new = NewTempCharVector(grid, nc, num_ghost);
    data = amps_CTAlloc(char, SizeOfCharVector(new));
    SetTempCharVectorData(new, data);

    return new;
}


/*--------------------------------------------------------------------------
 * FreeTempCharVector
 *--------------------------------------------------------------------------*/

void FreeTempCharVector(charvector)
CharVector *charvector;
{
   int i;


   for(i = 0; i < NumUpdateModes; i++)
      FreeCommPkg(CharVectorCommPkg(charvector, i));

   ForSubgridI(i, GridSubgrids(CharVectorGrid(charvector)))
   {
      tfree(CharVectorSubcharvector(charvector, i) -> data_index);
      tfree(CharVectorSubcharvector(charvector, i));
   }

   FreeSubgridArray(CharVectorDataSpace(charvector));

   tfree(charvector -> subcharvectors);
   tfree(charvector);
}


/*--------------------------------------------------------------------------
 * FreeCharVector
 *--------------------------------------------------------------------------*/

void     FreeCharVector(charvector)
CharVector  *charvector;
{
   amps_TFree(CharVectorData(charvector));
   FreeTempCharVector(charvector);
}


/*--------------------------------------------------------------------------
 * InitCharVector
 *--------------------------------------------------------------------------*/

void    InitCharVector(v, value)
CharVector *v;
char  value;
{
   Grid       *grid = CharVectorGrid(v);

   Subcharvector  *v_sub;
   char     *vp;

   Subgrid    *subgrid;

   int         ix,   iy,   iz;
   int         nx,   ny,   nz;
   int         nx_v, ny_v, nz_v;

   int         i_s, a;
   int         i, j, k, iv;


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

void    InitCharVectorAll(v, value)
CharVector *v;
char  value;
{
   Grid       *grid = CharVectorGrid(v);

   Subcharvector  *v_sub;
   char     *vp;

   Subgrid    *subgrid;

   int         ix_v, iy_v, iz_v;
   int         nx_v, ny_v, nz_v;

   int         i_s, a;
   int         i, j, k, iv;


   ForSubgridI(i_s, GridSubgrids(grid))
   {
      subgrid = GridSubgrid(grid, i_s);

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


void    InitCharVectorInc(v, value, inc)
CharVector *v;
char  value;
char  inc;
{
   Grid       *grid = CharVectorGrid(v);

   Subcharvector  *v_sub;
   char     *vp;

   Subgrid    *subgrid;

   int         ix,   iy,   iz;
   int         nx,   ny,   nz;
   int         nx_v, ny_v, nz_v;

   int         i_s, a;
   int         i, j, k, iv;


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
	   vp[iv] = value + (i + j + k)*inc;
	});
      }
   }
}




