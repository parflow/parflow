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
#include "grid.h"


/*--------------------------------------------------------------------------
 * NewGrid
 *--------------------------------------------------------------------------*/

Grid  *NewGrid(subgrids, all_subgrids, neighbors)
SubgridArray  *subgrids;
SubgridArray  *all_subgrids;
SubgridArray  *neighbors;
{
   Grid    *new;

   Subgrid *s;

   int      i, size;


   new = talloc(Grid, 1);

   (new -> subgrids)      = subgrids;
   (new -> all_subgrids)  = all_subgrids;
   (new -> neighbors)     = neighbors;

   size = 0;
   for (i = 0; i < SubgridArraySize(all_subgrids); i++)
   {
      s = SubgridArraySubgrid(all_subgrids, i);
      size += (s -> nx)*(s -> ny)*(s -> nz);
   }
   (new -> size) = size;

   return new;
}


/*--------------------------------------------------------------------------
 * FreeGrid
 *--------------------------------------------------------------------------*/

void  FreeGrid(grid)
Grid  *grid;
{

   FreeSubgridArray(GridAllSubgrids(grid));

   /* these subgrid arrays point to subgrids in all_subgrids */
   SubgridArraySize(GridSubgrids(grid)) = 0;
   FreeSubgridArray(GridSubgrids(grid));
   SubgridArraySize(GridNeighbors(grid)) = 0;
   FreeSubgridArray(GridNeighbors(grid));

   free(grid);
}



