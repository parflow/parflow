/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * ReadUserSubgrid, ReadUserGrid,
 * FreeUserGrid
 *
 * Routines for reading user_grid input.
 * 
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * ReadUserSubgrid
 *--------------------------------------------------------------------------*/

Subgrid    *ReadUserSubgrid()
{
    Subgrid  *new;

    int       ix, iy, iz;
    int       nx, ny, nz;
    int       rx, ry, rz;

    ix = GetIntDefault("UserGrid.IX", 0);
    iy = GetIntDefault("UserGrid.IY", 0);
    iz = GetIntDefault("UserGrid.IZ", 0);

    rx = GetIntDefault("UserGrid.RX", 0);
    ry = GetIntDefault("UserGrid.RY", 0);
    rz = GetIntDefault("UserGrid.RZ", 0);

    nx = GetInt("ComputationalGrid.NX");
    ny = GetInt("ComputationalGrid.NY");
    nz = GetInt("ComputationalGrid.NZ");

    new = NewSubgrid(ix, iy, iz, nx, ny, nz, rx, ry, rz, -1);

    return new;
}

/*--------------------------------------------------------------------------
 * ReadUserGrid
 *--------------------------------------------------------------------------*/

Grid      *ReadUserGrid()
{
   Grid          *user_grid;

   SubgridArray  *user_all_subgrids;
   SubgridArray  *user_subgrids;

   int            num_user_subgrids;

   int            i;


   num_user_subgrids = GetIntDefault("UserGrid.NumSubgrids", 1);
			     
   /* read user_subgrids */
   user_all_subgrids = NewSubgridArray();
   user_subgrids = NewSubgridArray();
   for(i = 0; i < num_user_subgrids; i++)
   {
      AppendSubgrid(ReadUserSubgrid(), user_all_subgrids);
      AppendSubgrid(SubgridArraySubgrid(user_all_subgrids, i), user_subgrids);
   }
   
   /* create user_grid */
   user_grid = NewGrid(user_subgrids, user_all_subgrids);

   return user_grid;
}


/*--------------------------------------------------------------------------
 * FreeUserGrid
 *--------------------------------------------------------------------------*/

void  FreeUserGrid(user_grid)
Grid  *user_grid;
{
   FreeGrid(user_grid);
}

