/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

static struct {
   Grid *grid;
   int   num_ghost;
} pf2kinsol_data;


void SetPf2KinsolData(grid, num_ghost)
Grid        *grid;
int          num_ghost;
{
   pf2kinsol_data.grid = grid;
   pf2kinsol_data.num_ghost = num_ghost;
}

N_Vector N_VNew(N, machEnv)
int    N;
void  *machEnv;
{
   Grid    *grid;
   int      num_ghost;

   grid      = pf2kinsol_data.grid;
   num_ghost = pf2kinsol_data.num_ghost;
   return(NewVector(grid, 1, num_ghost));
}

void N_VPrint(x)
N_Vector x;
{
  Grid       *grid     = VectorGrid(x);
  Subgrid    *subgrid;
 
  Subvector  *x_sub;

  double     *xp;

  int         ix,   iy,   iz;
  int         nx,   ny,   nz;
  int         nx_x, ny_x, nz_x;
  
  int         sg, i, j, k, i_x;

  ForSubgridI(sg, GridSubgrids(grid))
  {
     subgrid = GridSubgrid(grid, sg);

     x_sub = VectorSubvector(x, sg);

     ix = SubgridIX(subgrid);
     iy = SubgridIY(subgrid);
     iz = SubgridIZ(subgrid);

     nx = SubgridNX(subgrid);
     ny = SubgridNY(subgrid);
     nz = SubgridNZ(subgrid);

     nx_x = SubvectorNX(x_sub);
     ny_x = SubvectorNY(x_sub);
     nz_x = SubvectorNZ(x_sub);

     xp = SubvectorElt(x_sub, ix, iy, iz);

     i_x = 0;
     BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
	       i_x, nx_x, ny_x, nz_x, 1, 1, 1,
	       {
		  printf("%g\n", xp[i_x]);
                  fflush(NULL);
	       });
  }
  printf("\n");
  fflush(NULL);
}
