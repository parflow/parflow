/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.8 $
 *********************************************************************EHEADER*/

#include <math.h>
#include "pftools.h"

/*--------------------------------------------------------------------------
 * ReadBackground
 *--------------------------------------------------------------------------*/

Background  *ReadBackground(Tcl_Interp *interp)
{
   Background    *background;

   background = talloc(Background, 1);

   BackgroundX(background) = GetDouble(interp, "ComputationalGrid.Lower.X");
   BackgroundY(background) = GetDouble(interp, "ComputationalGrid.Lower.Y");
   BackgroundZ(background) = GetDouble(interp, "ComputationalGrid.Lower.Z");

   BackgroundDX(background) = GetDouble(interp, "ComputationalGrid.DX");
   BackgroundDY(background) = GetDouble(interp, "ComputationalGrid.DY");
   BackgroundDZ(background) = GetDouble(interp, "ComputationalGrid.DZ");
			     
   return background;
}


/*--------------------------------------------------------------------------
 * FreeBackground
 *--------------------------------------------------------------------------*/

void         FreeBackground(background)
Background  *background;
{
   free(background);
}


/*--------------------------------------------------------------------------
 * ReadUserSubgrid
 *--------------------------------------------------------------------------*/

Subgrid    *ReadUserSubgrid(Tcl_Interp *interp)
{
    Subgrid  *new;

    int       ix, iy, iz;
    int       nx, ny, nz;
    int       rx, ry, rz;

    ix = GetIntDefault(interp, "UserGrid.IX", 0);
    iy = GetIntDefault(interp, "UserGrid.IY", 0);
    iz = GetIntDefault(interp, "UserGrid.IZ", 0);

    rx = GetIntDefault(interp, "UserGrid.RX", 0);
    ry = GetIntDefault(interp, "UserGrid.RY", 0);
    rz = GetIntDefault(interp, "UserGrid.RZ", 0);

    nx = GetInt(interp, "ComputationalGrid.NX");
    ny = GetInt(interp, "ComputationalGrid.NY");
    nz = GetInt(interp, "ComputationalGrid.NZ");

    new = NewSubgrid(ix, iy, iz, nx, ny, nz, rx, ry, rz, -1);

    return new;
}


/*--------------------------------------------------------------------------
 * ReadUserGrid
 *--------------------------------------------------------------------------*/

Grid      *ReadUserGrid(Tcl_Interp *interp)
{
   Grid          *user_grid;

   SubgridArray  *user_subgrids;

   int            num_user_subgrids;

   num_user_subgrids = GetIntDefault(interp, "UserGrid.NumSubgrids", 1);
   
   /* read user_subgrids */
   user_subgrids = NewSubgridArray();
   for(; num_user_subgrids; num_user_subgrids--)
      AppendSubgrid(ReadUserSubgrid(interp), &user_subgrids);
   
   /* create user_grid */
   user_grid = NewGrid(user_subgrids, NewSubgridArray(), NewSubgridArray());

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


/*--------------------------------------------------------------------------
 * DistributeUserGrid:
 *   We currently assume that the user's grid consists of 1 subgrid only.
 *--------------------------------------------------------------------------*/

#define pqr_to_xyz(pqr, mxyz, lxyz, xyz)   (pqr*mxyz + min(pqr, lxyz) + xyz)

#define pqr_to_nxyz(pqr, mxyz, lxyz)  (pqr < lxyz ? mxyz+1 : mxyz)

#define pqr_to_process(p, q, r, P, Q, R)  ((((r)*(Q))+(q))*(P) + (p))


SubgridArray   *DistributeUserGrid(user_grid, num_procs, P, Q, R)
Grid           *user_grid;
int             num_procs;
int             P;
int             Q;
int             R;
{
   Subgrid     *user_subgrid = GridSubgrid(user_grid, 0);

   SubgridArray  *all_subgrids;

   int          process;

   int          x, y, z;
   int          nx, ny, nz;

   int          p, q, r;

   int          mx, my, mz, m;
   int          lx, ly, lz;


   nx = SubgridNX(user_subgrid);
   ny = SubgridNY(user_subgrid);
   nz = SubgridNZ(user_subgrid);

   /*-----------------------------------------------------------------------
    * Parflow specifies process layout
    *-----------------------------------------------------------------------*/

   if (!P || !Q || !R)
   {
      m = (int) pow((double) ((nx*ny*nz) / num_procs), (1.0 / 3.0));

      do
      {
	 P  = nx / m;
	 Q  = ny / m;
	 R  = nz / m;

	 P = P + ((nx % m) > P);
	 Q = Q + ((ny % m) > Q);
	 R = R + ((nz % m) > R);

	 m++;

      } while ((P*Q*R) > num_procs);
   }

   /*-----------------------------------------------------------------------
    * Check P, Q, R with process allocation
    *-----------------------------------------------------------------------*/

   if ((P*Q*R) == num_procs)
      printf("Using process grid (%d,%d,%d)\n", P, Q, R);
   else
      return NULL;

   /*-----------------------------------------------------------------------
    * Create all_subgrids
    *-----------------------------------------------------------------------*/

   all_subgrids = NewSubgridArray();

   x = SubgridIX(user_subgrid);
   y = SubgridIY(user_subgrid);
   z = SubgridIZ(user_subgrid);

   mx = nx / P;
   my = ny / Q;
   mz = nz / R;

   lx = (nx % P);
   ly = (ny % Q);
   lz = (nz % R);

   for (p = 0; p < P; p++)
      for (q = 0; q < Q; q++)
	 for (r = 0; r < R; r++)
	 {
	    process = pqr_to_process(p, q, r, P, Q, R);

	    AppendSubgrid(NewSubgrid(pqr_to_xyz(p, mx, lx, x),
				     pqr_to_xyz(q, my, ly, y),
				     pqr_to_xyz(r, mz, lz, z),
				     pqr_to_nxyz(p, mx, lx),
				     pqr_to_nxyz(q, my, ly),
				     pqr_to_nxyz(r, mz, lz),
				     0, 0, 0, process),
			  &all_subgrids);
	 }

   return all_subgrids;
}

