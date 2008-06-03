/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * DistributeUserGrid
 *
 * Distribute user grid into an array of subgrids.
 *
 *****************************************************************************/

#include <math.h>

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Macros for DistributeUserGrid
 *--------------------------------------------------------------------------*/

#define pqr_to_xyz(pqr, mxyz, lxyz, xyz)   (pqr*mxyz + min(pqr, lxyz) + xyz)

#define pqr_to_nxyz(pqr, mxyz, lxyz)  (pqr < lxyz ? mxyz+1 : mxyz)

#define pqr_to_process(p, q, r, P, Q, R)  ((((r)*(Q))+(q))*(P) + (p))


/*--------------------------------------------------------------------------
 * DistributeUserGrid:
 *   We currently assume that the user's grid consists of 1 subgrid only.
 *--------------------------------------------------------------------------*/

SubgridArray   *DistributeUserGrid(user_grid)
Grid           *user_grid;
{
   Subgrid     *user_subgrid = GridSubgrid(user_grid, 0);

   SubgridArray  *all_subgrids;

   int          num_procs;

   int          x, y, z;
   int          nx, ny, nz;

   int          P, Q, R;
   int          p, q, r;

   int          mx, my, mz, m;
   int          lx, ly, lz;


   nx = SubgridNX(user_subgrid);
   ny = SubgridNY(user_subgrid);
   nz = SubgridNZ(user_subgrid);

   /*-----------------------------------------------------------------------
    * User specifies process layout
    *-----------------------------------------------------------------------*/

   num_procs = GlobalsNumProcs;

   P = GlobalsNumProcsX;
   Q = GlobalsNumProcsY;
   R = GlobalsNumProcsZ;

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
   {
      if (!amps_Rank(amps_CommWorld))
         amps_Printf("Using process grid (%d,%d,%d)\n", P, Q, R);
   }
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
	    AppendSubgrid(NewSubgrid(pqr_to_xyz(p, mx, lx, x),
				     pqr_to_xyz(q, my, ly, y),
				     pqr_to_xyz(r, mz, lz, z),
				     pqr_to_nxyz(p, mx, lx),
				     pqr_to_nxyz(q, my, ly),
				     pqr_to_nxyz(r, mz, lz),
				     0, 0, 0,
				     pqr_to_process(p, q, r, P, Q, R)),
			  all_subgrids);
	 }

   return all_subgrids;
}

