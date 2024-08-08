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
* DistributeUserGrid
*
* Distribute user grid into an array of subgrids.
*
*****************************************************************************/

#include "parflow.h"
#include <math.h>

/*--------------------------------------------------------------------------
 * Macros for DistributeUserGrid
 *--------------------------------------------------------------------------*/

#define pqr_to_xyz(pqr, mxyz, lxyz, xyz)   (pqr*mxyz + pfmin(pqr, lxyz) + xyz)

#define pqr_to_nxyz(pqr, mxyz, lxyz)  (pqr < lxyz ? mxyz + 1 : mxyz)

/*--------------------------------------------------------------------------
 * DistributeUserGrid:
 *   We currently assume that the user's grid consists of 1 subgrid only.
 *--------------------------------------------------------------------------*/

SubgridArray   *DistributeUserGrid(
                                   Grid *user_grid)
{
  Subgrid     *user_subgrid = GridSubgrid(user_grid, 0);

  SubgridArray  *all_subgrids;

  int num_procs;

  int x, y, z;
  int nx, ny, nz;

  int P, Q, R;
  int p, q, r;

  int mx, my, mz, m;
  int lx, ly, lz;


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

  Grid *process_grid = ReadProcessGrid();

  /*
   * If user specified a process grid in input use that.
   *
   * NOTE: this is used mostly for debugging purposes so that one
   * can manually set the domain decomposition.
   */
  if (process_grid)
  {
    /*
     * These values do not make sense in this case;
     * don't make sense for the SAMRAI port with
     * multiple patches per processor.   There
     * is not PxQxR anymore.
     */
    GlobalsP = -99999;
    GlobalsQ = -99999;
    GlobalsR = -99999;

    all_subgrids = NewSubgridArray();

    SubgridArray* subgrid_array = GridAllSubgrids(process_grid);
    int i;
    ForSubgridI(i, subgrid_array)
    {
      Subgrid* new_subgrid = DuplicateSubgrid(SubgridArraySubgrid(subgrid_array, i));

      AppendSubgrid(new_subgrid, all_subgrids);
    }
  }
  else
  {
    /*-----------------------------------------------------------------------
     * Parflow specifies process layout
     *-----------------------------------------------------------------------*/

    if (!P || !Q || !R)
    {
      m = (int)pow((double)((nx * ny * nz) / num_procs), (1.0 / 3.0));

      do
      {
        P = nx / m;
        Q = ny / m;
        R = nz / m;

        P = P + ((nx % m) > P);
        Q = Q + ((ny % m) > Q);
        R = R + ((nz % m) > R);

        m++;
      }
      while ((P * Q * R) > num_procs);
    }

    /*-----------------------------------------------------------------------
     * Check P, Q, R with process allocation
     *-----------------------------------------------------------------------*/

    if ((P * Q * R) == num_procs)
    {
      static char first_call = 1;
      if (first_call && !amps_Rank(amps_CommWorld))
      {
        amps_Printf("Using process grid (%d,%d,%d)\n", P, Q, R);
        first_call = 0;
      }
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
    {
      for (q = 0; q < Q; q++)
      {
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

          if (pqr_to_process(p, q, r, P, Q, R) == amps_Rank(amps_CommWorld))
          {
            GlobalsP = p;
            GlobalsQ = q;
            GlobalsR = r;
          }
        }
      }
    }
  }

  FreeGrid(process_grid);

  return all_subgrids;
}

