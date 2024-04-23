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
*
*****************************************************************************/

#include "parflow.h"

#include <assert.h>

/*--------------------------------------------------------------------------
 * This routine returns the elevations on a patch of a
 * solid at each (x,y) coordinate of an array of subgrid.  The result is
 * returned in an array of 2D real arrays.
 *
 * This routine is called by the pressure boundary condition routine and
 * by the pressure initial condition routine which calculate hydrostatic
 * conditions relative to a reference patch on a reference solid.
 *--------------------------------------------------------------------------*/

double         **CalcElevations(
                                GeomSolid *   geom_solid,
                                int           ref_patch,
                                SubgridArray *subgrids,
                                ProblemData * problem_data)
{
  GrGeomSolid        *grgeom_solid;

  GrGeomExtentArray  *extent_array;

  Background         *bg = GlobalsBackground;

  Subgrid            *subgrid;

  Vector      *z_mult = ProblemDataZmult(problem_data);
  Vector      *rsz = ProblemDataRealSpaceZ(problem_data);
  Subvector   *z_mult_sub;
  Subvector   *rsz_sub;
  double      *z_mult_dat;
  double      *rsz_dat;

  double            **elevation_arrays;
  double             *elevation_array;
  double dz2, zupper, zlower, zinit;

  int ix, iy, iz;
  int nx, ny, nz;
  int rz;

  int                *fdir;

  int is, i, j, k, iel, ival;


  /*-----------------------------------------------------
   * Convert the Geom solid to a GrGeom solid, making
   * sure that the extent_array extends all the way to
   * the top and bottom of the background.
   *
   * Also set some other miscellaneous values.
   *-----------------------------------------------------*/

  zlower = BackgroundZLower(bg);
  zupper = BackgroundZUpper(bg);
  extent_array = GrGeomCreateExtentArray(subgrids, 0, 0, 0, 0, -1, -1);
  zinit = 0.0;

  GrGeomSolidFromGeom(&grgeom_solid, geom_solid, extent_array);

  GrGeomFreeExtentArray(extent_array);

  /*-----------------------------------------------------
   * For each (x,y) point, determine the elevation
   * and construct the elevation_arrays.
   *-----------------------------------------------------*/

  elevation_arrays = ctalloc(double *, SubgridArraySize(subgrids));

  /*
   * SGS TODO SHOULD HAVE ASSERT MACRO
   * This algorithm only works for one subgrid per rank due to the merge process.
   * Unless we could guarantee subgrids are ordered on each rank.
   */
  assert(SubgridArraySize(subgrids) == 1);

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    z_mult_sub = VectorSubvector(z_mult, is);
    rsz_sub = VectorSubvector(rsz, is);
    z_mult_dat = SubvectorData(z_mult_sub);
    rsz_dat = SubvectorData(rsz_sub);

    /* RDF: assume resolutions are the same in all 3 directions */
    rz = SubgridRZ(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = IndexSpaceZ(zlower, rz);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = IndexSpaceZ(zupper, rz) - iz + 1;

    dz2 = RealSpaceDZ(rz) / 2.0;

    elevation_array = ctalloc(double, (nx * ny));

    /* Initialize the elevation_array */
    for (iel = 0; iel < (nx * ny); iel++)
      elevation_array[iel] = FLT_MAX;

    /* Construct elevation_array */
    GrGeomPatchLoop(i, j, k, fdir, grgeom_solid, ref_patch,
                    rz, ix, iy, iz, nx, ny, nz,
    {
      if (fdir[2] != 0)
      {
        iel = (j - iy) * nx + (i - ix);
        ival = SubvectorEltIndex(z_mult_sub, i, j, k);

        if (((i >= SubgridIX(subgrid)) && (i < (SubgridIX(subgrid) + SubgridNX(subgrid)))) &&
            ((j >= SubgridIY(subgrid)) && (j < (SubgridIY(subgrid) + SubgridNY(subgrid)))) &&
            ((k >= SubgridIZ(subgrid)) && (k < (SubgridIZ(subgrid) + SubgridNZ(subgrid)))))
        {
          elevation_array[iel] = rsz_dat[ival] + fdir[2] * dz2 * z_mult_dat[ival];
        }
      }
    });

    /*
     * SGS TODO SHOULD HAVE ASSERT MACRO
     * This algorithm only works for one subgrid per rank due to the merge process.
     * Unless we could guarantee subgrids are ordered on each rank.
     *
     * SGS TODO this algorithm is inefficient, currently sends all arrays from each rank in the Z
     * dimension to the bottom rank, performs a reduction and then sends up the column.
     * MPI has calls that will do this, likely more efficiently.   AMPS is a little limiting.
     */
    assert(SubgridArraySize(subgrids) == 1);

    if (GlobalsR)
    {
      /*
       * Send elevation array to lowest Rank in this Z column.
       */

      int num = nx * ny;

      amps_Invoice invoice = amps_NewInvoice("%*d", num, elevation_array);

      int dstRank = pqr_to_process(GlobalsP,
                                   GlobalsQ,
                                   0,
                                   GlobalsNumProcsX,
                                   GlobalsNumProcsY,
                                   GlobalsNumProcsZ);

      amps_Send(amps_CommWorld, dstRank, invoice);

      amps_FreeInvoice(invoice);

      /*
       * Receive
       */
      invoice = amps_NewInvoice("%*d", num, elevation_array);

      amps_Recv(amps_CommWorld, dstRank, invoice);

      amps_FreeInvoice(invoice);
    }
    else
    {
      int R;
      int num = nx * ny;
      double* temp_array = ctalloc(double, num);


      for (R = 1; R < GlobalsNumProcsZ; R++)
      {
        int dstRank = pqr_to_process(GlobalsP,
                                     GlobalsQ,
                                     R,
                                     GlobalsNumProcsX,
                                     GlobalsNumProcsY,
                                     GlobalsNumProcsZ);

        /*
         * Receive and reduce results from all processors.
         */
        amps_Invoice invoice = amps_NewInvoice("%*d", num, temp_array);

        amps_Recv(amps_CommWorld, dstRank, invoice);

        amps_FreeInvoice(invoice);

        /*
         * Reduction
         */
        for (iel = 0; iel < (nx * ny); iel++)
        {
          elevation_array[iel] = MIN(elevation_array[iel], temp_array[iel]);
        }
      }

      tfree(temp_array);

      /*
       * Original algorithm had default value of 0.0. This forces unset values to 0.0
       * after reduction.
       */
      for (iel = 0; iel < (nx * ny); iel++)
      {
        if (elevation_array[iel] == FLT_MAX)
        {
          elevation_array[iel] = zinit;
        }
      }


      /*
       * Send reduced array to other ranks in column
       */
      for (R = 1; R < GlobalsNumProcsZ; R++)
      {
        int dstRank = pqr_to_process(GlobalsP,
                                     GlobalsQ,
                                     R,
                                     GlobalsNumProcsX,
                                     GlobalsNumProcsY,
                                     GlobalsNumProcsZ);

        amps_Invoice invoice = amps_NewInvoice("%*d", num, elevation_array);
        amps_Send(amps_CommWorld, dstRank, invoice);
        amps_FreeInvoice(invoice);
      }
    }

    elevation_arrays[is] = elevation_array;
  }

  GrGeomFreeSolid(grgeom_solid);

  return elevation_arrays;
}


