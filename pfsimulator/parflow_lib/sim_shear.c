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


/*--------------------------------------------------------------------------
 * This routine returns the height of either the top or bottom of a
 * solid at each (x,y) coordinate of an array of subgrid.  The result is
 * returned in an array of 2D real arrays.  The `type' field determines
 * whether to look at the bottom or top of the solid given by the following:
 *    1 - bottom
 *    2 - top
 *
 * The routine is called by the subsurface simulation routines to simulate
 * more complex subsurface stratification (non-horizontal).
 *--------------------------------------------------------------------------*/

double         **SimShear(
                          double **     shear_min_ptr,
                          double **     shear_max_ptr,
                          GeomSolid *   geom_solid,
                          SubgridArray *subgrids,
                          int           type)
{
  GrGeomSolid        *grgeom_solid;
  double             *shear_min;
  double             *shear_max;

  GrGeomExtentArray  *extent_array = NULL;

  Background         *bg = GlobalsBackground;

  Subgrid            *subgrid;

  double            **shear_arrays;
  double             *shear_array;
  double z, dz2, zupper, zlower, zinit = 0.0;

  int ix, iy, iz;
  int nx, ny, nz;
  int rz;

  int                *fdir, dir = 0;

  int is, i, j, k, ishear;


  /*-----------------------------------------------------
   * Convert the Geom solid to a GrGeom solid, making
   * sure that the extent_array extends all the way to
   * either the top or bottom of the background, depending
   * on the parameter `type'.
   *
   * Also set some other miscellaneous values.
   *-----------------------------------------------------*/

  zlower = BackgroundZLower(bg);
  zupper = BackgroundZUpper(bg);
  switch (type)
  {
    case 1:
      extent_array =
        GrGeomCreateExtentArray(subgrids, 3, 3, 3, 3, -1, 3);
      zinit = zupper + 1.0;
      dir = -1;
      break;

    case 2:
      extent_array =
        GrGeomCreateExtentArray(subgrids, 3, 3, 3, 3, 3, -1);
      zinit = zlower - 1.0;
      dir = 1;
      break;
  }

  GrGeomSolidFromGeom(&grgeom_solid, geom_solid, extent_array);

  GrGeomFreeExtentArray(extent_array);

  /*-----------------------------------------------------
   * For each (x,y) point, determine the min/max
   * intersection points (depending on `type'), and
   * construct the shear_arrays.
   *-----------------------------------------------------*/

  shear_arrays = ctalloc(double *, SubgridArraySize(subgrids));
  shear_min = ctalloc(double, SubgridArraySize(subgrids));
  shear_max = ctalloc(double, SubgridArraySize(subgrids));

  ForSubgridI(is, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, is);

    /* RDF: assume resolutions are the same in all 3 directions */
    rz = SubgridRZ(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = IndexSpaceZ(zlower, rz);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = IndexSpaceZ(zupper, rz) - iz + 1;

    dz2 = RealSpaceDZ(rz) / 2.0;

    shear_array = ctalloc(double, (nx * ny));

    /* Initialize the shear_array to a value that will not change below */
    for (ishear = 0; ishear < (nx * ny); ishear++)
      shear_array[ishear] = zinit;

    /* Construct shear_array above/below visible solid surface */
    shear_min[is] = zupper;
    shear_max[is] = zlower;
    GrGeomSurfLoop(i, j, k, fdir, grgeom_solid, rz, ix, iy, iz, nx, ny, nz,
    {
      if (fdir[2] == dir)
      {
        ishear = (j - iy) * nx + (i - ix);
        z = RealSpaceZ(k, rz) + fdir[2] * dz2;

        switch (type)
        {
            case 1:
              shear_array[ishear] = pfmin(z, shear_array[ishear]);
              break;

            case 2:
              shear_array[ishear] = pfmax(z, shear_array[ishear]);
              break;
        }

        shear_min[is] = pfmin(shear_min[is], shear_array[ishear]);
        shear_max[is] = pfmax(shear_max[is], shear_array[ishear]);
      }
    });

    if (shear_min[is] > shear_max[is])
      shear_min[is] = shear_max[is] = 0.0;

    /* Construct shear_array away from visible solid surface */
    for (ishear = 0; ishear < (nx * ny); ishear++)
    {
      if (shear_array[ishear] == zinit)
      {
        switch (type)
        {
          case 1:
            shear_array[ishear] = shear_min[is];
            break;

          case 2:
            shear_array[ishear] = shear_max[is];
            break;
        }
      }
    }

    shear_arrays[is] = shear_array;
  }

  GrGeomFreeSolid(grgeom_solid);

  *shear_min_ptr = shear_min;
  *shear_max_ptr = shear_max;

  return shear_arrays;
}


