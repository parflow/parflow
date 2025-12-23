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

#include "parflow.h"

#include <math.h>

void OverlandSum(ProblemData *problem_data,
                 Vector *     pressure,       /* Current pressure values */
                 double       dt,
                 Vector *     overland_sum)
{
  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);

  double dx, dy;
  int i, j, is;
  int ix, iy;
  int nx, ny;

  Subgrid     *subgrid;
  Grid        *grid = VectorGrid(pressure);

  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Vector      *mannings = ProblemDataMannings(problem_data);
  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);

  Subvector   *overland_sum_subvector;
  Subvector   *slope_x_subvector;
  Subvector   *slope_y_subvector;
  Subvector   *mannings_subvector;
  Subvector   *pressure_subvector;
  Subvector   *top_subvector;

  int index_overland_sum;
  int index_slope_x;
  int index_slope_y;
  int index_mannings;
  int index_pressure;
  int index_top;

  double *overland_sum_ptr;
  double *slope_x_ptr;
  double *slope_y_ptr;
  double *mannings_ptr;
  double *pressure_ptr;
  double *top_ptr;

  int ipatch;

  BCStruct    *bc_struct;

  BCPressureData *bc_pressure_data = ProblemDataBCPressureData(problem_data);
  int num_patches = BCPressureDataNumPatches(bc_pressure_data);

  bc_struct = NewBCStruct(GridSubgrids(grid),
                          gr_domain,
                          num_patches,
                          BCPressureDataPatchIndexes(bc_pressure_data),
                          BCPressureDataBCTypes(bc_pressure_data),
                          NULL);

  if (num_patches > 0)
  {
    for (ipatch = 0; ipatch < num_patches; ipatch++)
    {
      switch (BCPressureDataType(bc_pressure_data, ipatch))
      {
        case 7:
        {
          ForSubgridI(is, GridSubgrids(grid))
          {
            subgrid = GridSubgrid(grid, is);

            overland_sum_subvector = VectorSubvector(overland_sum, is);
            slope_x_subvector = VectorSubvector(slope_x, is);
            slope_y_subvector = VectorSubvector(slope_y, is);
            mannings_subvector = VectorSubvector(mannings, is);
            pressure_subvector = VectorSubvector(pressure, is);
            top_subvector = VectorSubvector(top, is);

            ix = SubgridIX(subgrid);
            iy = SubgridIY(subgrid);

            nx = SubgridNX(subgrid);
            ny = SubgridNY(subgrid);

            dx = SubgridDX(subgrid);
            dy = SubgridDY(subgrid);

            overland_sum_ptr = SubvectorData(overland_sum_subvector);
            slope_x_ptr = SubvectorData(slope_x_subvector);
            slope_y_ptr = SubvectorData(slope_y_subvector);
            mannings_ptr = SubvectorData(mannings_subvector);
            pressure_ptr = SubvectorData(pressure_subvector);
            top_ptr = SubvectorData(top_subvector);

            int state;
            const int inactive = -1;
            const int active = 1;

            for (i = ix; i < ix + nx; i++)
            {
              j = iy - 1;

              index_top = SubvectorEltIndex(top_subvector, i, j, 0);
              int k = (int)top_ptr[index_top];

              if (k < 0)
              {
                state = inactive;
              }
              else
              {
                state = active;
              }

              while (j < iy + ny)
              {
                if (state == inactive)
                {
                  index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                  k = (int)top_ptr[index_top];
                  while (k < 0 && j <= iy + ny)
                  {
                    j++;
                    index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                    k = (int)top_ptr[index_top];
                  }

                  // If still in interior
                  if (j < iy + ny)
                  {
                    if (k >= 0)
                    {
                      // inactive to active

                      index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);

                      // sloping to inactive active from active
                      if (slope_y_ptr[index_slope_y] > 0)
                      {
                        index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                        if (pressure_ptr[index_pressure] > 0)
                        {
                          index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                          index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);

                          overland_sum_ptr[index_overland_sum] +=
                            (sqrt(fabs(slope_y_ptr[index_slope_y])) / mannings_ptr[index_mannings]) *
                            pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dx * dt;
                        }
                      }
                    }

                    state = active;
                  }
                }
                else
                {
                  index_top = SubvectorEltIndex(top_subvector, i, j + 1, 0);
                  k = (int)top_ptr[index_top];
                  while (k >= 0 && j <= iy + ny)
                  {
                    j++;
                    index_top = SubvectorEltIndex(top_subvector, i, j + 1, 0);
                    k = (int)top_ptr[index_top];
                  }

                  // If still in interior
                  if (j < iy + ny)
                  {
                    index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                    k = (int)top_ptr[index_top];

                    // active to inactive


                    index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);

                    // sloping from active to inactive
                    if (slope_y_ptr[index_slope_y] < 0)
                    {
                      index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                      if (pressure_ptr[index_pressure] > 0)
                      {
                        index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                        index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);

                        overland_sum_ptr[index_overland_sum] +=
                          (sqrt(fabs(slope_y_ptr[index_slope_y])) / mannings_ptr[index_mannings]) *
                          pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dx * dt;
                      }
                    }
                  }

                  state = inactive;
                }
                j++;
              }
            }

#if 0
            for (i = ix; i < ix + nx; i++)
            {
              for (j = iy; j < iy + ny; j++)
              {
                index_top = SubvectorEltIndex(top_subvector, i, j, 0);

                int k = (int)top_ptr[index_top];
                if (!(k < 0))
                {
                  /*
                   * Compute runnoff if slope is running off of active region
                   */
                  index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                  index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);
                  index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
                  index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);
                  index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                  if (slope_y_ptr[index_slope_y] > 0)
                  {
                    if (pressure_ptr[index_pressure] > 0)
                    {
                      overland_sum_ptr[index_overland_sum] +=
                        (sqrt(fabs(slope_y_ptr[index_slope_y])) / mannings_ptr[index_mannings]) *
                        pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dx * dt;
                    }
                  }

                  /*
                   * Loop until going back outside of active area
                   */
                  while ((j + 1 < iy + ny) && !(top_ptr[SubvectorEltIndex(top_subvector, i, j + 1, 0)] < 0))
                  {
                    j++;
                  }

                  /*
                   * Found either domain boundary or outside of active area.
                   * Compute runnoff if slope is running off of active region.
                   */

                  index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                  k = (int)top_ptr[index_top];
                  index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                  index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);
                  index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
                  index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);
                  index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                  if (slope_y_ptr[index_slope_y] < 0)
                  {
                    if (pressure_ptr[index_pressure] > 0)
                    {
                      overland_sum_ptr[index_overland_sum] +=
                        (sqrt(fabs(slope_y_ptr[index_slope_y])) / mannings_ptr[index_mannings]) *
                        pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dx * dt;
                    }
                  }
                }
              }
            }
#endif


            for (j = iy; j < iy + ny; j++)
            {
              i = ix - 1;

              index_top = SubvectorEltIndex(top_subvector, i, j, 0);
              int k = (int)top_ptr[index_top];

              if (k < 0)
              {
                state = inactive;
              }
              else
              {
                state = active;
              }

              while (i < ix + nx)
              {
                if (state == inactive)
                {
                  index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                  k = (int)top_ptr[index_top];
                  while (k < 0 && i <= ix + nx)
                  {
                    i++;
                    index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                    k = (int)top_ptr[index_top];
                  }

                  // If still in interior
                  if (i < ix + nx)
                  {
                    if (k >= 0)
                    {
                      // inactive to active

                      index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);

                      // sloping to inactive active from active
                      if (slope_x_ptr[index_slope_x] > 0)
                      {
                        index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                        if (pressure_ptr[index_pressure] > 0)
                        {
                          index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                          index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);

                          overland_sum_ptr[index_overland_sum] +=
                            (sqrt(fabs(slope_x_ptr[index_slope_x])) / mannings_ptr[index_mannings]) *
                            pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dy * dt;
                        }
                      }
                    }

                    state = active;
                  }
                }
                else
                {
                  index_top = SubvectorEltIndex(top_subvector, i + 1, j, 0);
                  k = (int)top_ptr[index_top];
                  while (k >= 0 && i <= ix + nx)
                  {
                    i++;
                    index_top = SubvectorEltIndex(top_subvector, i + 1, j, 0);
                    k = (int)top_ptr[index_top];
                  }

                  // If still in interior
                  if (i < ix + nx)
                  {
                    index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                    k = (int)top_ptr[index_top];

                    // active to inactive
                    index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);

                    // sloping from active to inactive
                    if (slope_x_ptr[index_slope_x] < 0)
                    {
                      index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                      if (pressure_ptr[index_pressure] > 0)
                      {
                        index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                        index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);

                        overland_sum_ptr[index_overland_sum] +=
                          (sqrt(fabs(slope_x_ptr[index_slope_x])) / mannings_ptr[index_mannings]) *
                          pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dy * dt;
                      }
                    }
                  }

                  state = inactive;
                }
                i++;
              }
            }

#if 0
            for (j = iy; j < iy + ny; j++)
            {
              for (i = ix; i < ix + nx; i++)
              {
                index_top = SubvectorEltIndex(top_subvector, i, j, 0);

                int k = (int)top_ptr[index_top];
                if (!(k < 0))
                {
                  /*
                   * Compute runnoff if slope is running off of active region
                   */
                  index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                  index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);
                  index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
                  index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);
                  index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                  if (slope_x_ptr[index_slope_x] > 0)
                  {
                    if (pressure_ptr[index_pressure] > 0)
                    {
                      overland_sum_ptr[index_overland_sum] +=
                        (sqrt(fabs(slope_x_ptr[index_slope_y])) / mannings_ptr[index_mannings]) *
                        pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dy * dt;
                    }
                  }

                  /*
                   * Loop until going back outside of active area
                   */
                  while ((i + 1 < ix + nx) && !(top_ptr[SubvectorEltIndex(top_subvector, i + 1, j, 0)] < 0))
                  {
                    i++;
                  }

                  /*
                   * Found either domain boundary or outside of active area.
                   * Compute runnoff if slope is running off of active region.
                   */
                  index_top = SubvectorEltIndex(top_subvector, i, j, 0);
                  k = (int)top_ptr[index_top];
                  index_overland_sum = SubvectorEltIndex(overland_sum_subvector, i, j, 0);
                  index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);
                  index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
                  index_mannings = SubvectorEltIndex(mannings_subvector, i, j, 0);
                  index_pressure = SubvectorEltIndex(pressure_subvector, i, j, k);

                  if (slope_x_ptr[index_slope_x] < 0)
                  {
                    if (pressure_ptr[index_pressure] > 0)
                    {
                      overland_sum_ptr[index_overland_sum] +=
                        (sqrt(fabs(slope_x_ptr[index_slope_x])) / mannings_ptr[index_mannings]) *
                        pow(pressure_ptr[index_pressure], 5.0 / 3.0) * dy * dt;
                    }
                  }
                }
              }
            }
#endif
          }
        }
      }
    }
  }

  FreeBCStruct(bc_struct);
}

