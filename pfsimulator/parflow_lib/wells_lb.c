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
* Routine for setting up internal boundary conditions
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"

#define PRESSURE_WELL   0
#define FLUX_WELL       1

void LBWells(
             Lattice *    lattice,
             Problem *    problem,
             ProblemData *problem_data)
{
  /*--------------------------
   * Local variables
   *--------------------------*/

  /* Well data */
  WellData         *well_data = ProblemDataWellData(problem_data);
  WellDataPhysical *well_data_physical;
  WellDataValue    *well_data_value;

  /* Lattice variables */
  Vector *pressure = (lattice->pressure);
  CharVector *cellType = (lattice->cellType);
  double time = (lattice->t);

  /* Physical variables and coefficients */
  Subvector *sub_p;
  double    *pp;
  Subcharvector *sub_cellType;
  char      *cellTypep;
  double rho_g;
  double head, phead;
  double Z;

  /* Grid parameters and other structures */
  Grid      *grid = VectorGrid(pressure);
  Subgrid   *well_subgrid;
  Subgrid   *subgrid, *tmp_subgrid;
  int nx, ny, nz;
  double dz;
  int ix, iy, iz;
  int nx_v, ny_v, nz_v;

  /* Indices and counters */
  int grid_index;
  int i, j, k, index;
  int well;

  /* Time variables */
  TimeCycleData    *time_cycle_data;
  int cycle_number, interval_number;

  /*--------------------------
   *  Initializations
   *--------------------------*/
  rho_g = ProblemGravity(problem) * RHO;

  /*--------------------------
   *  Assign well boundary conditions
   *--------------------------*/
  if (WellDataNumPressWells(well_data) > 0)
  {
    time_cycle_data = WellDataTimeCycleData(well_data);

    for (grid_index = 0; grid_index < GridNumSubgrids(grid); grid_index++)
    {
      subgrid = GridSubgrid(grid, grid_index);

      sub_p = VectorSubvector(pressure, grid_index);
      sub_cellType = CharVectorSubcharvector(cellType, grid_index);

      for (well = 0; well < WellDataNumPressWells(well_data); well++)
      {
        well_data_physical = WellDataPressWellPhysical(well_data, well);
        cycle_number = WellDataPhysicalCycleNumber(well_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);

        well_data_value = WellDataPressWellIntervalValue(well_data, well, interval_number);

        well_subgrid = WellDataPhysicalSubgrid(well_data_physical);
        head = WellDataValuePhaseValue(well_data_value, 0);

        /*  Get the intersection of the well with the subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, well_subgrid)))
        {
          /*  If an intersection;  loop over it, and insert value  */
          ix = SubgridIX(tmp_subgrid);
          iy = SubgridIY(tmp_subgrid);
          iz = SubgridIZ(tmp_subgrid);

          nx = SubgridNX(tmp_subgrid);
          ny = SubgridNY(tmp_subgrid);
          nz = SubgridNZ(tmp_subgrid);

          nx_v = SubvectorNX(sub_p);
          ny_v = SubvectorNY(sub_p);
          nz_v = SubvectorNZ(sub_p);

          dz = SubgridDZ(well_subgrid);
          Z = RealSpaceZ(0, SubgridRZ(well_subgrid));

          pp = SubvectorElt(sub_p, ix, iy, iz);
          cellTypep = SubvectorElt(sub_cellType, ix, iy, iz);

          index = 0;
          BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                    index, nx_v, ny_v, nz_v, 1, 1, 1,
          {
            phead = head - rho_g * (Z + k * dz);
            pp[index] = phead;
            cellTypep[index] = 0;
          });

          /* done with this temporay subgrid */
          FreeSubgrid(tmp_subgrid);
        }      /* if (tmp_sub_grid ...)  */
      }     /* for (well = 0; ...)  */
    }    /*  for (grid_index = 0; ...)  */
  }   /* if ( WellDataNumPressWells(well_data) > 0 ) */
} /* End of function LBWells() */
