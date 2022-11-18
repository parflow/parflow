/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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

#define PRESSURE_RESERVOIR   0
#define FLUX_RESERVOIR       1

void LBReservoirs(
    Lattice *    lattice,
    Problem *    problem,
    ProblemData *problem_data)
{
  /*--------------------------
   * Local variables
   *--------------------------*/
  printf("We are in LB reservoirs\n");
  /* Reservoir data */
  ReservoirData         *reservoir_data = ProblemDataReservoirData(problem_data);
  ReservoirDataPhysical *reservoir_data_physical;
  ReservoirDataValue    *reservoir_data_value;

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
  Subgrid   *reservoir_subgrid;
  Subgrid   *subgrid, *tmp_subgrid;
  int nx, ny, nz;
  double dz;
  int ix, iy, iz;
  int nx_v, ny_v, nz_v;

  /* Indices and counters */
  int grid_index;
  int i, j, k, index;
  int reservoir;

  /* Time variables */
  TimeCycleData    *time_cycle_data;
  int cycle_number, interval_number;

  /*--------------------------
   *  Initializations
   *--------------------------*/
  rho_g = ProblemGravity(problem) * RHO;

  /*--------------------------
   *  Assign reservoir boundary conditions
   *--------------------------*/
  if (ReservoirDataNumPressReservoirs(reservoir_data) > 0)
  {
    time_cycle_data = ReservoirDataTimeCycleData(reservoir_data);

    for (grid_index = 0; grid_index < GridNumSubgrids(grid); grid_index++)
    {
      subgrid = GridSubgrid(grid, grid_index);

      sub_p = VectorSubvector(pressure, grid_index);
      sub_cellType = CharVectorSubcharvector(cellType, grid_index);

      for (reservoir = 0; reservoir < ReservoirDataNumPressReservoirs(reservoir_data); reservoir++)
      {
        reservoir_data_physical = ReservoirDataPressReservoirPhysical(reservoir_data, reservoir);
        cycle_number = ReservoirDataPhysicalCycleNumber(reservoir_data_physical);
        interval_number = TimeCycleDataComputeIntervalNumber(problem, time, time_cycle_data, cycle_number);

        reservoir_data_value = ReservoirDataPressReservoirIntervalValue(reservoir_data, reservoir, interval_number);

        reservoir_subgrid = ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical);
        head = ReservoirDataValuePhaseValue(reservoir_data_value, 0);

        /*  Get the intersection of the reservoir with the intake_subgrid  */
        if ((tmp_subgrid = IntersectSubgrids(subgrid, reservoir_subgrid)))
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

          dz = SubgridDZ(reservoir_subgrid);
          Z = RealSpaceZ(0, SubgridRZ(reservoir_subgrid));

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

          /* done with this temporay intake_subgrid */
          FreeSubgrid(tmp_subgrid);
        }      /* if (tmp_sub_grid ...)  */
      }     /* for (reservoir = 0; ...)  */
    }    /*  for (grid_index = 0; ...)  */
  }   /* if ( ReservoirDataNumPressReservoirs(reservoir_data) > 0 ) */
} /* End of function LBReservoirs() */