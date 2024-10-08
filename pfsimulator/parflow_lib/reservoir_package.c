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

#include "parflow.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/
typedef struct {
  /* reservoir info */
  int num_reservoirs;

  int       *type;
  void     **data;

  int overland_flow_solver;
  NameArray reservoir_names;
} PublicXtra;

typedef void InstanceXtra;

typedef struct {
  char    *name;
  double intake_x_location, intake_y_location;
  int has_secondary_intake_cell;
  double secondary_intake_x_location, secondary_intake_y_location;
  double release_x_location, release_y_location;
  double max_storage, min_release_storage, storage, release_rate;
} Type0;                      /* basic vertical reservoir */

/*--------------------------------------------------------------------------
 * ReservoirPackage
 *--------------------------------------------------------------------------*/

// Temp solution while I get this fix checked in as part of it's own independent module that handles
// var dz correctly. Works for all cases that I am aware of. (Ben West)
/** @brief Calculates a subgrids total volume, accounting for variable dz. Assumes subgrid is fully
 * contained within the current rank.
 *
 * @param subgrid the subgrid we are checking
 * @param problem_data the problems problem data structure
 * @return the volume of the subgrid
 */
double GetSubgridVolume(Subgrid *subgrid, ProblemData* problem_data)
{
  double dx = SubgridDX(subgrid);
  double dy = SubgridDY(subgrid);
  double dz = SubgridDZ(subgrid);
  GrGeomSolid *gr_domain = problem_data->gr_domain;

  double volume = 0;
  SubgridArray   *subgrids = problem_data->dz_mult->grid->subgrids;
  Subgrid        *tmp_subgrid;
  int subgrid_index;

  ForSubgridI(subgrid_index, subgrids)
  {
    tmp_subgrid = SubgridArraySubgrid(subgrids, subgrid_index);
    Subvector *dz_mult_subvector = VectorSubvector(problem_data->dz_mult, subgrid_index);
    double* dz_mult_data = SubvectorData(dz_mult_subvector);
    Subgrid *intersection = IntersectSubgrids(subgrid, tmp_subgrid);
    int nx = SubgridNX(intersection);
    int ny = SubgridNY(intersection);
    int nz = SubgridNZ(intersection);
    int r = SubgridRZ(intersection);
    int ix = SubgridIX(intersection);
    int iy = SubgridIY(intersection);
    int iz = SubgridIZ(intersection);
    int i, j, k;
    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      int index = SubvectorEltIndex(dz_mult_subvector, i, j, k);
      double dz_mult = dz_mult_data[index];
      volume += dz_mult * dx * dy * dz;
    });
  }
  return volume;
};


/** @brief Checks whether a subgrid intersects with the current ranks subgrid
 *
 * @param subgrid the subgrid we are checking
 * @param grid the problems grid
 * @return True or False corresponding to whether the subgrid intersects
 */
bool SubgridLivesOnThisRank(Subgrid* subgrid, Grid *grid)
{
  int subgrid_index;
  Subgrid* rank_subgrid, *tmp_subgrid;

  ForSubgridI(subgrid_index, GridSubgrids(grid))
  {
    rank_subgrid = SubgridArraySubgrid(GridSubgrids(grid), subgrid_index);
    if ((tmp_subgrid = IntersectSubgrids(rank_subgrid, subgrid)))
    {
      return true;
    }
  }
  return false;
}

/** @brief Sets the slops at the outlet faces of a cell to 0 to stop flow.
 * Assumes an overlandkinematic boundary condition
 *
 *
 *
 * @param i the x index of the cell in question
 * @param j the y index of the cell in question
 * @return Null, but modifies the problem datas x and y slopes
 */
void StopOutletFlowAtCellOverlandKinematic(int i, int j, ProblemData* problem_data, Grid* grid)
{
  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Subvector   *slope_x_subvector;
  Subvector   *slope_y_subvector;
  int index_slope_x;
  int index_slope_y;
  double *slope_x_ptr;
  double *slope_y_ptr;
  int subgrid_index;
  Subgrid *subgrid;

  ForSubgridI(subgrid_index, GridSubgrids(grid))
  {
    int subgrid_x_floor, subgrid_y_floor, subgrid_x_ceiling, subgrid_y_ceiling;

    subgrid = GridSubgrid(grid, subgrid_index);
    slope_x_subvector = VectorSubvector(slope_x, subgrid_index);
    slope_y_subvector = VectorSubvector(slope_y, subgrid_index);
    slope_x_ptr = SubvectorData(slope_x_subvector);
    slope_y_ptr = SubvectorData(slope_y_subvector);
    index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
    subgrid_x_floor = SubgridIX(subgrid);
    subgrid_y_floor = SubgridIY(subgrid);
    subgrid_x_ceiling = subgrid_x_floor + SubgridNX(subgrid) - 1;
    subgrid_y_ceiling = subgrid_y_floor + SubgridNY(subgrid) - 1;

    // Check all 4 faces, as long as they live on this subgrid. First the East face
    if (i + 1 >= subgrid_x_floor && i + 1 <= subgrid_x_ceiling && j >= subgrid_y_floor && j <= subgrid_y_ceiling)
    {
      index_slope_x = SubvectorEltIndex(slope_x_subvector, i + 1, j, 0);
      if (slope_x_ptr[index_slope_x] > 0)
      {
        slope_x_ptr[index_slope_x] = 0;
      }
    }
    // South face
    if (i >= subgrid_x_floor && i <= subgrid_x_ceiling && j - 1 >= subgrid_y_floor && j - 1 <= subgrid_y_ceiling)
    {
      index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j - 1, 0);
      if (slope_y_ptr[index_slope_y] > 0)
      {
        slope_y_ptr[index_slope_y] = 0;
      }
    }
    if (i >= subgrid_x_floor && i <= subgrid_x_ceiling && j >= subgrid_y_floor && j <= subgrid_y_ceiling)
    {
      index_slope_x = SubvectorEltIndex(slope_x_subvector, i, j, 0);
      if (slope_x_ptr[index_slope_x] < 0)
      {
        slope_x_ptr[index_slope_x] = 0;
      }
    }
    // South face
    if (i >= subgrid_x_floor && i <= subgrid_x_ceiling && j >= subgrid_y_floor && j <= subgrid_y_ceiling)
    {
      index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
      if (slope_y_ptr[index_slope_y] < 0)
      {
        slope_y_ptr[index_slope_y] = 0;
      }
    }
  }
}

/** @brief Sets the slops at the outlet faces of a cell to 0 to stop flow.
 * Assumes an overlandflow boundary condition
 *
 *
 *
 * @param i the x index of the cell in question
 * @param j the y index of the cell in question
 * @return Null, but modifies the problem datas x and y slopes
 */
void StopOutletFlowAtCellOverlandFlow(int i, int j, ProblemData* problem_data, Grid* grid)
{
  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Subvector   *slope_x_subvector;
  Subvector   *slope_y_subvector;
  int index_slope_x;
  int index_slope_y;
  double *slope_x_ptr;
  double *slope_y_ptr;
  int subgrid_index;

  ForSubgridI(subgrid_index, GridSubgrids(grid))
  {
    slope_x_subvector = VectorSubvector(slope_x, subgrid_index);
    slope_y_subvector = VectorSubvector(slope_y, subgrid_index);
    index_slope_y = SubvectorEltIndex(slope_y_subvector, i, j, 0);
    index_slope_x = SubvectorEltIndex(slope_y_subvector, i, j, 0);
    slope_x_ptr = SubvectorData(slope_x_subvector);
    slope_y_ptr = SubvectorData(slope_y_subvector);
    slope_x_ptr[index_slope_x] = 0;
    slope_y_ptr[index_slope_y] = 0;
  }
}

void         ReservoirPackage(
                              ProblemData *problem_data)
{
  PFModule         *this_module = ThisPFModule;
  PublicXtra       *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0            *dummy0;

  Subgrid          *new_intake_subgrid;
  Subgrid          *new_secondary_intake_subgrid;
  Subgrid          *new_release_subgrid;

  ReservoirData         *reservoir_data = ProblemDataReservoirData(problem_data);
  ReservoirDataPhysical *reservoir_data_physical;

  int i, reservoir_index;

  int intake_ix, intake_iy;
  int secondary_intake_ix, secondary_intake_iy;
  int release_ix, release_iy;
  int iz_lower;
  int nx, ny, nz;
  int rx, ry, rz;
  int current_mpi_rank;
  int intake_cell_rank, secondary_intake_cell_rank, release_cell_rank;
  int split_color;
  int grid_nz;

  double release_subgrid_volume;
  bool part_of_reservoir_lives_on_this_rank;

  /* Allocate the reservoir data */

  ReservoirDataNumReservoirs(reservoir_data) = (public_xtra->num_reservoirs);
  //it feels like there should be a better way to get the grid than this,but I couldn't find one...
  Grid* grid = VectorGrid(problem_data->rsz);
  grid_nz = grid->background->nz;

  if ((public_xtra->num_reservoirs) > 0)
  {
    ReservoirDataReservoirPhysicals(reservoir_data) = ctalloc(ReservoirDataPhysical *, (public_xtra->num_reservoirs));
  }

  reservoir_index = 0;

  if ((public_xtra->num_reservoirs) > 0)
  {
    /* Load the reservoir data */
    for (i = 0; i < (public_xtra->num_reservoirs); i++)
    {
      dummy0 = (Type0*)(public_xtra->data[i]);
      reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
      intake_ix = IndexSpaceX((dummy0->intake_x_location), 0);
      intake_iy = IndexSpaceY((dummy0->intake_y_location), 0);
      secondary_intake_ix = IndexSpaceX((dummy0->secondary_intake_x_location), 0);
      secondary_intake_iy = IndexSpaceY((dummy0->secondary_intake_y_location), 0);
      release_ix = IndexSpaceX((dummy0->release_x_location), 0);
      release_iy = IndexSpaceY((dummy0->release_y_location), 0);

      secondary_intake_cell_rank = -1;
      release_cell_rank = -1;

      iz_lower = grid_nz - 1;

      nx = 1;
      ny = 1;
      nz = 1;

      rx = 0;
      ry = 0;
      rz = 0;
#ifdef PARFLOW_HAVE_MPI
      current_mpi_rank = amps_Rank(amps_CommWorld);
#else
      current_mpi_rank = 1;
#endif

      new_intake_subgrid = NewSubgrid(intake_ix, intake_iy, iz_lower,
                                      nx, ny, nz,
                                      rx, ry, rz,
                                      current_mpi_rank);

      new_secondary_intake_subgrid = NewSubgrid(secondary_intake_ix, secondary_intake_iy, iz_lower,
                                                nx, ny, nz,
                                                rx, ry, rz,
                                                current_mpi_rank);

      new_release_subgrid = NewSubgrid(release_ix, release_iy, iz_lower,
                                       nx, ny, nz,
                                       rx, ry, rz,
                                       current_mpi_rank);

      if (SubgridLivesOnThisRank(new_intake_subgrid, grid))
      {
        intake_cell_rank = current_mpi_rank;
      }

      if (SubgridLivesOnThisRank(new_secondary_intake_subgrid, grid))
      {
        secondary_intake_cell_rank = current_mpi_rank;
      }
      release_subgrid_volume = 1;
      if (SubgridLivesOnThisRank(new_release_subgrid, grid))
      {
        release_cell_rank = current_mpi_rank;
        release_subgrid_volume = GetSubgridVolume(new_release_subgrid, problem_data);
      }
      //If we are multiprocessor we need to do some reductions to determine the correct ranks
      // for the reservoirs
#ifdef PARFLOW_HAVE_MPI
      amps_Invoice reservoir_cells_invoice = amps_NewInvoice("%i%i%i", &intake_cell_rank, &secondary_intake_cell_rank, &release_cell_rank);
      amps_AllReduce(amps_CommWorld, reservoir_cells_invoice, amps_Max);
      amps_FreeInvoice(reservoir_cells_invoice);

      part_of_reservoir_lives_on_this_rank =
        (release_cell_rank == current_mpi_rank) ||
        (intake_cell_rank == current_mpi_rank) ||
        (secondary_intake_cell_rank == current_mpi_rank);

      MPI_Comm new_reservoir_communicator;
      split_color = part_of_reservoir_lives_on_this_rank ? 1 : MPI_UNDEFINED;
      MPI_Comm_split(amps_CommWorld, split_color, current_mpi_rank, &new_reservoir_communicator);
#endif
//     edit the slopes to prevent stuff running through the reservoir
      if (ReservoirDataOverlandFlowSolver(reservoir_data) == OVERLAND_FLOW)
      {
        StopOutletFlowAtCellOverlandFlow(intake_ix, intake_iy, problem_data, grid);
        if (ReservoirDataPhysicalHasSecondaryIntakeCell(reservoir_data_physical))
        {
          StopOutletFlowAtCellOverlandFlow(secondary_intake_ix, secondary_intake_iy, problem_data, grid);
        }
      }
      else if (ReservoirDataOverlandFlowSolver(reservoir_data) == OVERLAND_KINEMATIC)
      {
        StopOutletFlowAtCellOverlandKinematic(intake_ix, intake_iy, problem_data, grid);
        if (ReservoirDataPhysicalHasSecondaryIntakeCell(reservoir_data_physical))
        {
          StopOutletFlowAtCellOverlandKinematic(secondary_intake_ix, secondary_intake_iy, problem_data, grid);
        }
      }

      reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
      ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy0->name)) + 1);
      strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy0->name));
      ReservoirDataPhysicalIntakeCellMpiRank(reservoir_data_physical) = (intake_cell_rank);
      ReservoirDataPhysicalIntakeXLower(reservoir_data_physical) = (dummy0->intake_x_location);
      ReservoirDataPhysicalIntakeYLower(reservoir_data_physical) = (dummy0->intake_y_location);
      ReservoirDataPhysicalSecondaryIntakeCellMpiRank(reservoir_data_physical) = (secondary_intake_cell_rank);
      ReservoirDataPhysicalSecondaryIntakeXLower(reservoir_data_physical) = (dummy0->secondary_intake_x_location);
      ReservoirDataPhysicalSecondaryIntakeYLower(reservoir_data_physical) = (dummy0->secondary_intake_y_location);
      ReservoirDataPhysicalHasSecondaryIntakeCell(reservoir_data_physical) = (dummy0->has_secondary_intake_cell);
      ReservoirDataPhysicalReleaseCellMpiRank(reservoir_data_physical) = (release_cell_rank);
      ReservoirDataPhysicalReleaseXLower(reservoir_data_physical) = (dummy0->release_x_location);
      ReservoirDataPhysicalReleaseYLower(reservoir_data_physical) = (dummy0->release_y_location);
      ReservoirDataPhysicalIntakeXUpper(reservoir_data_physical) = (dummy0->intake_x_location);
      ReservoirDataPhysicalIntakeYUpper(reservoir_data_physical) = (dummy0->intake_y_location);
      ReservoirDataPhysicalSecondaryIntakeXUpper(reservoir_data_physical) = (dummy0->secondary_intake_x_location);
      ReservoirDataPhysicalSecondaryIntakeYUpper(reservoir_data_physical) = (dummy0->secondary_intake_y_location);
      ReservoirDataPhysicalReleaseXUpper(reservoir_data_physical) = (dummy0->release_x_location);
      ReservoirDataPhysicalReleaseYUpper(reservoir_data_physical) = (dummy0->release_y_location);
      ReservoirDataPhysicalMaxStorage(reservoir_data_physical) = (dummy0->max_storage);
      ReservoirDataPhysicalMinReleaseStorage(reservoir_data_physical) = (dummy0->min_release_storage);
      ReservoirDataPhysicalIntakeAmountSinceLastPrint(reservoir_data_physical) = (0);
      ReservoirDataPhysicalReleaseAmountSinceLastPrint(reservoir_data_physical) = (0);
      ReservoirDataPhysicalReleaseAmountInSolver(reservoir_data_physical) = (0);
      ReservoirDataPhysicalReleaseRate(reservoir_data_physical) = (dummy0->release_rate);
      ReservoirDataPhysicalStorage(reservoir_data_physical) = (dummy0->storage);
      ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) = new_intake_subgrid;
      ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) = new_secondary_intake_subgrid;
      ReservoirDataPhysicalReleaseSubgrid(reservoir_data_physical) = new_release_subgrid;
      ReservoirDataPhysicalSize(reservoir_data_physical) = release_subgrid_volume;
      ReservoirDataReservoirPhysical(reservoir_data, reservoir_index) = reservoir_data_physical;
#ifdef PARFLOW_HAVE_MPI
      reservoir_data_physical->mpi_communicator = new_reservoir_communicator;
#endif
      /* Put in values for this reservoir */
      reservoir_index++;
    }
    ReservoirDataNumReservoirs(reservoir_data) = (reservoir_index);
  }
}


/*--------------------------------------------------------------------------
 * ReservoirPackageInitInstanceXtra
 *--------------------------------------------------------------------------*/
PFModule *ReservoirPackageInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * ReservoirPackageFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  ReservoirPackageFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * ReservoirPackageNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *ReservoirPackageNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;


  char *reservoir_names;
  char * EMPTY_NAMES_LIST = "";
  char *reservoir_name;

  char key[IDB_MAX_KEY_LEN];

  int num_reservoirs;

  char          *switch_name;
  int switch_value;
  NameArray overland_flow_solver_na;

  public_xtra = ctalloc(PublicXtra, 1);

  reservoir_names = GetStringDefault("Reservoirs.Names", EMPTY_NAMES_LIST);

  public_xtra->reservoir_names = NA_NewNameArray(reservoir_names);

  num_reservoirs = NA_Sizeof(public_xtra->reservoir_names);


  if (num_reservoirs > 0)
  {
    (public_xtra->type) = ctalloc(int, num_reservoirs);
    (public_xtra->data) = ctalloc(void *, num_reservoirs);
    overland_flow_solver_na = NA_NewNameArray("OverlandFlow OverlandKinematic");
    sprintf(key, "Reservoirs.Overland_Flow_Solver");
    switch_name = GetString(key);
    switch_value = NA_NameToIndexExitOnError(overland_flow_solver_na, switch_name, key);
    switch (switch_value)
    {
      case 0:
      {
        public_xtra->overland_flow_solver = OVERLAND_FLOW;
        break;
      }

      case 1:
      {
        public_xtra->overland_flow_solver = OVERLAND_KINEMATIC;
        break;
      }

      default:
      {
        InputError("Reservoirs.Overland_Flow_Solver must be one of OverlandFlow or OverlandKinematic, not %s%s\n", switch_name, "");
      }
    }

    for (int i = 0; i < num_reservoirs; i++)
    {
      reservoir_name = NA_IndexToName(public_xtra->reservoir_names, i);

      dummy0 = ctalloc(Type0, 1);
      dummy0->name = strdup(reservoir_name);

      sprintf(key, "Reservoirs.%s.Release_X", reservoir_name);
      dummy0->release_x_location = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Release_Y", reservoir_name);
      dummy0->release_y_location = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Intake_X", reservoir_name);
      dummy0->intake_x_location = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Intake_Y", reservoir_name);
      dummy0->intake_y_location = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Has_Secondary_Intake_Cell", reservoir_name);
      dummy0->has_secondary_intake_cell = GetInt(key);
      // I would use a name array to do this validation, but the input is an int. Long term there
      // needs to be a fix for bool values that are nested under a dynamic key (e.g. the reservoir name)
      // within the table reader code
      if (!(dummy0->has_secondary_intake_cell == 0 || dummy0->has_secondary_intake_cell == 1))
      {
        InputError("Reservoirs.%s.HasSecondaryIntakeCell must be one of 0 (if false) or 1 (if true)", reservoir_name, "");
      }

      if (dummy0->has_secondary_intake_cell)
      {
        sprintf(key, "Reservoirs.%s.Secondary_Intake_X", reservoir_name);
        dummy0->secondary_intake_x_location = GetDouble(key);

        sprintf(key, "Reservoirs.%s.Secondary_Intake_Y", reservoir_name);
        dummy0->secondary_intake_y_location = GetDouble(key);
      }
      else
      {
        dummy0->secondary_intake_x_location = -1;
        dummy0->secondary_intake_y_location = -1;
      };
      sprintf(key, "Reservoirs.%s.Min_Release_Storage", reservoir_name);
      dummy0->min_release_storage = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Release_Rate", reservoir_name);
      dummy0->release_rate = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Max_Storage", reservoir_name);
      dummy0->max_storage = GetDouble(key);

      sprintf(key, "Reservoirs.%s.Storage", reservoir_name);
      dummy0->storage = GetDouble(key);

      (public_xtra->num_reservoirs)++;
      (public_xtra->data[i]) = (void*)dummy0;
    }
  }


  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*-------------------------------------------------------------------------
 * ReservoirPackageFreePublicXtra
 *-------------------------------------------------------------------------*/

void  ReservoirPackageFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0         *dummy0;

  int num_reservoirs;
  int i;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->reservoir_names);

    /* Free the reservoir information */
    num_reservoirs = (public_xtra->num_reservoirs);
    if (num_reservoirs > 0)
    {
      for (i = 0; i < num_reservoirs; i++)
      {
        dummy0 = (Type0*)(public_xtra->data[i]);

        if ((dummy0->name))
        {
          tfree((dummy0->name));
        }
      }
      tfree(public_xtra->data);
    }
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * ReservoirPackageSizeOfTempData
 *--------------------------------------------------------------------------*/

int  ReservoirPackageSizeOfTempData()
{
  return 0;
}
