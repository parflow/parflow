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
//#include "time_series.h"
//#include "time_series.c"
#include <stdio.h>
#include <string.h>

#define PRESSURE_RESERVOIR   0
#define FLUX_RESERVOIR       1

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {
    int num_phases;
    int num_contaminants;

    /* reservoir info */
    int num_units;
    int num_reservoirs;

    int       *type;
    void     **data;

    /* Timing Cycle information */
    int num_cycles;

    int       *interval_divisions;
    int      **intervals;
    int       *repeat_counts;

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

bool subgrid_lives_on_this_rank(Subgrid* subgrid, Grid *grid){
  int subgrid_index;
  Subgrid* rank_subgrid, *tmp_subgrid;
  ForSubgridI(subgrid_index, GridSubgrids(grid)){
    rank_subgrid = SubgridArraySubgrid(GridSubgrids(grid), subgrid_index);
    if((tmp_subgrid = IntersectSubgrids(rank_subgrid, subgrid))){
      return true;
    }
  }
  return false;
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

  TimeCycleData    *time_cycle_data;

  ReservoirData         *reservoir_data = ProblemDataReservoirData(problem_data);
  ReservoirDataPhysical *reservoir_data_physical;

  int i, sequence_number, phase, contaminant, indx, press_reservoir, reservoir_index;

  int intake_ix, intake_iy;
  int secondary_intake_ix, secondary_intake_iy;
  int release_ix, release_iy;
  int iz_lower, iz_upper;
  int nx, ny, nz;
  double dx, dy, dz;
  int rx, ry, rz;
  int current_mpi_rank;
  int intake_cell_rank, secondary_intake_cell_rank, release_cell_rank;
  int split_color;
  int grid_nz;

  double          **phase_values;
  double intake_subgrid_volume;
  double secondary_intake_subgrid_volume;
  double release_subgrid_volume;
  double intake_x_lower, intake_x_upper, intake_y_lower, intake_y_upper, z_lower, z_upper;
  double secondary_intake_x_lower, secondary_intake_x_upper, secondary_intake_y_lower, secondary_intake_y_upper;
  double release_x_lower, release_x_upper, release_y_lower, release_y_upper;
  double max_storage, min_release_storage, Storage, release_rate;
  double intake_amount_since_last_print, release_amount_since_last_print;

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
  sequence_number = 0;

  if ((public_xtra->num_units) > 0)
  {
    /* Load the reservoir data */
    for (i = 0; i < (public_xtra->num_units); i++)
    {

      dummy0 = (Type0*)(public_xtra->data[i]);

      intake_ix = IndexSpaceX((dummy0->intake_x_location), 0);
      intake_iy = IndexSpaceY((dummy0->intake_y_location), 0);
      secondary_intake_ix = IndexSpaceX((dummy0->secondary_intake_x_location), 0);
      secondary_intake_iy = IndexSpaceY((dummy0->secondary_intake_y_location), 0);
      release_ix = IndexSpaceX((dummy0->release_x_location), 0);
      release_iy = IndexSpaceY((dummy0->release_y_location), 0);
      Vector * index_of_domain_top = ProblemDataIndexOfDomainTop(problem_data);

      intake_cell_rank = -1;
      secondary_intake_cell_rank = -1;
      release_cell_rank= -1;



      iz_lower = grid_nz - 1;
      iz_upper = grid_nz - 1;

      nx = 1;
      ny = 1;
      nz = 1;

      rx = 0;
      ry = 0;
      rz = 0;

      current_mpi_rank = amps_Rank(amps_CommWorld);

      new_intake_subgrid = NewSubgrid(intake_ix, intake_iy, iz_lower,
                                      nx, ny, nz,
                                      rx, ry, rz,
                                      current_mpi_rank);

      dx = SubgridDX(new_intake_subgrid);
      dy = SubgridDY(new_intake_subgrid);
      dz = SubgridDZ(new_intake_subgrid);

      intake_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

      new_secondary_intake_subgrid = NewSubgrid(secondary_intake_ix, secondary_intake_iy, iz_lower,
                                                nx, ny, nz,
                                                rx, ry, rz,
                                                current_mpi_rank);

      dx = SubgridDX(new_secondary_intake_subgrid);
      dy = SubgridDY(new_secondary_intake_subgrid);
      dz = SubgridDZ(new_secondary_intake_subgrid);


      secondary_intake_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

      new_release_subgrid = NewSubgrid(release_ix, release_iy, iz_lower,
                                       nx, ny, nz,
                                       rx, ry, rz,
                                       current_mpi_rank);

      dx = SubgridDX(new_release_subgrid);
      dy = SubgridDY(new_release_subgrid);
      dz = SubgridDZ(new_release_subgrid);

      if (subgrid_lives_on_this_rank(new_intake_subgrid, grid)) {
        intake_cell_rank = current_mpi_rank;
      }
      amps_Invoice intake_cell_invoice = amps_NewInvoice("%i", &intake_cell_rank);
      amps_AllReduce(amps_CommWorld, intake_cell_invoice, amps_Max);
      amps_FreeInvoice(intake_cell_invoice);
      if (subgrid_lives_on_this_rank(new_secondary_intake_subgrid, grid)) {
        secondary_intake_cell_rank = current_mpi_rank;
      }
      amps_Invoice secondary_intake_cell_invoice = amps_NewInvoice("%i", &secondary_intake_cell_rank);
      amps_AllReduce(amps_CommWorld, secondary_intake_cell_invoice, amps_Max);
      amps_FreeInvoice(secondary_intake_cell_invoice);
      if (subgrid_lives_on_this_rank(new_release_subgrid, grid)) {
        release_cell_rank = current_mpi_rank;
      }
      amps_Invoice release_cell_invoice = amps_NewInvoice("%i", &release_cell_rank);
      amps_AllReduce(amps_CommWorld, release_cell_invoice, amps_Max);
      amps_FreeInvoice(release_cell_invoice);

      part_of_reservoir_lives_on_this_rank =
              (release_cell_rank==current_mpi_rank) ||
              (intake_cell_rank==current_mpi_rank) ||
              (secondary_intake_cell_rank==current_mpi_rank);

      MPI_Comm new_reservoir_communicator;
      split_color = part_of_reservoir_lives_on_this_rank ? 1 : MPI_UNDEFINED;
      MPI_Comm_split(MPI_COMM_WORLD, split_color, current_mpi_rank, &new_reservoir_communicator);

      release_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);
      reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
      ReservoirDataPhysicalNumber(reservoir_data_physical) = sequence_number;
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
      ReservoirDataPhysicalDiameter(reservoir_data_physical) = pfmin(dx, dy);
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
      reservoir_data_physical->mpi_communicator = new_reservoir_communicator;
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

#if 0
  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
#endif
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

PFModule  *ReservoirPackageNewPublicXtra(
    int num_phases,
    int num_contaminants)
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  Type0         *dummy0;

  int num_cycles;


  char *reservoir_names;
  char * EMPTY_NAMES_LIST = "";
  char *reservoir_name;

  char key[IDB_MAX_KEY_LEN];

  int num_units;

  char *switch_name;
  int switch_value;
  NameArray switch_na;

  switch_na = NA_NewNameArray("False True");


  public_xtra = ctalloc(PublicXtra, 1);




  reservoir_names = GetStringDefault("Reservoirs.Names", EMPTY_NAMES_LIST);

  public_xtra->reservoir_names = NA_NewNameArray(reservoir_names);

  num_units = NA_Sizeof(public_xtra->reservoir_names);


  num_cycles = public_xtra->num_cycles = num_units;

  public_xtra->interval_divisions = ctalloc(int, num_cycles);
  public_xtra->intervals = ctalloc(int *, num_cycles);
  public_xtra->repeat_counts = ctalloc(int, num_cycles);

  public_xtra->num_units = num_units;



  if (num_units > 0) {
      (public_xtra->type) = ctalloc(int, num_units);
      (public_xtra->data) = ctalloc(void *, num_units);
      int i;
      for (i = 0; i < num_units; i++) {
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
          switch_name = GetStringDefault(key, "False");
          switch_value = NA_NameToIndexExitOnError(switch_na, switch_name, key);
          dummy0->has_secondary_intake_cell = switch_value;

          if (dummy0->has_secondary_intake_cell){
              sprintf(key, "Reservoirs.%s.Secondary_Intake_X", reservoir_name);
              dummy0->secondary_intake_x_location = GetDouble(key);

              sprintf(key, "Reservoirs.%s.Secondary_Intake_Y", reservoir_name);
              dummy0->secondary_intake_y_location = GetDouble(key);

          }
          else{
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

  int num_units, num_cycles;
  int i;

  if (public_xtra)
  {
    NA_FreeNameArray(public_xtra->reservoir_names);

    /* Free the reservoir information */
    num_units = (public_xtra->num_units);
    if (num_units > 0)
    {
      for (i = 0; i < num_units; i++)
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