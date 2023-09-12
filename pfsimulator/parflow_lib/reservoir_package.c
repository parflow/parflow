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
    int num_press_reservoirs;
    int num_flux_reservoirs;

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
    char *release_curve_file;
//  TimeSeries release_curve;
    double intake_x_location;
    double intake_y_location;
    int has_secondary_intake_cell;
    double secondary_intake_x_location;
    double secondary_intake_y_location;
    double release_x_location;
    double release_y_location;
    double z_lower, z_upper;
    double max_capacity, min_release_capacity, current_storage, release_rate;
} Type0;                      /* basic vertical reservoir */

/*--------------------------------------------------------------------------
 * ReservoirPackage
 *--------------------------------------------------------------------------*/

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

  int i, sequence_number, phase, contaminant, indx, press_reservoir, flux_reservoir;

  int intake_ix, intake_iy;
  int secondary_intake_ix, secondary_intake_iy;
  int release_ix, release_iy;
  int iz_lower, iz_upper;
  int nx, ny, nz;
  double dx, dy, dz;
  int rx, ry, rz;
  int process;
  int reservoir_action, action, mechanism, method;

  double          **phase_values;
  double intake_subgrid_volume;
  double secondary_intake_subgrid_volume;
  double release_subgrid_volume;
  double intake_x_lower, intake_x_upper, intake_y_lower, intake_y_upper, z_lower, z_upper;
  double secondary_intake_x_lower, secondary_intake_x_upper, secondary_intake_y_lower, secondary_intake_y_upper;
  double release_x_lower, release_x_upper, release_y_lower, release_y_upper;
  double max_capacity, min_release_capacity, Current_Storage, release_rate;
  double intake_amount_since_last_print, release_amount_since_last_print;
  /* Allocate the reservoir data */

  ReservoirDataNumReservoirs(reservoir_data) = (public_xtra->num_reservoirs);

  if ((public_xtra->num_reservoirs) > 0)
  {
    ReservoirDataNumPressReservoirs(reservoir_data) = (public_xtra->num_press_reservoirs);
    if ((public_xtra->num_press_reservoirs) > 0)
    {
      ReservoirDataPressReservoirPhysicals(reservoir_data) = ctalloc(ReservoirDataPhysical *, (public_xtra->num_press_reservoirs));
    }

    ReservoirDataNumFluxReservoirs(reservoir_data) = (public_xtra->num_flux_reservoirs);
    if ((public_xtra->num_flux_reservoirs) > 0)
    {
      ReservoirDataFluxReservoirPhysicals(reservoir_data) = ctalloc(ReservoirDataPhysical *, (public_xtra->num_flux_reservoirs));
    }
  }

  press_reservoir = 0;
  flux_reservoir = 0;
  sequence_number = 0;

  if ((public_xtra->num_units) > 0)
  {
    /* Load the reservoir data */
    for (i = 0; i < (public_xtra->num_units); i++)
    {
      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = (Type0*)(public_xtra->data[i]);

          intake_ix = IndexSpaceX((dummy0->intake_x_location), 0);
          intake_iy = IndexSpaceY((dummy0->intake_y_location), 0);
          secondary_intake_ix = IndexSpaceX((dummy0->secondary_intake_x_location), 0);
          secondary_intake_iy = IndexSpaceY((dummy0->secondary_intake_y_location), 0);
          release_ix = IndexSpaceX((dummy0->release_x_location), 0);
          release_iy = IndexSpaceY((dummy0->release_y_location), 0);
          iz_lower = IndexSpaceZ((dummy0->z_lower), 0);
          iz_upper = IndexSpaceZ((dummy0->z_upper), 0);

          nx = 1;
          ny = 1;
          nz = iz_upper - iz_lower + 1;

          rx = 0;
          ry = 0;
          rz = 0;

          process = amps_Rank(amps_CommWorld);

          new_intake_subgrid = NewSubgrid(intake_ix, intake_iy, iz_lower,
                                          nx, ny, nz,
                                          rx, ry, rz,
                                          process);


          dx = SubgridDX(new_intake_subgrid);
          dy = SubgridDY(new_intake_subgrid);
          dz = SubgridDZ(new_intake_subgrid);

          intake_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

          new_secondary_intake_subgrid = NewSubgrid(secondary_intake_ix, secondary_intake_iy, iz_lower,
                                                    nx, ny, nz,
                                                    rx, ry, rz,
                                                    process);

          dx = SubgridDX(new_secondary_intake_subgrid);
          dy = SubgridDY(new_secondary_intake_subgrid);
          dz = SubgridDZ(new_secondary_intake_subgrid);

          secondary_intake_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);

          new_release_subgrid = NewSubgrid(release_ix, release_iy, iz_lower,
                                          nx, ny, nz,
                                          rx, ry, rz,
                                          process);

          dx = SubgridDX(new_release_subgrid);
          dy = SubgridDY(new_release_subgrid);
          dz = SubgridDZ(new_release_subgrid);

          release_subgrid_volume = (nx * dx) * (ny * dy) * (nz * dz);
            reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
            ReservoirDataPhysicalNumber(reservoir_data_physical) = sequence_number;
            ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy0->name)) + 1);
            strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy0->name));
    //            ReservoirDataPhysicalReleaseCurveFile(reservoir_data_physical) = ctalloc(char, strlen((dummy0->release_curve_file)) + 1);
    //            strcpy(ReservoirDataPhysicalReleaseCurveFile(reservoir_data_physical), (dummy0->release_curve_file));
            // TODO dont hardcode column names below
    //            TimeSeries* tmp_time_series = NewTimeSeries(ReservoirDataPhysicalReleaseCurveFile(reservoir_data_physical), "times", "values");
    //            reservoir_data_physical->release_curve = ctalloc(TimeSeries,1);
    //            reservoir_data_physical->release_curve = tmp_time_series;
    //            printf("Time series first value: %f\n", GetValue(reservoir_data_physical->release_curve, 0));
    //            printf("Time series second value: %f\n", GetValue(reservoir_data_physical->release_curve, 4.0));
            ReservoirDataPhysicalIntakeXLower(reservoir_data_physical) = (dummy0->intake_x_location);
            ReservoirDataPhysicalIntakeYLower(reservoir_data_physical) = (dummy0->intake_y_location);
            ReservoirDataPhysicalSecondaryIntakeXLower(reservoir_data_physical) = (dummy0->secondary_intake_x_location);
            ReservoirDataPhysicalSecondaryIntakeYLower(reservoir_data_physical) = (dummy0->secondary_intake_y_location);
            ReservoirDataPhysicalHasSecondaryIntakeCell(reservoir_data_physical) = (dummy0->has_secondary_intake_cell);
            ReservoirDataPhysicalReleaseXLower(reservoir_data_physical) = (dummy0->release_x_location);
            ReservoirDataPhysicalReleaseYLower(reservoir_data_physical) = (dummy0->release_y_location);
            ReservoirDataPhysicalZLower(reservoir_data_physical) = (dummy0->z_lower);
            ReservoirDataPhysicalIntakeXUpper(reservoir_data_physical) = (dummy0->intake_x_location);
            ReservoirDataPhysicalIntakeYUpper(reservoir_data_physical) = (dummy0->intake_y_location);
            ReservoirDataPhysicalSecondaryIntakeXUpper(reservoir_data_physical) = (dummy0->secondary_intake_x_location);
            ReservoirDataPhysicalSecondaryIntakeYUpper(reservoir_data_physical) = (dummy0->secondary_intake_y_location);
            ReservoirDataPhysicalReleaseXUpper(reservoir_data_physical) = (dummy0->release_x_location);
            ReservoirDataPhysicalReleaseYUpper(reservoir_data_physical) = (dummy0->release_y_location);
            ReservoirDataPhysicalZUpper(reservoir_data_physical) = (dummy0->z_upper);
            ReservoirDataPhysicalDiameter(reservoir_data_physical) = pfmin(dx, dy);
            ReservoirDataPhysicalMaxCapacity(reservoir_data_physical) = (dummy0->max_capacity);
            ReservoirDataPhysicalMinReleaseCapacity(reservoir_data_physical) = (dummy0->min_release_capacity);
            ReservoirDataPhysicalIntakeAmountSinceLastPrint(reservoir_data_physical) = (0);
            ReservoirDataPhysicalReleaseAmountSinceLastPrint(reservoir_data_physical) = (0);
            ReservoirDataPhysicalReleaseAmountInSolver(reservoir_data_physical) = (0);
            ReservoirDataPhysicalReleaseRate(reservoir_data_physical) = (dummy0->release_rate);
            ReservoirDataPhysicalCurrentCapacity(reservoir_data_physical) = (dummy0->current_storage);
            ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) = new_intake_subgrid;
            ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) = new_secondary_intake_subgrid;
            ReservoirDataPhysicalReleaseSubgrid(reservoir_data_physical) = new_release_subgrid;
            ReservoirDataPhysicalSize(reservoir_data_physical) = release_subgrid_volume;
            ReservoirDataFluxReservoirPhysical(reservoir_data, flux_reservoir) = reservoir_data_physical;

            /* Put in values for this reservoir */
            flux_reservoir++;

          sequence_number++;
          break;
        }
      }
    }
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
  int global_cycle;


  char *reservoir_names;
  char *reservoir_name;
  char *release_curve_file;

  char *cycle_name;

  char key[IDB_MAX_KEY_LEN];

  char *switch_name;

  int phase;
  int contaminant;
  int num_units;


  public_xtra = ctalloc(PublicXtra, 1);


  (public_xtra->num_phases) = num_phases;
  (public_xtra->num_contaminants) = num_contaminants;

  reservoir_names = GetString("Reservoirs.Names");

  public_xtra->reservoir_names = NA_NewNameArray(reservoir_names);

  num_units = NA_Sizeof(public_xtra->reservoir_names);


  num_cycles = public_xtra->num_cycles = num_units;

  public_xtra->interval_divisions = ctalloc(int, num_cycles);
  public_xtra->intervals = ctalloc(int *, num_cycles);
  public_xtra->repeat_counts = ctalloc(int, num_cycles);

  public_xtra->num_units = num_units;

  public_xtra->num_press_reservoirs = 0;
  public_xtra->num_flux_reservoirs = 0;

  if (num_units > 0) {
      (public_xtra->type) = ctalloc(int, num_units);
      (public_xtra->data) = ctalloc(void *, num_units);
      int i;
      for (i = 0; i < num_units; i++) {
          reservoir_name = NA_IndexToName(public_xtra->reservoir_names, i);

          switch ((public_xtra->type[i])) {
              case 0: {
                  dummy0 = ctalloc(Type0, 1);

                  sprintf(key, "Reservoirs.%s.Release_X", reservoir_name);
                  dummy0->release_x_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Release_Y", reservoir_name);
                  dummy0->release_y_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Intake_X", reservoir_name);
                  dummy0->intake_x_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Intake_Y", reservoir_name);
                  dummy0->intake_y_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Secondary_Intake_X", reservoir_name);
                  dummy0->secondary_intake_x_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Secondary_Intake_Y", reservoir_name);
                  dummy0->secondary_intake_y_location = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Has_Secondary_Intake_Cell", reservoir_name);
                  dummy0->has_secondary_intake_cell = GetInt(key);

                  sprintf(key, "Reservoirs.%s.ZUpper", reservoir_name);
                  dummy0->z_upper = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Min_Release_Capacity", reservoir_name);
                  dummy0->min_release_capacity = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Release_Rate", reservoir_name);
                  dummy0->release_rate = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Max_Capacity", reservoir_name);
                  dummy0->max_capacity = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.Current_Storage", reservoir_name);
                  dummy0->current_storage = GetDouble(key);

                  sprintf(key, "Reservoirs.%s.ZLower", reservoir_name);
                  dummy0->z_lower = 1.0;

                  default: {
                      InputError("Error: invalid type <%s> for key <%s>\n",
                                 switch_name, key);
                  }
              }
          }
          (public_xtra->num_flux_reservoirs)++;
      }
      (public_xtra->num_reservoirs) =  (public_xtra->num_flux_reservoirs);
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
  int i, interval_number, interval_division;

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
        //TODO figure out why this seg faults
    //            if ((dummy0->release_curve_file))
    //            {
    //              tfree((dummy0->release_curve_file));
    //            }
        tfree(dummy0);


      }

      tfree(public_xtra->data);
      tfree(public_xtra->type);
    }

    /* Free the time cycling information */
    num_cycles = (public_xtra->num_cycles);

    tfree((public_xtra->repeat_counts));

    for (i = 0; i < num_cycles; i++)
    {
      tfree((public_xtra->intervals[i]));
    }
    tfree((public_xtra->intervals));

    tfree((public_xtra->interval_divisions));


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