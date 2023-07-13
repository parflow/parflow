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
    int action;
    int mechanism;
    double intake_x_location;
    double intake_y_location;
    int has_secondary_intake_cell;
    double secondary_intake_x_location;
    double secondary_intake_y_location;
    double release_x_location;
    double release_y_location;
    double z_lower, z_upper;
    double max_capacity, min_release_capacity, current_storage, release_rate;
    int status;
    int method;
} Type0;                      /* basic vertical reservoir */

typedef struct {
    char    *name;
    int mechanism_ext;
    int mechanism_inj;
    double xlocation;
    double ylocation;
    double z_lower_ext, z_upper_ext;
    double z_lower_inj, z_upper_inj;
    int method_ext;
    int method_inj;
} Type1;                      /* basic vertical reservoir, recirculating */

/*--------------------------------------------------------------------------
 * ReservoirPackage
 *--------------------------------------------------------------------------*/

void         ReservoirPackage(
    ProblemData *problem_data)
{
  PFModule         *this_module = ThisPFModule;
  PublicXtra       *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Type0            *dummy0;
  Type1            *dummy1;

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
  double max_capacity, min_release_capacity, current_storage, release_rate;
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

          if ((dummy0->mechanism) == PRESSURE_RESERVOIR)
          {
            /* Put in physical data for this reservoir */
            reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
            ReservoirDataPhysicalNumber(reservoir_data_physical) = sequence_number;
            ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy0->name)) + 1);
            strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy0->name));
//            ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy0->release_curve_file)) + 1);
//            strcpy(ReservoirDataPhysicalReleaseCurveFile(reservoir_data_physical), (dummy0->release_curve_file));
            // TODO dont hardcode column names below
//            TimeSeries* tmp_time_series = NewTimeSeries(ReservoirDataPhysicalReleaseCurveFile(reservoir_data_physical), "times", "values");
//            reservoir_data_physical->release_curve = ctalloc(TimeSeries,1);
//            reservoir_data_physical->release_curve = tmp_time_series;
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
            ReservoirDataPhysicalReleaseRate(reservoir_data_physical) = (dummy0->release_rate);
            ReservoirDataPhysicalCurrentCapacity(reservoir_data_physical) = (dummy0->current_storage);
            ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) = new_intake_subgrid;
            ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) = new_secondary_intake_subgrid;
            ReservoirDataPhysicalReleaseSubgrid(reservoir_data_physical) = new_release_subgrid;
            ReservoirDataPhysicalSize(reservoir_data_physical) = intake_subgrid_volume;
            ReservoirDataPhysicalAction(reservoir_data_physical) = (dummy0->action);
            ReservoirDataPressReservoirPhysical(reservoir_data, press_reservoir) = reservoir_data_physical;

            /* Put in values for this reservoir */
            press_reservoir++;
          }
          else if ((dummy0->mechanism) == FLUX_RESERVOIR)
          {
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
            ReservoirDataPhysicalAction(reservoir_data_physical) = (dummy0->action);
            ReservoirDataFluxReservoirPhysical(reservoir_data, flux_reservoir) = reservoir_data_physical;

            /* Put in values for this reservoir */
            flux_reservoir++;
          }
          sequence_number++;
          break;
        }

        case 1:
        {
          dummy1 = (Type1*)(public_xtra->data[i]);

          intake_x_lower = (dummy1->xlocation);
          intake_y_lower = (dummy1->ylocation);
          intake_x_upper = (dummy1->xlocation);
          intake_y_upper = (dummy1->ylocation);

          /* reservoir_action = 0 means we're doing extraction, reservoir_action = 1 means we're doing injection    */
          /* The ordering of the extraction and injection reservoirs is important.  The partner_ptr of the     */
          /*   injection reservoir needs to point to allocated data (in the extraction reservoir).  If the order is */
          /*   reversed then this storage wont exist.                                                     */

          for (reservoir_action = 0; reservoir_action < 2; reservoir_action++)
          {
            intake_ix = IndexSpaceX((dummy1->xlocation), 0);
            intake_iy = IndexSpaceY((dummy1->ylocation), 0);
            if (reservoir_action == 0)
            {
              z_lower = (dummy1->z_lower_ext);
              z_upper = (dummy1->z_upper_ext);

              action = EXTRACTION_RESERVOIR;
              mechanism = (dummy1->mechanism_ext);
              method = (dummy1->method_ext);
            }
            else
            {
              z_lower = (dummy1->z_lower_inj);
              z_upper = (dummy1->z_upper_inj);

              action = INJECTION_RESERVOIR;
              mechanism = (dummy1->mechanism_inj);
              method = (dummy1->method_inj);
            }

            iz_lower = IndexSpaceZ(z_lower, 0);
            iz_upper = IndexSpaceZ(z_upper, 0);

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

            if (mechanism == PRESSURE_RESERVOIR)
            {
              /* Put in physical data for this reservoir */
              reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
              ReservoirDataPhysicalNumber(reservoir_data_physical) = sequence_number;
              if (action == EXTRACTION_RESERVOIR)
              {
                ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy1->name)) + 14);
                strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy1->name));
                strcat(ReservoirDataPhysicalName(reservoir_data_physical), " (extraction)");
              }
              else
              {
                ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy1->name)) + 13);
                strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy1->name));
                strcat(ReservoirDataPhysicalName(reservoir_data_physical), " (injection)");
              }

              ReservoirDataPhysicalIntakeXLower(reservoir_data_physical) = intake_x_lower;
              ReservoirDataPhysicalIntakeYLower(reservoir_data_physical) = intake_y_lower;
              ReservoirDataPhysicalSecondaryIntakeXLower(reservoir_data_physical) = secondary_intake_x_lower;
              ReservoirDataPhysicalSecondaryIntakeYLower(reservoir_data_physical) = secondary_intake_y_lower;
              ReservoirDataPhysicalZLower(reservoir_data_physical) = z_lower;
              ReservoirDataPhysicalIntakeXUpper(reservoir_data_physical) = intake_x_upper;
              ReservoirDataPhysicalIntakeYUpper(reservoir_data_physical) = intake_y_upper;
              ReservoirDataPhysicalSecondaryIntakeXUpper(reservoir_data_physical) = secondary_intake_x_upper;
              ReservoirDataPhysicalSecondaryIntakeYUpper(reservoir_data_physical) = secondary_intake_y_upper;
              ReservoirDataPhysicalZUpper(reservoir_data_physical) = z_upper;

              ReservoirDataPhysicalDiameter(reservoir_data_physical) = pfmin(dx, dx);
              ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) = new_intake_subgrid;
              ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) = new_secondary_intake_subgrid;
              ReservoirDataPhysicalSize(reservoir_data_physical) = intake_subgrid_volume;
              ReservoirDataPhysicalAction(reservoir_data_physical) = action;
              ReservoirDataPressReservoirPhysical(reservoir_data, press_reservoir) = reservoir_data_physical;

              /* Put in values for this reservoir */
              press_reservoir++;
            }
            else if (mechanism == FLUX_RESERVOIR)
            {
              reservoir_data_physical = ctalloc(ReservoirDataPhysical, 1);
              ReservoirDataPhysicalNumber(reservoir_data_physical) = sequence_number;
              if (action == EXTRACTION_RESERVOIR)
              {
                ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy1->name)) + 14);
                strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy1->name));
                strcat(ReservoirDataPhysicalName(reservoir_data_physical), " (extraction)");
              }
              else
              {
                ReservoirDataPhysicalName(reservoir_data_physical) = ctalloc(char, strlen((dummy1->name)) + 13);
                strcpy(ReservoirDataPhysicalName(reservoir_data_physical), (dummy1->name));
                strcat(ReservoirDataPhysicalName(reservoir_data_physical), " (injection)");
              }

              ReservoirDataPhysicalIntakeXLower(reservoir_data_physical) = intake_x_lower;
              ReservoirDataPhysicalIntakeYLower(reservoir_data_physical) = intake_y_lower;
              ReservoirDataPhysicalSecondaryIntakeXLower(reservoir_data_physical) = secondary_intake_x_lower;
              ReservoirDataPhysicalSecondaryIntakeYLower(reservoir_data_physical) = secondary_intake_y_lower;
              ReservoirDataPhysicalZLower(reservoir_data_physical) = z_lower;
              ReservoirDataPhysicalIntakeXUpper(reservoir_data_physical) = intake_x_upper;
              ReservoirDataPhysicalIntakeYUpper(reservoir_data_physical) = intake_y_upper;
              ReservoirDataPhysicalSecondaryIntakeXUpper(reservoir_data_physical) = secondary_intake_x_upper;
              ReservoirDataPhysicalSecondaryIntakeYUpper(reservoir_data_physical) = secondary_intake_y_upper;
              ReservoirDataPhysicalZUpper(reservoir_data_physical) = z_upper;

              ReservoirDataPhysicalDiameter(reservoir_data_physical) = pfmin(dx, dx);
              ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) = new_intake_subgrid;
              ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) = new_secondary_intake_subgrid;
              ReservoirDataPhysicalSize(reservoir_data_physical) = intake_subgrid_volume;
              ReservoirDataPhysicalAction(reservoir_data_physical) = action;
              ReservoirDataFluxReservoirPhysical(reservoir_data, flux_reservoir) = reservoir_data_physical;

              /* Put in values for this reservoir */
              flux_reservoir++;
            }
            sequence_number++;
          }
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
  Type1         *dummy1;

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

  NameArray inputtype_na;
  NameArray action_na;
  NameArray mechanism_na;
  NameArray methodpress_na;
  NameArray methodflux_na;

  inputtype_na = NA_NewNameArray("Vertical Recirc");
  action_na = NA_NewNameArray("Injection Extraction");
  mechanism_na = NA_NewNameArray("Pressure Flux");
  methodpress_na = NA_NewNameArray("Standard");
  methodflux_na = NA_NewNameArray("Standard Weighted Patterned");

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

  if (num_units > 0)
  {
    (public_xtra->type) = ctalloc(int, num_units);
    (public_xtra->data) = ctalloc(void *, num_units);
    int i;
    for (i = 0; i < num_units; i++)
    {
      reservoir_name = NA_IndexToName(public_xtra->reservoir_names, i);

      sprintf(key, "Reservoirs.%s.InputType", reservoir_name);
      switch_name = GetString(key);
      public_xtra->type[i] = NA_NameToIndex(inputtype_na, switch_name);

      switch ((public_xtra->type[i]))
      {
        case 0:
        {
          dummy0 = ctalloc(Type0, 1);

          /*** Read in the physical data for the reservoir ***/
          dummy0->name = strdup(reservoir_name);
          sprintf(key, "Reservoirs.%s.Action", reservoir_name);
          switch_name = GetString(key);
          dummy0->action = NA_NameToIndex(action_na, switch_name);
          if (dummy0->action < 0)
          {
            InputError("Error: invalid action <%s> for key <%s>\n",
                       switch_name, key);
          }

          sprintf(key, "Reservoirs.%s.Type", reservoir_name);
          switch_name = GetString(key);
          dummy0->mechanism = NA_NameToIndex(mechanism_na, switch_name);
          if (dummy0->mechanism < 0)
          {
            InputError("Error: invalid type <%s> for key <%s>\n",
                       switch_name, key);
          }
//          sprintf(key, "Reservoirs.%s.ReleaseCurveFile", reservoir_name);
//          dummy0->release_curve_file = GetString(key);

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

          sprintf(key, "Reservoirs.%s.current_storage", reservoir_name);
          dummy0->current_storage = GetDouble(key);


          sprintf(key, "Reservoirs.%s.ZLower", reservoir_name);
          dummy0->z_lower = GetDouble(key);

          if ((dummy0->mechanism) == PRESSURE_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.Method", reservoir_name);
            switch_name = GetString(key);
            (dummy0->method) = NA_NameToIndex(methodpress_na, switch_name);
            if ((dummy0->method) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }
          else if ((dummy0->mechanism) == FLUX_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.Method", reservoir_name);
            switch_name = GetString(key);
            (dummy0->method) = NA_NameToIndex(methodflux_na, switch_name);
            if ((dummy0->method) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }


          /*** Bump the counter for the reservoir type ***/
          if ((dummy0->mechanism) == PRESSURE_RESERVOIR)
          {
            (public_xtra->num_press_reservoirs)++;
          }
          else if ((dummy0->mechanism) == FLUX_RESERVOIR)
          {
            (public_xtra->num_flux_reservoirs)++;
          }

          (public_xtra->data[i]) = (void*)dummy0;

          break;
        }

        case 1:
        {
          dummy1 = ctalloc(Type1, 1);

          /*** Read in the physical data for the reservoir ***/

          dummy1->name = strdup(reservoir_name);

          sprintf(key, "Reservoirs.%s.ExtractionType", reservoir_name);
          switch_name = GetString(key);
          dummy1->mechanism_ext =
              NA_NameToIndex(mechanism_na, switch_name);

          if (dummy1->mechanism_ext < 0)
          {
            InputError("Error: invalid extraction type <%s> for key <%s>\n",
                       switch_name, key);
          }

          sprintf(key, "Reservoirs.%s.InjectionType", reservoir_name);
          switch_name = GetString(key);
          dummy1->mechanism_inj =
              NA_NameToIndex(mechanism_na, switch_name);
          if (dummy1->mechanism_inj < 0)
          {
            InputError("Error: invalid injection type <%s> for key <%s>\n",
                       switch_name, key);
          }

          sprintf(key, "Reservoirs.%s.X", reservoir_name);
          dummy1->xlocation = GetDouble(key);

          sprintf(key, "Reservoirs.%s.Y", reservoir_name);
          dummy1->ylocation = GetDouble(key);

          sprintf(key, "Reservoirs.%s.ExtractionZUpper", reservoir_name);
          dummy1->z_upper_ext = GetDouble(key);


          sprintf(key, "Reservoirs.%s.ExtractionZLower", reservoir_name);
          dummy1->z_lower_ext = GetDouble(key);


          sprintf(key, "Reservoirs.%s.InjectionZUpper", reservoir_name);
          dummy1->z_upper_inj = GetDouble(key);


          sprintf(key, "Reservoirs.%s.InjectionZLower", reservoir_name);
          dummy1->z_lower_inj = GetDouble(key);

          if ((dummy1->mechanism_ext) == PRESSURE_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.ExtractionMethod", reservoir_name);
            switch_name = GetString(key);
            (dummy1->method_ext) = NA_NameToIndex(methodpress_na, switch_name);
            if ((dummy1->method_ext) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }
          else if ((dummy1->mechanism_ext) == FLUX_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.ExtractionMethod", reservoir_name);
            switch_name = GetString(key);
            (dummy1->method_ext) = NA_NameToIndex(methodflux_na, switch_name);
            if ((dummy1->method_ext) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }

          if ((dummy1->mechanism_inj) == PRESSURE_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.InjectionMethod", reservoir_name);
            switch_name = GetString(key);
            (dummy1->method_inj) = NA_NameToIndex(methodpress_na, switch_name);
            if ((dummy1->method_inj) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }
          else if ((dummy1->mechanism_inj) == FLUX_RESERVOIR)
          {
            sprintf(key, "Reservoirs.%s.InjectionMethod", reservoir_name);
            switch_name = GetString(key);
            (dummy1->method_inj) = NA_NameToIndex(methodflux_na, switch_name);
            if ((dummy1->method_inj) < 0)
            {
              InputError("Error: invalid action <%s> for key <%s>\n",
                         switch_name, key);
            }
          }

          sprintf(key, "Reservoirs.%s.Cycle", reservoir_name);
          cycle_name = GetString(key);

          global_cycle = NA_NameToIndex(GlobalsCycleNames, cycle_name);

          if (global_cycle < 0)
          {
            InputError("Error: invalid cycle name <%s> for key <%s>\n",
                       cycle_name, key);
          }


          public_xtra->repeat_counts[i] =
              GlobalsRepeatCounts[global_cycle];


          /*** Bump the counter for both reservoir types ***/
          if ((dummy1->mechanism_inj) == PRESSURE_RESERVOIR)
          {
            (public_xtra->num_press_reservoirs)++;
          }
          else if ((dummy1->mechanism_inj) == FLUX_RESERVOIR)
          {
            (public_xtra->num_flux_reservoirs)++;
          }
          if ((dummy1->mechanism_ext) == PRESSURE_RESERVOIR)
          {
            (public_xtra->num_press_reservoirs)++;
          }
          else if ((dummy1->mechanism_ext) == FLUX_RESERVOIR)
          {
            (public_xtra->num_flux_reservoirs)++;
          }

          (public_xtra->data[i]) = (void*)dummy1;

          break;
        }

        default:
        {
          InputError("Error: invalid type <%s> for key <%s>\n",
                     switch_name, key);
        }
      }
    }
  }

  (public_xtra->num_reservoirs) = (public_xtra->num_press_reservoirs) + (public_xtra->num_flux_reservoirs);

  NA_FreeNameArray(methodflux_na);
  NA_FreeNameArray(methodpress_na);
  NA_FreeNameArray(mechanism_na);
  NA_FreeNameArray(action_na);
  NA_FreeNameArray(inputtype_na);

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
  Type1         *dummy1;

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
        switch ((public_xtra->type[i]))
        {
          case 0:
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

            break;
          }

          case 1:
          {
            dummy1 = (Type1*)(public_xtra->data[i]);

            if ((dummy1->name))
            {
              tfree((dummy1->name));
            }

            tfree(dummy1);

            break;
          }
        }
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