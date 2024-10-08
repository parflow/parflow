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

#ifndef _RESERVOIR_HEADER
#define _RESERVOIR_HEADER


/*----------------------------------------------------------------
 * Reservoir Physical Values structure
 *----------------------------------------------------------------*/

typedef struct {
  int number;
  char          *name;
  double intake_x_lower, intake_y_lower, z_lower;
  double intake_x_upper, intake_y_upper, z_upper;
  int has_secondary_intake_cell;
  double secondary_intake_x_lower, secondary_intake_y_lower;
  int intake_cell_mpi_rank, secondary_intake_cell_mpi_rank, release_cell_mpi_rank;
  double secondary_intake_x_upper, secondary_intake_y_upper;
  double release_x_lower, release_y_lower, release_z_lower;
  double release_x_upper, release_y_upper, release_z_upper;
  double max_storage, min_release_storage, storage, release_rate;
  double intake_amount_since_last_print, release_amount_since_last_print, release_amount_in_solver;
  Subgrid       *intake_subgrid;
  Subgrid       *secondary_intake_subgrid;
  Subgrid       *release_subgrid;
  double size;
    #ifdef PARFLOW_HAVE_MPI
  MPI_Comm mpi_communicator;
    #endif
} ReservoirDataPhysical;

/*------------------------------------------------------------------
 * Reservoir Data structure
 *----------------------------------------------------------------*/

typedef struct {
  int num_reservoirs;
  int overland_flow_solver;
  ReservoirDataPhysical  **reservoir_physicals;
} ReservoirData;

/*--------------------------------------------------------------------------
 * Accessor macros: ReservoirDataPhysical
 *--------------------------------------------------------------------------*/
#define ReservoirDataPhysicalNumber(reservoir_data_physical) \
        ((reservoir_data_physical)->number)

#define ReservoirDataPhysicalName(reservoir_data_physical) \
        ((reservoir_data_physical)->name)

#define ReservoirDataPhysicalMpiCommunicator(reservoir_data_physical) \
        ((reservoir_data_physical)->mpi_communicator)

#define ReservoirDataPhysicalIntakeCellMpiRank(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_cell_mpi_rank)

#define ReservoirDataPhysicalIntakeXLower(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_x_lower)

#define ReservoirDataPhysicalIntakeYLower(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_y_lower)

#define ReservoirDataPhysicalSecondaryIntakeCellMpiRank(reservoir_data_physical) \
        ((reservoir_data_physical)->secondary_intake_cell_mpi_rank)

#define ReservoirDataPhysicalSecondaryIntakeXLower(reservoir_data_physical) \
        ((reservoir_data_physical)->secondary_intake_x_lower)

#define ReservoirDataPhysicalSecondaryIntakeYLower(reservoir_data_physical) \
        ((reservoir_data_physical)->secondary_intake_y_lower)


#define ReservoirDataPhysicalIntakeXUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_x_upper)

#define ReservoirDataPhysicalIntakeYUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_y_upper)

#define ReservoirDataPhysicalSecondaryIntakeXUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_x_upper)

#define ReservoirDataPhysicalSecondaryIntakeYUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_y_upper)


#define ReservoirDataPhysicalIntakeSubgrid(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_subgrid)

#define ReservoirDataPhysicalSecondaryIntakeSubgrid(reservoir_data_physical) \
        ((reservoir_data_physical)->secondary_intake_subgrid)

#define ReservoirDataPhysicalHasSecondaryIntakeCell(reservoir_data_physical) \
        ((reservoir_data_physical)->has_secondary_intake_cell)

#define ReservoirDataPhysicalReleaseCellMpiRank(reservoir_data_physical) \
        ((reservoir_data_physical)->release_cell_mpi_rank)

#define ReservoirDataPhysicalReleaseXLower(reservoir_data_physical) \
        ((reservoir_data_physical)->release_x_lower)

#define ReservoirDataPhysicalReleaseYLower(reservoir_data_physical) \
        ((reservoir_data_physical)->release_y_lower)

#define ReservoirDataPhysicalReleaseXUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->release_x_upper)

#define ReservoirDataPhysicalReleaseYUpper(reservoir_data_physical) \
        ((reservoir_data_physical)->release_y_upper)


#define ReservoirDataPhysicalReleaseSubgrid(reservoir_data_physical) \
        ((reservoir_data_physical)->release_subgrid)

#define ReservoirDataPhysicalSize(reservoir_data_physical) \
        ((reservoir_data_physical)->size)


#define ReservoirDataPhysicalMaxStorage(reservoir_data_physical) \
        ((reservoir_data_physical)->max_storage)

#define ReservoirDataPhysicalStorage(reservoir_data_physical) \
        ((reservoir_data_physical)->storage)

#define ReservoirDataPhysicalReleaseAmountSinceLastPrint(reservoir_data_physical) \
        ((reservoir_data_physical)->release_amount_since_last_print)

#define ReservoirDataPhysicalIntakeAmountSinceLastPrint(reservoir_data_physical) \
        ((reservoir_data_physical)->intake_amount_since_last_print)

#define ReservoirDataPhysicalReleaseAmountInSolver(reservoir_data_physical) \
        ((reservoir_data_physical)->release_amount_in_solver)

#define ReservoirDataPhysicalMinReleaseStorage(reservoir_data_physical) \
        ((reservoir_data_physical)->min_release_storage)

#define ReservoirDataPhysicalReleaseRate(reservoir_data_physical) \
        ((reservoir_data_physical)->release_rate)



/*--------------------------------------------------------------------------
 * Accessor macros: ReservoirData
 *--------------------------------------------------------------------------*/



/*---------------------------- Flux reservoir data ------------------------------*/
#define ReservoirDataNumReservoirs(reservoir_data)      ((reservoir_data)->num_reservoirs)

#define ReservoirDataOverlandFlowSolver(reservoir_data)      ((reservoir_data)->overland_flow_solver)

#define ReservoirDataReservoirPhysicals(reservoir_data) \
        ((reservoir_data)->reservoir_physicals)
#define ReservoirDataReservoirPhysical(reservoir_data, i) \
        ((reservoir_data)->reservoir_physicals[i])




/*--------------------------------------------------------------------------
 * Reservoir Data constants used in the program.
 *--------------------------------------------------------------------------*/

/*       Actions        */
#define INJECTION_RESERVOIR  0
#define EXTRACTION_RESERVOIR 1

/*       Methods        */
/* These should match  with */
/* the Name Array Defs from */
/* the  Reservoir Package input. */
#define FLUX_STANDARD  0
#define FLUX_WEIGHTED  1
#define FLUX_PATTERNED 2

#define OVERLAND_FLOW 0
#define OVERLAND_KINEMATIC 1

/*     Write Options      */
#define RESERVOIRDATA_WRITEHEADER     1

#endif