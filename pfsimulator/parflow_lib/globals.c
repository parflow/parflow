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

/** @file
 * @brief Routines for manipulating global structures.
 */

#define PARFLOW_GLOBALS

#include "parflow.h"
#include "globals.h"

#include <limits.h>
#include <stddef.h>

/*--------------------------------------------------------------------------
 * NewGlobals
 *--------------------------------------------------------------------------*/
void   NewGlobals(char *run_name)
{
  globals_ptr = ctalloc(Globals, 1);

  sprintf(GlobalsRunName, "%s", run_name);
  sprintf(GlobalsInFileName, "%s.%s", run_name, "pfidb");
  sprintf(GlobalsOutFileName, "%s.%s", run_name, "out");

  globals_ptr->logging_level = 0;

#ifdef min
#undef min
#endif
  globals_ptr->num_procs = INT_MIN;
  globals_ptr->num_procs_x = INT_MIN;
  globals_ptr->num_procs_y = INT_MIN;
  globals_ptr->num_procs_z = INT_MIN;

  globals_ptr->background = 0;
  globals_ptr->user_grid = 0;
  globals_ptr->max_ref_level = -1;

  globals_ptr->geom_names = 0;
  globals_ptr->geometries = 0;

  globals_ptr->phase_names = 0;
  globals_ptr->contaminant_names = 0;

  /* Timing Cycle information */
  globals_ptr->cycle_names = 0;
  globals_ptr->num_cycles = 0;

  globals_ptr->interval_names = 0;
  globals_ptr->interval_divisions = 0;
  globals_ptr->intervals = 0;
  globals_ptr->repeat_counts = 0;

  globals_ptr->use_clustering = 0;
}


/*--------------------------------------------------------------------------
 * FreeGlobals
 *--------------------------------------------------------------------------*/

void  FreeGlobals()
{
  free(globals_ptr);
}

/*--------------------------------------------------------------------------
 * LogGlobals
 *--------------------------------------------------------------------------*/

void  LogGlobals()
{
  IfLogging(0)
  {
    FILE *log_file;

    log_file = OpenLogFile("Globals");

    fprintf(log_file, "Run Name: %s\n",
            GlobalsRunName);
    fprintf(log_file, "Logging Level = %d\n",
            GlobalsLoggingLevel);
    fprintf(log_file, "Num processes = %d\n",
            GlobalsNumProcs);
    fprintf(log_file, "Process grid = (%d,%d,%d)\n",
            GlobalsNumProcsX,
            GlobalsNumProcsY,
            GlobalsNumProcsZ);

    CloseLogFile(log_file);
  }
}

#if PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA

/**
 * @brief A struct for constant global data in read-only device memory.
 **/
__constant__ Globals dev_globals;

/**
 * @brief A struct member for constant global data in read-only device memory.
 **/
__constant__ Background dev_background;

/**
 * @brief A function to copy constant global data to read-only device memory.
 **/
void CopyGlobalsToDevice()
{
  // Temporary pointers for getting the symbol addresses
  Globals *tmp_globals_ptr;
  Background *tmp_background_ptr;

  // Get the symbol addresses from the __constant__ allocations
  CUDA_ERR(cudaGetSymbolAddress((void**)&tmp_globals_ptr, dev_globals));
  CUDA_ERR(cudaGetSymbolAddress((void**)&tmp_background_ptr, dev_background));

  // Copy background data from host to device
  CUDA_ERR(cudaMemcpy(tmp_background_ptr, globals->background, sizeof(Background), cudaMemcpyHostToDevice));

  // Assign dev_background address to dev_globals.background
  CUDA_ERR(cudaMemcpyToSymbol(dev_globals, &tmp_background_ptr, sizeof(Background*), offsetof(Globals, background), cudaMemcpyHostToDevice));

  // Assign dev_globals address to dev_globals_ptr
  CUDA_ERR(cudaMemcpyToSymbol(dev_globals_ptr, &tmp_globals_ptr, sizeof(Globals*), 0, cudaMemcpyHostToDevice));
}
#endif // PARFLOW_ACC_BACKEND == PARFLOW_BACKEND_CUDA
