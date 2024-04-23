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

#ifndef _GLOBALS_HEADER
#define _GLOBALS_HEADER

#include "Parflow.hxx"

#ifdef HAVE_SAMRAI
#include "SAMRAI/tbox/Pointer.h"
#endif

/*----------------------------------------------------------------
 * Globals structure
 *----------------------------------------------------------------*/
typedef struct _Globals {
  char run_name[256];
  char in_file_name[256];
  char out_file_name[256];

  int logging_level;

  int num_procs;              /* number of processes */
  int num_procs_x;            /* number of processes in x */
  int num_procs_y;            /* number of processes in y */
  int num_procs_z;            /* number of processes in z */


  /* This process in PxQxR process grid */
  int p;
  int q;
  int r;

  /* RDF the following just doesn't seem to make sense here */
  Background     *background;
  Grid           *user_grid;         /* user specified grid */
  int max_ref_level;

  /* Need access to Geometry Names from all modules */
  /* Geometry names are the names for each of the geometries; there
   * may be more than one per geometry input */
  NameArray geom_names;
  GeomSolid       **geometries;

  NameArray phase_names;
  NameArray contaminant_names;

  /* Timing Cycle information */
  NameArray cycle_names;
  int num_cycles;

  NameArray *interval_names;
  int       *interval_divisions;
  int      **intervals;
  int       *repeat_counts;

  // SGS For debugging remove
  Grid     *grid3d;
  Grid     *grid2d;

  int use_clustering;

#ifdef HAVE_SAMRAI
  SAMRAI::tbox::Pointer < Parflow > parflow_simulation;
#endif
} Globals;

#ifdef PARFLOW_GLOBALS
amps_ThreadLocalDcl(Globals  *, globals_ptr);
amps_ThreadLocalDcl(IDB *, input_database);
#else
amps_ThreadLocalDcl(extern Globals  *, globals_ptr);
amps_ThreadLocalDcl(extern IDB *, input_database);
#endif

#define globals amps_ThreadLocal(globals_ptr)

/*--------------------------------------------------------------------------
 * Accessor macros: Globals
 *--------------------------------------------------------------------------*/

#define GlobalsRunName         (globals->run_name)
#define GlobalsInFileName      (globals->in_file_name)
#define GlobalsOutFileName     (globals->out_file_name)

#define GlobalsLoggingLevel    (globals->logging_level)

#define GlobalsNumProcs        (globals->num_procs)
#define GlobalsNumProcsX       (globals->num_procs_x)
#define GlobalsNumProcsY       (globals->num_procs_y)
#define GlobalsNumProcsZ       (globals->num_procs_z)

#define GlobalsP       (globals->p)
#define GlobalsQ       (globals->q)
#define GlobalsR       (globals->r)

#define GlobalsBackground      (globals->background)
#define GlobalsUserGrid        (globals->user_grid)
#define GlobalsMaxRefLevel     (globals->max_ref_level)

#define GlobalsGeomNames       (globals->geom_names)
#define GlobalsPhaseNames      (globals->phase_names)

#define GlobalsCycleNames         (globals->cycle_names)
#define GlobalsNumCycles          (globals->num_cycles)
#define GlobalsIntervalDivisions  (globals->interval_divisions)
#define GlobalsIntervalNames      (globals->interval_names)
#define GlobalsIntervals          (globals->intervals)
#define GlobalsRepeatCounts       (globals->repeat_counts)

#define GlobalsContaminatNames    (globals->contaminant_names)
#define GlobalsGeometries         (globals->geometries)

#define GlobalsParflowSimulation   (globals->parflow_simulation)

#define GlobalsUseClustering      (globals->use_clustering)

#define pqr_to_process(p, q, r, P, Q, R)  ((((r) * (Q)) + (q)) * (P) + (p))

#endif
