/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *****************************************************************************/

#ifndef _GLOBALS_HEADER
#define _GLOBALS_HEADER


/*----------------------------------------------------------------
 * Globals structure
 *----------------------------------------------------------------*/

typedef struct
{
   char     run_name[256];
   char     in_file_name[256];
   char     out_file_name[256];
	    
   int      logging_level;

   int      num_procs;        /* number of processes */
   int      num_procs_x;      /* number of processes in x */
   int      num_procs_y;      /* number of processes in y */
   int      num_procs_z;      /* number of processes in z */

   /* RDF the following just doesn't seem to make sense here */
   Background     *background;
   Grid           *user_grid;        /* user specified grid */
   int             max_ref_level;

   /* Need access to Geometry Names from all modules */
   /* Geometry names are the names for each of the geometries; there
      may be more than one per geometry input */
   NameArray       geom_names;
   GeomSolid       **geometries;
   
   NameArray       phase_names;
   NameArray       contaminant_names;

   /* Timing Cycle information */
   NameArray cycle_names;
   int        num_cycles;

   NameArray *interval_names;
   int       *interval_divisions;
   int      **intervals;
   int       *repeat_counts;

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

#define GlobalsRunName         (globals -> run_name)
#define GlobalsInFileName      (globals -> in_file_name)
#define GlobalsOutFileName     (globals -> out_file_name)
			      
#define GlobalsLoggingLevel    (globals -> logging_level)

#define GlobalsNumProcs        (globals -> num_procs)
#define GlobalsNumProcsX       (globals -> num_procs_x)
#define GlobalsNumProcsY       (globals -> num_procs_y)
#define GlobalsNumProcsZ       (globals -> num_procs_z)

#define GlobalsBackground      (globals -> background)
#define GlobalsUserGrid        (globals -> user_grid)
#define GlobalsMaxRefLevel     (globals -> max_ref_level)

#define GlobalsGeomNames       (globals -> geom_names)
#define GlobalsPhaseNames      (globals -> phase_names)

#define GlobalsCycleNames         (globals -> cycle_names)
#define GlobalsNumCycles          (globals -> num_cycles)
#define GlobalsIntervalDivisions  (globals -> interval_divisions)
#define GlobalsIntervalNames      (globals -> interval_names)
#define GlobalsIntervals          (globals -> intervals)
#define GlobalsRepeatCounts       (globals -> repeat_counts)

#define GlobalsContaminatNames    (globals -> contaminant_names)
#define GlobalsGeometries    (globals -> geometries)

#endif
