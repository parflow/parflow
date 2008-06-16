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
 * Routine to be invoked by WRF model.
 *
 *****************************************************************************/

#include "parflow.h"
#include "solver.h"

amps_ThreadLocalDcl(PFModule *, Solver_module);
amps_ThreadLocalDcl(PFModule *, solver);

void wrfparflowinit_() {
   // SGS this needs to come from somewhere
   char *input_file = "sgs_richards_test";

   printf("Hello from WRFParflowInit\n");

   /* Begin of main includes */

   /*-----------------------------------------------------------------------
    * Initialize AMPS 
    *-----------------------------------------------------------------------*/
   
   // SGS this is wrong; need initialize from existing MPI state.
   if (amps_EmbeddedInit())
   {
      amps_Printf("Error: initalization failed\n");
      exit(1);
   }
   
   /*-----------------------------------------------------------------------
    * Set up globals structure
    *-----------------------------------------------------------------------*/
   NewGlobals(input_file);
   
   /*-----------------------------------------------------------------------
    * Read the Users Input Deck
    *-----------------------------------------------------------------------*/
   amps_ThreadLocal(input_database) = IDB_NewDB(GlobalsInFileName);
   
   /*-----------------------------------------------------------------------
    * Setup log printing
    *-----------------------------------------------------------------------*/
   NewLogging();
   
   /*-----------------------------------------------------------------------
    * Setup timing table
    *-----------------------------------------------------------------------*/
   NewTiming();
   
   /* End of main includes */
   
   /* Begin of Solver includes */
   
   GlobalsNumProcsX = GetIntDefault("Process.Topology.P", 1);
   GlobalsNumProcsY = GetIntDefault("Process.Topology.Q", 1);
   GlobalsNumProcsZ = GetIntDefault("Process.Topology.R", 1);
   
   GlobalsNumProcs = amps_Size(amps_CommWorld);
   
   GlobalsBackground = ReadBackground();
   
   GlobalsUserGrid = ReadUserGrid();
   
   SetBackgroundBounds(GlobalsBackground, GlobalsUserGrid);
   
   GlobalsMaxRefLevel = 0;
   
   amps_ThreadLocal(Solver_module) = PFModuleNewModule(SolverRichards, ("Solver"));
   
   amps_ThreadLocal(solver) = PFModuleNewInstance(amps_ThreadLocal(Solver_module), ());
   
   /* End of solver includes */
   
   SetupRichards(amps_ThreadLocal(solver));
}

void wrfparflowadvance_() 
{
   // SGS These need to come from arguments; passed by WRF
   double        start_time          = 0.0;
   double        stop_time           = 0.001;
   // SGS This looks very odd....came from sgs_richards_test logfile but 
   // seems very small.
   double        dt                  = 1.776357e-18;
   
   Grid         *grid;
   
   Vector       *pressure_out;
   Vector       *porosity_out;
   Vector       *saturation_out;
   
   Vector       *evap_trans;

   // AdvanceRichards should not use select_time_step module to compute dt.
   // Use the provided dt with possible subcycling if it does not converge.
   int compute_time_step = 0; 
                           
   /* Create the flow grid */
   grid = CreateGrid(GlobalsUserGrid);
   
   evap_trans = NewVector( grid, 1, 1 );
   InitVectorAll(evap_trans, 0.0);
   
   AdvanceRichards(amps_ThreadLocal(solver),
		   start_time, 
		   stop_time, 
		   dt, 
		   compute_time_step,
		   evap_trans,
		   &pressure_out, 
		   &porosity_out,
		   &saturation_out);
}
