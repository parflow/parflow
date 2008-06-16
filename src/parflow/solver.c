/*
 * BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright notice,
 * contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 * ********************************************************************EHEADER
 * */
/******************************************************************************
 *
 * Routines for initializing the solver.
 *
 *****************************************************************************/

#include "parflow.h"
#include "solver.h"

amps_ThreadLocalDcl(PFModule *, Solver_module);

/*--------------------------------------------------------------------------
 * Solve
 *--------------------------------------------------------------------------*/

void
                Solve()
{
   PFModule       *solver;

   
   
   BeginTiming(SolverSetupTimingIndex);
   NewSolver();

   solver = PFModuleNewInstance(amps_ThreadLocal(Solver_module), ());
   EndTiming(SolverSetupTimingIndex);

   BeginTiming(SolverTimingIndex);
   PFModuleInvoke(void, solver, ());
   EndTiming(SolverTimingIndex);

   BeginTiming(SolverCleanupTimingIndex);
   PFModuleFreeInstance(solver);
   EndTiming(SolverCleanupTimingIndex);

   FreeSolver();
}


/*--------------------------------------------------------------------------
 * NewSolver
 *--------------------------------------------------------------------------*/

void
                NewSolver()
{
   char key[IDB_MAX_KEY_LEN];
   
   char *switch_name;

   int solver;
   NameArray solver_na;

   solver_na = NA_NewNameArray("Richards Diffusion Impes");

   /*-----------------------------------------------------------------------
    * Read global solver input
    *-----------------------------------------------------------------------*/

   GlobalsNumProcsX = GetIntDefault("Process.Topology.P", 1);
   GlobalsNumProcsY = GetIntDefault("Process.Topology.Q", 1);
   GlobalsNumProcsZ = GetIntDefault("Process.Topology.R", 1);

   GlobalsNumProcs = amps_Size(amps_CommWorld);

   GlobalsBackground = ReadBackground();

   GlobalsUserGrid = ReadUserGrid();

   SetBackgroundBounds(GlobalsBackground, GlobalsUserGrid);

   GlobalsMaxRefLevel = 0;

   switch_name = GetStringDefault("Solver", "Impes");
   solver  = NA_NameToIndex(solver_na, switch_name);

   switch (solver)
   {
      case 0: 
      {
	 amps_ThreadLocal(Solver_module) = PFModuleNewModule(SolverRichards, ("Solver"));
	 break;
      }
      case 1:
      {
	 amps_ThreadLocal(Solver_module) = PFModuleNewModule(SolverDiffusion, ("Solver"));
	 break;
      }
      case 2:
      {
	 amps_ThreadLocal(Solver_module) = PFModuleNewModule(SolverImpes, ("Solver"));
	 break;
      }
      default:
      {
	 InputError("Error: Invalid value <%s> for key <%s>\n", switch_name,
		     key);
      }
   }
   NA_FreeNameArray(solver_na);
}

/*--------------------------------------------------------------------------
 * FreeSolver
 *--------------------------------------------------------------------------*/

void
                FreeSolver()
{
   if (amps_ThreadLocal(Solver_module))
   {
      PFModuleFreeModule(amps_ThreadLocal(Solver_module));
      amps_ThreadLocal(Solver_module) = NULL;
   }

   FreeUserGrid(GlobalsUserGrid);

   FreeBackground(GlobalsBackground);
}
