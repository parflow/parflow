/*
 * BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER
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

void Solve()
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
	 amps_ThreadLocal(Solver_module) = PFModuleNewModuleType(SolverNewPublicXtraInvoke, SolverRichards, ("Solver"));
	 break;
      }
      case 1:
      {
	 amps_ThreadLocal(Solver_module) = PFModuleNewModuleType(SolverNewPublicXtraInvoke, SolverDiffusion, ("Solver"));
	 break;
      }
      case 2:
      {
	 amps_ThreadLocal(Solver_module) = PFModuleNewModuleType(SolverNewPublicXtraInvoke, SolverImpes, ("Solver"));
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
