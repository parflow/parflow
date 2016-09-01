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
#ifdef HAVE_P4EST
   const char     *Nkey[3];
   const char     *mkey[3];
   int             P[3],l, N, m, p, t, sum;
#endif
   solver_na = NA_NewNameArray("Richards Diffusion Impes");

   /*-----------------------------------------------------------------------
    * Read global solver input
    *-----------------------------------------------------------------------*/

   if (!USE_P4EST){
      GlobalsNumProcsY = GetIntDefault("Process.Topology.Q", 1);
      GlobalsNumProcsX = GetIntDefault("Process.Topology.P", 1);
      GlobalsNumProcsZ = GetIntDefault("Process.Topology.R", 1);
   }else{
#ifdef HAVE_P4EST
       BeginTiming(P4ESTSetupTimingIndex);

      /** Retrieve desired dimensions of the grid */
      Nkey[0] = "ComputationalGrid.NX";
      Nkey[1] = "ComputationalGrid.NY";
      Nkey[2] = "ComputationalGrid.NZ";

      /** Retrieve desired dimensions of a subgrid */
      mkey[0] = "ComputationalSubgrid.MX";
      mkey[1] = "ComputationalSubgrid.MY";
      mkey[2] = "ComputationalSubgrid.MZ";

      /** Ensure that with the user input is possible to
        * create a consistent decomposition of the grid. If
        * not we quit the program and print an error message */
      for (t=0; t<3; ++t){
          N    = GetIntDefault(Nkey[t], 1);
          m    = GetIntDefault(mkey[t], 1);
          P[t] = N / m;
          l    = N % m;;
          sum  = 0;
          for(p = 0; p < P[t]; ++p){
             sum += ( p < l ) ? m + 1 : m;
          }
          if ( sum != N ){

              InputError("Error: invalid combination of <%s> and <%s>\n",
                         Nkey[t], mkey[t]);
          }
      }

      /** Reinterpret GlobalsNumProcs{X,Y,Z} as the number of subgrids in
        * that coordinate direction */
      GlobalsNumProcsX = P[0];
      GlobalsNumProcsY = P[1];
      GlobalsNumProcsZ = P[2];

      /*Pupulate GlobalsSubgridPoints{X,Y,Z}*/
      GlobalsSubgridPointsX = GetIntDefault(mkey[0], 1);
      GlobalsSubgridPointsY = GetIntDefault(mkey[1], 1);
      GlobalsSubgridPointsZ = GetIntDefault(mkey[2], 1);

      EndTiming(P4ESTSetupTimingIndex);
#else
      PARFLOW_ERROR("ParFlow compiled without p4est");
#endif
   }

   GlobalsNumProcs = amps_Size(amps_CommWorld);

#ifdef HAVE_P4EST
   /** Support for empty processors not yet implemented.
    *  Print an error message and quit the program */
   if ( GlobalsNumProcs >
        GlobalsNumProcsX * GlobalsNumProcsY * GlobalsNumProcsZ){
       PARFLOW_ERROR("Number of processors bigger as the number of subgrids");
   }
#endif

   GlobalsBackground = ReadBackground();

   GlobalsUserGrid = ReadUserGrid();

   SetBackgroundBounds(GlobalsBackground, GlobalsUserGrid);

   GlobalsMaxRefLevel = 0;



   /*-----------------------------------------------------------------------
    * Initialize SAMRAI hierarchy
    *-----------------------------------------------------------------------*/
   // SGS FIXME is this a good place for this?  need UserGrid

#ifdef HAVE_SAMRAI
   // SGS FIXME is this correct for restarts?
   double time = 0.0;
   GlobalsParflowSimulation -> initializePatchHierarchy(time);
#endif

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
