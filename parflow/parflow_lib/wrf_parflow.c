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
amps_ThreadLocalDcl(Vector   *, evap_trans);
amps_ThreadLocalDcl(int      *, top);

void wrfparflowinit_() {
   // SGS this needs to come from somewhere
   char *input_file = "sgs_richards_test";

   Grid         *grid;

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

   /* Create the flow grid */
   grid = CreateGrid(GlobalsUserGrid);
   
   /* Create the PF vector holding flux */
   amps_ThreadLocal(evap_trans) = NewVector( grid, 1, 1 );
   InitVectorAll(amps_ThreadLocal(evap_trans), 0.0);

   ProblemData *problem_data      = GetProblemDataRichards(amps_ThreadLocal(solver));
   Problem     *problem           = GetProblemRichards(amps_ThreadLocal(solver));
   PFModule    *ic_phase_pressure = GetICPhasePressureRichards(amps_ThreadLocal(solver));

   amps_ThreadLocal(top)          = ComputeTop(ic_phase_pressure, problem, problem_data, amps_ThreadLocal(evap_trans));
}

void wrfparflowadvance_(float  *current_time, 
			float  *dt,
                        double *wrf_flux,
                        double *wrf_pressure,
                        double *wrf_porosity,
                        double *wrf_saturation,
			int    *num_soil_layers,
                        int    *ghost_size)
{
   double        stop_time           = *current_time + *dt;
   
   Vector       *pressure_out;
   Vector       *porosity_out;
   Vector       *saturation_out;
   
   // AdvanceRichards should not use select_time_step module to compute dt.
   // Use the provided dt with possible subcycling if it does not converge.
   int compute_time_step = 0; 

   WRF2PF(wrf_flux, *num_soil_layers, *ghost_size, amps_ThreadLocal(evap_trans), amps_ThreadLocal(top));
   
   AdvanceRichards(amps_ThreadLocal(solver),
		   *current_time, 
		   stop_time, 
		   *dt, 
		   compute_time_step,
		   amps_ThreadLocal(evap_trans),
		   &pressure_out, 
		   &porosity_out,
		   &saturation_out);

   PF2WRF(pressure_out,   wrf_pressure,   *num_soil_layers, *ghost_size, amps_ThreadLocal(top));
   PF2WRF(porosity_out,   wrf_porosity,   *num_soil_layers, *ghost_size, amps_ThreadLocal(top));
   PF2WRF(saturation_out, wrf_saturation, *num_soil_layers, *ghost_size, amps_ThreadLocal(top));
}



/* 
 * Copy data from a WRF array to a PF vector based on the
 * k-index data for the top of the domain.
 */
void WRF2PF(
   double *wrf_array,   /* WRF array */
   int     wrf_depth,   /* Depth (Z) of WRF array, X,Y are assumed 
			   to be same as PF vector subgrid */
   int     ghost_size,  /* Number of ghost cells */
   Vector *pf_vector,
   int *top) 
{

   Grid       *grid     = VectorGrid(pf_vector);
   int sg;

   ForSubgridI(sg, GridSubgrids(grid))
   {
      Subgrid *subgrid = GridSubgrid(grid, sg);
      
      int ix = SubgridIX(subgrid);
      int iy = SubgridIY(subgrid);
      // int iz = SubgridIZ(subgrid);

      int nx = SubgridNX(subgrid);
      int ny = SubgridNY(subgrid);
      int nz = SubgridNZ(subgrid);

      int wrf_nx =  nx + 2 * ghost_size;
      int wrf_ny =  ny + 2 * ghost_size;
      int wrf_nz =  wrf_depth;

      Subvector *subvector = VectorSubvector(pf_vector, sg);
      
      int subvector_nx = SubvectorNX(subvector);
      int subvector_ny = SubvectorNY(subvector);
      int subvector_nz = SubvectorNZ(subvector);

      double *subvector_data = SubvectorElt(subvector, ix, iy, SubgridIZ(subgrid));

      int i, j, k;


      for (i = ix; i < ix + nx; i++)
      {					       
	 for (j = iy; j < iy + ny; j++)		
	 {
	    int top_index = (i-ix) + ((j-iy) * nx);

	    // SGS What to do if near bottom such that
	    // there are not wrf_depth values?
	    int iz = top[top_index] - (wrf_depth - 1);

	    for (k = iz; k < iz + wrf_depth; k++)		
	    {
	       int pf_index = SubvectorEltIndex(subvector, i, j, k);
	       int wrf_index = (i-ix + ghost_size) 
		  + ((j-iy + ghost_size) * (wrf_nx) 
		     + (wrf_depth - (k - iz) - 1 ) * wrf_nx * wrf_ny);
	       subvector_data[pf_index] = wrf_array[wrf_index];
	    }
	 }
      }
   }

}

/* 
 * Copy data from a PF vector to a WRF array based on the
 * k-index data for the top of the domain.
 */
void PF2WRF(
   Vector *pf_vector,
   double *wrf_array,   /* WRF array */
   int     wrf_depth,   /* Depth (Z) of WRF array, X,Y are assumed 
			   to be same as PF vector subgrid */
   int     ghost_size,  /* Number of ghost cells */
   int *top) 
{

   Grid       *grid     = VectorGrid(pf_vector);
   int sg;

   ForSubgridI(sg, GridSubgrids(grid))
   {
      Subgrid *subgrid = GridSubgrid(grid, sg);
      
      int ix = SubgridIX(subgrid);
      int iy = SubgridIY(subgrid);
      // int iz = SubgridIZ(subgrid);

      int nx = SubgridNX(subgrid);
      int ny = SubgridNY(subgrid);
      int nz = SubgridNZ(subgrid);

      int wrf_nx =  nx + 2 * ghost_size;
      int wrf_ny =  ny + 2 * ghost_size;
      int wrf_nz =  wrf_depth;

      Subvector *subvector = VectorSubvector(pf_vector, sg);
      
      int subvector_nx = SubvectorNX(subvector);
      int subvector_ny = SubvectorNY(subvector);
      int subvector_nz = SubvectorNZ(subvector);

      double *subvector_data = SubvectorElt(subvector, ix, iy, SubgridIZ(subgrid));

      int i, j, k;


      for (i = ix; i < ix + nx; i++)
      {					       
	 for (j = iy; j < iy + ny; j++)		
	 {
	    int top_index = (i-ix) + ((j-iy) * nx);

	    // SGS What to do if near bottom such that
	    // there are not wrf_depth values?
	    int iz = top[top_index] - (wrf_depth - 1);

	    for (k = iz; k < iz + wrf_depth; k++)		
	    {
	       int pf_index = SubvectorEltIndex(subvector, i, j, k);
	       int wrf_index = (i-ix + ghost_size) 
		  + ((j-iy + ghost_size) * (wrf_nx) 
		     + (wrf_depth - (k - iz) - 1 ) * wrf_nx * wrf_ny);
	       wrf_array[wrf_index] = subvector_data[pf_index] ;
	    }
	 }
      }
   }

}
