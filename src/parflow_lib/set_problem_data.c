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
 * Module for initializing the problem structure.
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct
{
   PFModule  *geometries;
   PFModule  *domain;
   PFModule  *permeability;
   PFModule  *porosity;
   PFModule  *wells;
   PFModule  *bc_pressure;
   PFModule  *bc_temperature;
   PFModule  *specific_storage; //sk
   PFModule  *x_slope; //sk
   PFModule  *y_slope; //sk
   PFModule  *mann; //sk
   int        site_data_not_formed;

   /* InitInstanceXtra arguments */
   Problem   *problem;
   Grid      *grid;
   double    *temp_data;

} InstanceXtra;


/*--------------------------------------------------------------------------
 * SetProblemData
 *--------------------------------------------------------------------------*/

void          SetProblemData(problem_data)
ProblemData  *problem_data;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   PFModule      *geometries       = (instance_xtra -> geometries);
   PFModule      *domain           = (instance_xtra -> domain);
   PFModule      *permeability     = (instance_xtra -> permeability);
   PFModule      *porosity         = (instance_xtra -> porosity);
   PFModule      *wells            = (instance_xtra -> wells);
   PFModule      *bc_pressure      = (instance_xtra -> bc_pressure);
   PFModule      *bc_temperature   = (instance_xtra -> bc_temperature);
   PFModule      *specific_storage = (instance_xtra -> specific_storage); //sk
   PFModule      *x_slope          = (instance_xtra -> x_slope); //sk
   PFModule      *y_slope          = (instance_xtra -> y_slope); //sk
   PFModule      *mann             = (instance_xtra -> mann); //sk

   /* Note: the order in which these modules are called is important */
   PFModuleInvoke(void, wells, (problem_data));
   if ( (instance_xtra -> site_data_not_formed) )
   {
      PFModuleInvoke(void, geometries, (problem_data));
      PFModuleInvoke(void, domain, (problem_data));
      PFModuleInvoke(void, permeability,
		     (problem_data,
		      ProblemDataPermeabilityX(problem_data),
		      ProblemDataPermeabilityY(problem_data),
		      ProblemDataPermeabilityZ(problem_data),
		      ProblemDataNumSolids(problem_data),
                      ProblemDataSolids(problem_data),
                      ProblemDataGrSolids(problem_data)));
      PFModuleInvoke(void, porosity,
		     (problem_data,
                      ProblemDataPorosity(problem_data),
		      ProblemDataNumSolids(problem_data),
                      ProblemDataSolids(problem_data),
                      ProblemDataGrSolids(problem_data)));
      PFModuleInvoke(void, specific_storage,                 //sk
		     (problem_data,
		      ProblemDataSpecificStorage(problem_data)));
      PFModuleInvoke(void, x_slope,                 //sk
		     (problem_data,
		      ProblemDataTSlopeX(problem_data)));
      PFModuleInvoke(void, y_slope,                 //sk
		     (problem_data,
		      ProblemDataTSlopeY(problem_data)));
      PFModuleInvoke(void, mann,                 //sk
		     (problem_data,
		      ProblemDataMannings(problem_data)));
      (instance_xtra -> site_data_not_formed) = 0;
   }
   PFModuleInvoke(void, bc_pressure, (problem_data));
   PFModuleInvoke(void, bc_temperature, (problem_data));
}


/*--------------------------------------------------------------------------
 * SetProblemDataInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SetProblemDataInitInstanceXtra(problem, grid, grid2d, temp_data)
Problem   *problem;
Grid      *grid,*grid2d;
double    *temp_data;
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra;


   if ( PFModuleInstanceXtra(this_module) == NULL )
      instance_xtra = ctalloc(InstanceXtra, 1);
   else
      instance_xtra = PFModuleInstanceXtra(this_module);

   /*-----------------------------------------------------------------------
    * Initialize data associated with `problem'
    *-----------------------------------------------------------------------*/

   if ( problem != NULL )
      (instance_xtra -> problem) = problem;

   /*-----------------------------------------------------------------------
    * Initialize data associated with `grid'
    *-----------------------------------------------------------------------*/

   if ( grid != NULL )
      (instance_xtra -> grid) = grid;

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
      (instance_xtra -> temp_data) = temp_data;

   /*-----------------------------------------------------------------------
    * Initialize module instances
    *-----------------------------------------------------------------------*/

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> geometries) =
         PFModuleNewInstance(ProblemGeometries(problem), (grid));
      (instance_xtra -> domain) =
         PFModuleNewInstance(ProblemDomain(problem), (grid));
      (instance_xtra -> permeability) =
         PFModuleNewInstance(ProblemPermeability(problem), (grid, temp_data));
      (instance_xtra -> porosity) =
         PFModuleNewInstance(ProblemPorosity(problem), (grid, temp_data));
      (instance_xtra -> specific_storage) =                                   //sk
	     PFModuleNewInstance(ProblemSpecStorage(problem), ());
      (instance_xtra -> x_slope) =                                   //sk
	     PFModuleNewInstance(ProblemXSlope(problem), (grid));
      (instance_xtra -> y_slope) =                                   //sk
	     PFModuleNewInstance(ProblemYSlope(problem), (grid));
      (instance_xtra -> mann) =                                   //sk
	     PFModuleNewInstance(ProblemMannings(problem), (grid));

      (instance_xtra -> site_data_not_formed) = 1;
      (instance_xtra -> wells) =
          PFModuleNewInstance(ProblemWellPackage(problem), ());

      (instance_xtra -> bc_pressure) =
          PFModuleNewInstance(ProblemBCPressurePackage(problem), (problem));
      (instance_xtra -> bc_temperature) =
          PFModuleNewInstance(ProblemBCTemperaturePackage(problem), (problem));

   }
   else
   {
      PFModuleReNewInstance((instance_xtra -> geometries), (grid));
      PFModuleReNewInstance((instance_xtra -> domain), (grid));
      PFModuleReNewInstance((instance_xtra -> permeability),
			    (grid, temp_data));
      PFModuleReNewInstance((instance_xtra -> porosity),
			    (grid, temp_data));
      PFModuleReNewInstance((instance_xtra -> specific_storage), ());    //sk
      PFModuleReNewInstance((instance_xtra -> x_slope), (grid));    //sk
      PFModuleReNewInstance((instance_xtra -> y_slope), (grid));    //sk
      PFModuleReNewInstance((instance_xtra -> mann), (grid));    //sk
      PFModuleReNewInstance((instance_xtra -> wells), ());
      PFModuleReNewInstance((instance_xtra -> bc_temperature), (problem));
   }

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * SetProblemDataFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);


   if(instance_xtra)
   {
      PFModuleFreeInstance(instance_xtra -> bc_temperature);
      PFModuleFreeInstance(instance_xtra -> wells);

      PFModuleFreeInstance(instance_xtra -> geometries);
      PFModuleFreeInstance(instance_xtra -> domain);
      PFModuleFreeInstance(instance_xtra -> porosity);
      PFModuleFreeInstance(instance_xtra -> permeability);
      PFModuleFreeInstance(instance_xtra -> specific_storage);   //sk
      PFModuleFreeInstance(instance_xtra -> x_slope);   //sk
      PFModuleFreeInstance(instance_xtra -> y_slope);   //sk
      PFModuleFreeInstance(instance_xtra -> mann);   //sk

      tfree(instance_xtra);
   }
}


/*--------------------------------------------------------------------------
 * SetProblemDataNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *SetProblemDataNewPublicXtra()
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;


#if 0
   public_xtra = ctalloc(PublicXtra, 1);
#endif
   public_xtra = NULL;

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * SetProblemDataFreePublicXtra
 *--------------------------------------------------------------------------*/

void  SetProblemDataFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);


   if(public_xtra)
   {
      tfree(public_xtra);
   }
}


/*--------------------------------------------------------------------------
 * SetProblemDataSizeOfTempData
 *--------------------------------------------------------------------------*/

int       SetProblemDataSizeOfTempData()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra   = PFModuleInstanceXtra(this_module);

   int  sz = 0;


   /* set `sz' to max of each of the called modules */
   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> geometries));
   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> porosity));
   sz = max(sz, PFModuleSizeOfTempData(instance_xtra -> permeability));

   return sz;
}
