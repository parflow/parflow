/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "parflow.h"

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct
{

   int      type;
   void    *data;

} PublicXtra;

typedef void InstanceXtra;

typedef struct
{
   double   step;

} Type0;                       /* constant step-size */

typedef struct
{
   double   initial_step;
   double   factor;
   double   min_step;
   double   max_step;
                               
} Type1;                       /* step increases to a max value */

/*--------------------------------------------------------------------------
 * SelectTimeStep:
 *    This routine returns a time step size.
 *--------------------------------------------------------------------------*/

void     SelectTimeStep(dt, dt_info, time, problem, problem_data)

double      *dt;          /* Time step size */
char        *dt_info;     /* Character flag indicating what requirement 
			     chose the time step */
double       time;
Problem     *problem;
ProblemData *problem_data;
{
   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra   = PFModulePublicXtra(this_module);

   Type0         *dummy0;
   Type1         *dummy1;

   double         well_dt, bc_dt;

   switch((public_xtra -> type))
   {
   
   case 0:
   {
      double constant;

      dummy0 = (Type0 *)(public_xtra -> data);

      constant = (dummy0 -> step);
      
      (*dt) = constant;

      break;
   }     /* End case 0 */

   case 1:
   {
      double initial_step;
      double factor;
      double max_step;
      double min_step;

      dummy1 = (Type1 *)(public_xtra -> data);

      initial_step = (dummy1 -> initial_step);
      factor = (dummy1 -> factor);
      max_step = (dummy1 -> max_step);
      min_step = (dummy1 -> min_step);

      if ((*dt) == 0.0)
      {
	 (*dt) = initial_step;
      }
      else
      {
	 (*dt) = (*dt)*factor;
	 if ((*dt) < min_step) (*dt) = min_step;
	 if ((*dt) > max_step) (*dt) = max_step;
      }

      break;
   }     /* End case 1 */

   }     /* End switch */

   /*-----------------------------------------------------------------
    * Get delta t's for all wells and boundary conditions.
    *-----------------------------------------------------------------*/

   well_dt = TimeCycleDataComputeNextTransition(problem, time,
                WellDataTimeCycleData(ProblemDataWellData(problem_data)));

   bc_dt = TimeCycleDataComputeNextTransition(problem, time,
	 BCPressureDataTimeCycleData(ProblemDataBCPressureData(problem_data)));

   /*-----------------------------------------------------------------
    * Compute the new dt value based on time stepping criterion imposed
    * by the user or on system parameter changes.  Indicate what 
    * determined the value of `dt'.
    *-----------------------------------------------------------------*/

   if ( well_dt < 0.0 ) well_dt = (*dt);
   if ( bc_dt   < 0.0 ) bc_dt   = (*dt);
      
   if ((*dt) > well_dt)
   {
      (*dt) = well_dt;
      (*dt_info) = 'w';
   }
   else if ((*dt) > bc_dt)
   {
      (*dt) = bc_dt;
      (*dt_info) = 'b';
   }
   else
   {
      (*dt_info) = 'p';
   }


}

/*--------------------------------------------------------------------------
 * SelectTimeStepInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *SelectTimeStepInitInstanceXtra()
{
   PFModule      *this_module  = ThisPFModule;
   InstanceXtra  *instance_xtra;

   instance_xtra = NULL;

   PFModuleInstanceXtra(this_module) = instance_xtra;
   return this_module;
}


/*--------------------------------------------------------------------------
 * SelectTimeStepFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  SelectTimeStepFreeInstanceXtra()
{
   PFModule      *this_module   = ThisPFModule;
   InstanceXtra  *instance_xtra = PFModuleInstanceXtra(this_module);

   if (instance_xtra)
   {
      tfree(instance_xtra);
   }

}


/*--------------------------------------------------------------------------
 * SelectTimeStepNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *SelectTimeStepNewPublicXtra()
{

   PFModule      *this_module   = ThisPFModule;
   PublicXtra    *public_xtra;

   Type0            *dummy0;
   Type1            *dummy1;

   char *switch_name;

   NameArray type_na;

   type_na = NA_NewNameArray("Constant Growth");

   public_xtra = ctalloc(PublicXtra, 1);

   switch_name = GetString("TimeStep.Type");
   public_xtra -> type = NA_NameToIndex(type_na, switch_name);
   
   switch((public_xtra -> type))
   {
      case 0:
      {
	 dummy0 = ctalloc(Type0, 1);

	 dummy0 -> step = GetDouble("TimeStep.Value");
	 
	 (public_xtra -> data) = (void *) dummy0;

	 break;
      }
         
      case 1:
      {
	 dummy1 = ctalloc(Type1, 1);

	 dummy1 -> initial_step = GetDouble("TimeStep.InitialStep");
	 dummy1 -> factor = GetDouble("TimeStep.GrowthFactor");
	 dummy1 -> max_step = GetDouble("TimeStep.MaxStep");
	 dummy1 -> min_step = GetDouble("TimeStep.MinStep");

	 (public_xtra -> data) = (void *) dummy1;
	 
	 break;
      }

      default:
      {
	 InputError("Error: invalid type <%s> for key <%s>\n",
		    switch_name, "TimeStep.Type");
      }
   }
   

   NA_FreeNameArray(type_na);

   PFModulePublicXtra(this_module) = public_xtra;
   return this_module;
}

/*-------------------------------------------------------------------------
 * SelectTimeStepFreePublicXtra
 *-------------------------------------------------------------------------*/

void  SelectTimeStepFreePublicXtra()
{
   PFModule    *this_module   = ThisPFModule;
   PublicXtra  *public_xtra   = PFModulePublicXtra(this_module);

   Type0        *dummy0;
   Type1        *dummy1;

   if ( public_xtra )
   {
      switch((public_xtra -> type))
      {
      case 0:
      {
	 dummy0 = (Type0 *)(public_xtra -> data);
	 tfree(dummy0);
	 break;
      }
      case 1:
      {
	 dummy1 = (Type1 *)(public_xtra -> data);
	 tfree(dummy1);
	 break;
      }
      }

      tfree(public_xtra);
   }
}

/*--------------------------------------------------------------------------
 * SelectTimeStepSizeOfTempData
 *--------------------------------------------------------------------------*/

int  SelectTimeStepSizeOfTempData()
{
   return 0;
}
