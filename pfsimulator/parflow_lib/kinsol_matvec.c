/*BHEADER**********************************************************************
 *  * (c) 1997   The Regents of the University of California
 *   *
 *    * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 *     * notice, contact person, and disclaimer.
 *      *
 *       * $Revision: 1.1.1.1 $
 *        *********************************************************************EHEADER*/

#include "parflow.h"

/* Generic interface for matrix vector product */


/*  This routine provides the interface between KINSOL and ParFlow
 *      for richards' equation jacobian evaluations and matrix-vector multiplies.*/

int       KINSolMatVec(
void     *current_state,
N_Vector  xn,
N_Vector  yn,
int      *recompute,
N_Vector  speciesNVector)
{
   PFModule    *richards_jacobian_eval = StateJacEval(((State*)current_state));
   Matrix      *J                = StateJac(         ((State*)current_state) );
   Matrix      *JC                = StateJacC(         ((State*)current_state) );
   Vector      *saturation       = StateSaturation(  ((State*)current_state) );
   Vector      *density          = StateDensity(     ((State*)current_state) );

#ifdef withTemperature
   PFModule    *temperature_jacobian_eval = StateJacEvalTemperature(((State*)current_state));

   Vector      *temperature; 
   Vector      *heat_capacity_water       = StateHeatCapacityWater(  ((State*)current_state) );
   Vector      *heat_capacity_rock       = StateHeatCapacityRock(  ((State*)current_state) );
   Vector      *x_velocity       	= StateXVelocity(  ((State*)current_state) );
   Vector      *y_velocity              = StateYVelocity(  ((State*)current_state) );
   Vector      *z_velocity              = StateZVelocity(  ((State*)current_state) );
#endif

#ifdef FGTest
   Vector      *saturation2       = StateSaturation2(  ((State*)current_state) );
#endif
   ProblemData *problem_data     = StateProblemData( ((State*)current_state) );
   double       dt               = StateDt(          ((State*)current_state) );
   double       time             = StateTime(        ((State*)current_state) );

   Vector      *pressure;
   Vector      *x, *y;


   pressure = NV_CONTENT_PF(speciesNVector)->dims[0];

   x = NV_CONTENT_PF(xn)->dims[0];
   y = NV_CONTENT_PF(yn)->dims[0];

   InitVector(y, 0.0);

   if ( *recompute )
   {
      PFModuleInvokeType(RichardsJacobianEvalInvoke, richards_jacobian_eval,
      (pressure, &J, &JC, saturation, density, problem_data,
      dt, time, 0));
   }


   if(JC == NULL)
     Matvec(1.0, J, x, 0.0, y);
   else
     MatvecSubMat(current_state, 1.0, J, JC, x, 0.0, y);

#ifdef withTemperature
   temperature = NV_CONTENT_PF(speciesNVector)->dims[1];

   x = NV_CONTENT_PF(xn)->dims[1];
   y = NV_CONTENT_PF(yn)->dims[1];

   InitVector(y, 0.0);

   if ( *recompute )
   {
      PFModuleInvokeType(TemperatureJacobianEvalInvoke, temperature_jacobian_eval,
      (temperature,&J , pressure, saturation, density, heat_capacity_water, heat_capacity_rock, x_velocity, y_velocity, z_velocity ,problem_data,
      dt, time, 0));

   }

   // if(JC == NULL)
     Matvec(1.0, J, x, 0.0, y);
   //else
   //  MatvecSubMat(current_state, 1.0, J, JC, x, 0.0, y);
#endif

#ifdef FGTest
   pressure = NV_CONTENT_PF(speciesNVector)->dims[1];

   x = NV_CONTENT_PF(xn)->dims[1];
   y = NV_CONTENT_PF(yn)->dims[1];

   InitVector(y, 0.0);

   if ( *recompute )
   {
      PFModuleInvokeType(RichardsJacobianEvalInvoke, richards_jacobian_eval,
      (pressure, &J, &JC, saturation2, density, problem_data,
      dt, time, 0));
   }

    if(JC == NULL)
     Matvec(1.0, J, x, 0.0, y);
   else
     MatvecSubMat(current_state, 1.0, J, JC, x, 0.0, y);
#endif

   return(0);
}

