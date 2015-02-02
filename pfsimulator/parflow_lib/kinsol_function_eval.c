/*  This routine provides the interface between KINSOL and ParFlow
 *      for function evaluations.  */

#include "parflow.h"


void     KINSolFunctionEval(
int      size,
N_Vector speciesNVector,
N_Vector fvaln,
void    *current_state)
{
   PFModule  *press_function_eval = StatePressFunc(        ((State*)current_state) );
   ProblemData *problem_data   = StateProblemData( ((State*)current_state) );
   Vector      *old_pressure   = StateOldPressure(((State*)current_state) );
   Vector      *saturation     = StateSaturation(  ((State*)current_state) );
   Vector      *old_saturation = StateOldSaturation(((State*)current_state) );
   Vector      *density        = StateDensity(     ((State*)current_state) );
   Vector      *old_density    = StateOldDensity(  ((State*)current_state) );


#ifdef FGTest
   Vector      *old_pressure2   = StateOldPressure2(((State*)current_state) );
   Vector      *saturation2     = StateSaturation2(  ((State*)current_state) );
   Vector      *old_saturation2 = StateOldSaturation2(((State*)current_state) );
#endif

   double       dt             = StateDt(          ((State*)current_state) );
   double       time           = StateTime(        ((State*)current_state) );
   Vector       *evap_trans    = StateEvapTrans(   ((State*)current_state) );
   Vector       *ovrl_bc_flx   = StateOvrlBcFlx(   ((State*)current_state) );
   Vector       *pressure,*fval;
#ifdef withTemperature
PFModule  *temp_function_eval = StateFuncTemp(        ((State*)current_state) );

   Vector       *temperature;
   Vector       *heat_capacity_water  = StateHeatCapacityWater(((State*)current_state) );
   Vector       *heat_capacity_rock   = StateHeatCapacityRock(((State*)current_state) );
   Vector       *viscosity		= StateViscosity(((State*)current_state) );
   Vector       *old_temperature	= StateOldTemperature(((State*)current_state) );
   Vector       *clm_energy_source	= StateClmEnergySource(((State*)current_state) );
   Vector       *forc_t 		= StateForcT(((State*)current_state) );
   Vector       *x_velocity		= StateXVelocity(((State*)current_state) );
   Vector       *y_velocity		= StateYVelocity(((State*)current_state) );
   Vector       *z_velocity		= StateZVelocity(((State*)current_state) );
#endif

   (void) size;

   pressure = NV_CONTENT_PF(speciesNVector)->dims[0];

#ifdef withTemperature
   temperature = NV_CONTENT_PF(speciesNVector)->dims[1];
#endif

   fval = NV_CONTENT_PF(fvaln)->dims[0];
   PFModuleInvokeType(PressFunctionEvalInvoke, press_function_eval,
                  (pressure, fval, problem_data, saturation, old_saturation,
                   density, old_density, dt, time, old_pressure, evap_trans,
                   ovrl_bc_flx) );


#ifdef withTemperature
   fval = NV_CONTENT_PF(fvaln)->dims[1];

   PFModuleInvokeType(TempFunctionEvalInvoke, temp_function_eval,
                      (temperature, fval, problem_data, pressure, old_pressure, saturation, old_saturation,
                       density, old_density, heat_capacity_water, heat_capacity_rock, viscosity,
                       dt, time, old_temperature, evap_trans, clm_energy_source, forc_t, x_velocity, y_velocity, z_velocity) );

#endif



#ifdef FGTest
   pressure = NV_CONTENT_PF(speciesNVector)->dims[1];
   fval = NV_CONTENT_PF(fvaln)->dims[1];

   PFModuleInvokeType(PressFunctionEvalInvoke, press_function_eval,
                  (pressure, fval, problem_data, saturation2, old_saturation2,
                   density, old_density, dt, time, old_pressure2, evap_trans,
                   ovrl_bc_flx) );
#endif
   return;
}

