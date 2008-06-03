/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

typedef struct
{
   PFModule    *press_function_eval;
   PFModule    *temp_function_eval;
   PFModule    *richards_jacobian_eval;
   PFModule    *temperature_jacobian_eval;
   PFModule    *precond_pressure;
   PFModule    *precond_temperature;

   ProblemData *problem_data;

   Matrix      *jacobian_matrix_press;
   Matrix      *jacobian_matrix_temp;

   Vector      *pressure;
   Vector      *old_pressure;
   Vector      *temperature;
   Vector      *old_temperature;
   Vector      *density;
   Vector      *heat_capacity_water;
   Vector      *heat_capacity_rock;
   Vector      *internal_energy;
   Vector      *old_density;
   Vector      *viscosity;
   Vector      *old_viscosity;
   Vector      *saturation;
   Vector      *old_saturation;

   double       dt;
   double       time;
   double       *outflow; /*sk*/
   
   Vector       *evap_trans; /*sk*/
   Vector       *clm_energy_source; /*sk*/
   Vector       *forc_t; /*sk*/
   Vector       *ovrl_bc_flx; /*sk*/
   
   Vector       *x_velocity, *y_velocity, *z_velocity;
} State;

/*--------------------------------------------------------------------------
 * Accessor macros: State
 *--------------------------------------------------------------------------*/

#define StateFuncPress(state)          ((state)->press_function_eval)
#define StateFuncTemp(state)           ((state)->temp_function_eval)
#define StateProblemData(state)        ((state)->problem_data)
#define StateOldPressure(state)        ((state)->old_pressure)
#define StateOldTemperature(state)     ((state)->old_temperature)
#define StateOldSaturation(state)      ((state)->old_saturation)
#define StateDensity(state)            ((state)->density)
#define StateOldDensity(state)         ((state)->old_density)
#define StateHeatCapacityWater(state)  ((state)->heat_capacity_water)
#define StateHeatCapacityRock(state)   ((state)->heat_capacity_rock)
#define StateInternalEnergy(state)     ((state)->internal_energy)
#define StateViscosity(state)          ((state)->viscosity)
#define StateOldViscosity(state)       ((state)->viscosity)
#define StateSaturation(state)         ((state)->saturation)
#define StateDt(state)                 ((state)->dt)
#define StateTime(state)               ((state)->time)
#define StateJacEvalPressure(state)    ((state)->richards_jacobian_eval)
#define StateJacEvalTemperature(state) ((state)->temperature_jacobian_eval)
#define StateJacPressure(state)        ((state)->jacobian_matrix_press)
#define StateJacTemperature(state)     ((state)->jacobian_matrix_temp)
#define StatePrecondPressure(state)    ((state)->precond_pressure)
#define StatePrecondTemperature(state) ((state)->precond_temperature)
#define StateOutflow(state)            ((state)->outflow) /*sk*/
#define StateEvapTrans(state)          ((state)->evap_trans) /*sk*/
#define StateClmEnergySource(state)    ((state)->clm_energy_source) /*sk*/
#define StateForcT(state)              ((state)->forc_t) /*sk*/
#define StateOvrlBcFlx(state)          ((state)->ovrl_bc_flx) /*sk*/
#define StateXVelocity(state)          ((state)->x_velocity) /*sk*/
#define StateYVelocity(state)          ((state)->y_velocity) /*sk*/
#define StateZVelocity(state)          ((state)->z_velocity) /*sk*/
