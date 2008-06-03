/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
 
#include "parflow.h"

/* Generic interface for matrix vector product */

/*  This routine provides the interface between KINSOL and ParFlow
    for richards' equation jacobian evaluations and matrix-vector multiplies.*/
 
int       KINSolMatVec(x, y, multispecies, recompute, current_state)
void     *current_state; 
N_Vector  x; 
N_Vector  y;
int      *recompute;
N_Vector  multispecies;
{ 
   N_VectorContent_Parflow content,xcont,ycont;
   Vector      *x_specie, *y_specie; 
   Vector      *specie; 
    
   PFModule    *richards_jacobian_eval    = StateJacEvalPressure(    ((State*)current_state));
   PFModule    *temperature_jacobian_eval = StateJacEvalTemperature( ((State*)current_state));
   Matrix      *J_press                   = StateJacPressure(        ((State*)current_state) );
   Matrix      *J_temp                    = StateJacTemperature(     ((State*)current_state) );
   Vector      *saturation                = StateSaturation(         ((State*)current_state) );
   Vector      *density                   = StateDensity(            ((State*)current_state) );
   Vector      *heat_capacity_water       = StateHeatCapacityWater(  ((State*)current_state) );
   Vector      *heat_capacity_rock        = StateHeatCapacityRock(   ((State*)current_state) );
   Vector      *viscosity                 = StateViscosity(          ((State*)current_state) );
   Vector      *x_velocity                = StateXVelocity(          ((State*)current_state) );
   Vector      *y_velocity                = StateYVelocity(          ((State*)current_state) );
   Vector      *z_velocity                = StateZVelocity(          ((State*)current_state) );
   ProblemData *problem_data              = StateProblemData(        ((State*)current_state) );
   double       dt                        = StateDt(                 ((State*)current_state) );
   double       time                      = StateTime(               ((State*)current_state) );
   int          n, nspecies;      
    
   Vector      *pressure;
   Vector      *temperature;

   content = NV_CONTENT_PF(multispecies);
   xcont = NV_CONTENT_PF(x); 
   ycont = NV_CONTENT_PF(y);
   nspecies = content->num_species;
 
   pressure = content->specie[0];
   temperature = content->specie[1];
   for (n = 0; n < nspecies; n++)
   {
     x_specie = xcont->specie[n];
     y_specie = ycont->specie[n];
 
     if ( *recompute )
     {
       if (n == 0) {
         PFModuleInvoke(void, richards_jacobian_eval,
                        (pressure, &J_press, temperature, saturation, density, viscosity, problem_data,
                        dt, time, 0 ));
       } else if (n == (nspecies-1)) {
         PFModuleInvoke(void, temperature_jacobian_eval,
                        (temperature, &J_temp, pressure, saturation, density, heat_capacity_water, heat_capacity_rock, 
                         x_velocity, y_velocity, z_velocity, problem_data, 
                         dt, time, 0 ));
       }
 
 
     }
 
     if (n == 0)
      Matvec(1.0, J_press, x_specie, 0.0, y_specie);
     else if (n == (nspecies-1))
      Matvec(1.0, J_temp, x_specie, 0.0, y_specie);
      
   }
 
   return(0);
}

