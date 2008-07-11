/*  This routine provides the interface between KINSOL and ParFlow
    for function evaluations.  */
 
#include "parflow.h"


int KINSolFunctionEval(multispecies, fval, current_state)
N_Vector multispecies;
N_Vector fval;
void    *current_state;
{
   PFModule    *press_function_eval  = StateFuncPress(        ((State*)current_state) );
   PFModule    *temp_function_eval   = StateFuncTemp(         ((State*)current_state) );
   ProblemData *problem_data         = StateProblemData(      ((State*)current_state) );
   Vector      *old_pressure         = StateOldPressure(      ((State*)current_state) );
   Vector      *old_temperature      = StateOldTemperature(   ((State*)current_state) );
   Vector      *saturation           = StateSaturation(       ((State*)current_state) );
   Vector      *old_saturation       = StateOldSaturation(    ((State*)current_state) );
   Vector      *density              = StateDensity(          ((State*)current_state) );
   Vector      *old_density          = StateOldDensity(       ((State*)current_state) );
   Vector      *heat_capacity_water  = StateHeatCapacityWater(((State*)current_state) );
   Vector      *heat_capacity_rock   = StateHeatCapacityRock( ((State*)current_state) );
   Vector      *internal_energy      = StateInternalEnergy(   ((State*)current_state) );
   Vector      *viscosity            = StateViscosity(        ((State*)current_state) );
   double       dt                   = StateDt(               ((State*)current_state) );
   double       time                 = StateTime(             ((State*)current_state) );
   double       *outflow             = StateOutflow(          ((State*)current_state) );
   Vector       *evap_trans          = StateEvapTrans(        ((State*)current_state) );
   Vector       *clm_energy_source   = StateClmEnergySource(  ((State*)current_state) );
   Vector       *forc_t              = StateForcT(            ((State*)current_state) );
   Vector       *ovrl_bc_flx         = StateOvrlBcFlx(        ((State*)current_state) );
   Vector       *x_velocity          = StateXVelocity(        ((State*)current_state) );
   Vector       *y_velocity          = StateYVelocity(        ((State*)current_state) );
   Vector       *z_velocity          = StateZVelocity(        ((State*)current_state) );
 
   Vector       *pressure;
   Vector       *temperature;

   N_VectorContent_Parflow ms_content = NV_CONTENT_PF(multispecies);
   N_VectorContent_Parflow fval_content = NV_CONTENT_PF(fval);
   Vector *ms_specie;
   Vector *fval_specie;
   Vector *old_specie;
   int n, nspecies;
 
   int ip,ix,iy,iz,nx,ny,nz,nx_f,ny_f,nz_f;
   int i,j,k;
   Subvector *ms_sub, *fval_sub;
   double    *msp, *fvalp;
   double    ms[1][1][10], val[1][1][10];
   Grid *grid = VectorGrid(saturation);
   Subgrid *subgrid;
 
 
   nspecies = ms_content->num_species;
   pressure = ms_content->specie[0];
   temperature = ms_content->specie[1];
 
   for (n = 0; n < nspecies; n++)
   //for (n = 0; n < (nspecies-1); n++)
   {
     fval_specie = fval_content->specie[n];
     ms_specie = ms_content->specie[n];
 
     if (n == 0) {
       PFModuleInvoke(void, press_function_eval,
                      (pressure, fval_specie, problem_data, temperature, saturation, old_saturation,
                       density, old_density, viscosity, dt, time, old_pressure, outflow, evap_trans,
                       ovrl_bc_flx,x_velocity,y_velocity,z_velocity) );
 
     } else if (n == (nspecies-1)) {
       PFModuleInvoke(void, temp_function_eval,
                      (temperature, fval_specie, problem_data, pressure, old_pressure, saturation, old_saturation,
                       density, old_density, heat_capacity_water, heat_capacity_rock, viscosity, 
                       dt, time, old_temperature, evap_trans, clm_energy_source, forc_t, x_velocity, y_velocity, z_velocity) );
     }
 
 
      subgrid = GridSubgrid(grid, 0);
      ms_sub  = VectorSubvector(ms_specie, 0);
      fval_sub = VectorSubvector(fval_specie, 0);
 
      ix = SubgridIX(subgrid)-1;
      iy = SubgridIY(subgrid)-1;
      iz = SubgridIZ(subgrid)-1;
          
      nx = SubgridNX(subgrid)+2;
      ny = SubgridNY(subgrid)+2;
      nz = SubgridNZ(subgrid)+2;
       
      nx_f = SubvectorNX(fval_sub);
      ny_f = SubvectorNY(fval_sub);
      nz_f = SubvectorNZ(fval_sub);
       
      fvalp = SubvectorData(fval_sub);
      msp  = SubvectorData(ms_sub);
       
      ip  = SubvectorEltIndex(fval_sub,ix, iy, iz);
       
      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                ip, nx_f, ny_f, nz_f, 1, 1, 1, 
             {  
               //ms[i][j][k] = msp[ip];
               //val[i][j][k] = fvalp[ip];
               //printf("%d %d %d %30.20f %e\n",i,j,k,msp[ip],fvalp[ip]);
             });
   }
 
  return(0);
}

