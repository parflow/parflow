/*  This routine provides the interface between KINSOL and ParFlow
 *      for function evaluations.  */

#include "parflow.h"


void     KINSolFunctionEval(
int      size,
N_Vector speciesNVector,
N_Vector fvaln,
void    *current_state)
{
   PFModule  *nl_function_eval = StateFunc(        ((State*)current_state) );
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
   (void) size;

   pressure = NV_CONTENT_PF(speciesNVector)->dims[0];
   fval = NV_CONTENT_PF(fvaln)->dims[0];

   PFModuleInvokeType(NlFunctionEvalInvoke, nl_function_eval,
                  (pressure, fval, problem_data, saturation, old_saturation,
                   density, old_density, dt, time, old_pressure, evap_trans,
                   ovrl_bc_flx) );
#ifdef FGTest
   pressure = NV_CONTENT_PF(speciesNVector)->dims[1];
   fval = NV_CONTENT_PF(fvaln)->dims[1];

   PFModuleInvokeType(NlFunctionEvalInvoke, nl_function_eval,
                  (pressure, fval, problem_data, saturation2, old_saturation2,
                   density, old_density, dt, time, old_pressure2, evap_trans,
                   ovrl_bc_flx) );
#endif
   return;
}

