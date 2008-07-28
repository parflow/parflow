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
   PFModule    *nl_function_eval;
   PFModule    *richards_jacobian_eval;
   PFModule    *precond;

   ProblemData *problem_data;

   Matrix      *jacobian_matrix;

   Vector      *old_density;
   Vector      *old_saturation;
   Vector      *old_pressure;
   Vector      *density;
   Vector      *saturation;

   double       dt;
   double       time;
   double       *outflow; /*sk*/
   
   Vector       *evap_trans; /*sk*/
   Vector       *ovrl_bc_flx; /*sk*/
   
} State;



/*--------------------------------------------------------------------------
 * Accessor macros: State
 *--------------------------------------------------------------------------*/

#define StateFunc(state)          ((state)->nl_function_eval)
#define StateProblemData(state)   ((state)->problem_data)
#define StateOldDensity(state)    ((state)->old_density)
#define StateOldPressure(state)   ((state)->old_pressure)
#define StateOldSaturation(state) ((state)->old_saturation)
#define StateDensity(state)       ((state)->density)
#define StateSaturation(state)    ((state)->saturation)
#define StateDt(state)            ((state)->dt)
#define StateTime(state)          ((state)->time)
#define StateJacEval(state)       ((state)->richards_jacobian_eval)
#define StateJac(state)           ((state)->jacobian_matrix)
#define StatePrecond(state)       ((state)->precond)
#define StateOutflow(state)       ((state)->outflow) /*sk*/
#define StateEvapTrans(state)     ((state)->evap_trans) /*sk*/
#define StateOvrlBcFlx(state)     ((state)->ovrl_bc_flx) /*sk*/
