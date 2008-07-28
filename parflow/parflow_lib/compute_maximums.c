/*BHEADER********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *******************************************************************EHEADER*/

/***************************************************************************
 *
 *
 ***************************************************************************/

#include "parflow.h"
#include "problem_eval.h"


/*-------------------------------------------------------------------------
 * compute_phase_maximum
 *-------------------------------------------------------------------------*/

double  ComputePhaseMaximum(phase_u_max,dx,phase_v_max,dy,phase_w_max,dz)
double  phase_u_max;
double  dx;
double  phase_v_max;
double  dy;
double  phase_w_max;
double  dz;
{
   double  maximum, tmp;

   /* X direction */
   maximum = phase_u_max/dx;

   /* Y direction */
   tmp = phase_v_max/dy;
   if ( tmp > maximum )
      maximum = tmp;

   /* Z direction */
   tmp = phase_w_max/dz;
   if ( tmp > maximum )
      maximum = tmp;

   return maximum;
}

/*-------------------------------------------------------------------------
 * compute_total_maximum
 *-------------------------------------------------------------------------*/

double  ComputeTotalMaximum(problem,eval_struct,s_lower,s_upper,
                            total_u_max,dx,
                            total_v_max,dy,
                            total_w_max,beta_max,dz)
Problem    *problem;
EvalStruct *eval_struct;
double      s_lower;
double      s_upper;
double      total_u_max;
double      dx;
double      total_v_max;
double      dy;
double      total_w_max;
double      beta_max;
double      dz;
{

   PFModule     *phase_density       = ProblemPhaseDensity(problem);

   double  a, b, den0, den1, dtmp, g;

   double  f_prime_max, h_prime_max;
   double  point, value, tmp, maximum;
   int     i;
   a = 1.0 / ProblemPhaseViscosity(problem,0);
   b = 1.0 / ProblemPhaseViscosity(problem,1);

   /* CSW  Hard-coded in an assumption here for constant density. 
    *      Use dtmp as dummy argument here. */
   PFModuleInvoke(void, phase_density, (0, NULL, NULL, &dtmp, &den0, CALCFCN));
   PFModuleInvoke(void, phase_density, (1, NULL, NULL, &dtmp, &den1, CALCFCN));

   g = -ProblemGravity(problem);

   /**************************************************************
    *                                                            *
    * Find the maximum value of f' over the interval, given that *
    *      the real roots of f'' have been previously found      *
    *                                                            *
    **************************************************************/
   f_prime_max = fabs( Fprime_OF_S(s_lower,a,b) );
   for(i = 0; i < EvalNumFPoints(eval_struct); i++)
   {
      point = EvalFPoint(eval_struct, i);
      value = EvalFValue(eval_struct, i);
      if ( (point > s_lower) && ( point < s_upper) )
      {
         tmp = fabs( value );
         if ( tmp > f_prime_max )
            f_prime_max = tmp;
      }
   }
   tmp = fabs( Fprime_OF_S(s_upper,a,b) );
   if ( tmp > f_prime_max )
      f_prime_max = tmp;

   /**************************************************************
    *                                                            *
    * Find the maximum value of h' over the interval, given that *
    *      the real roots of h'' have been previously found      *
    *                                                            *
    **************************************************************/
   h_prime_max = fabs( Hprime_OF_S(s_lower,a,b) );
   for(i = 0; i < EvalNumHPoints(eval_struct); i++)
   {
      point = EvalHPoint(eval_struct, i);
      value = EvalHValue(eval_struct, i);
      if ( (point > s_lower) && ( point < s_upper) )
      {
         tmp = fabs( value );
         if ( tmp > h_prime_max )
            h_prime_max = tmp;
      }
   }
   tmp = fabs( Hprime_OF_S(s_upper,a,b) );
   if ( tmp > h_prime_max )
      h_prime_max = tmp;

   /**************************************************************
    *                                                            *
    *       Find the maximum for use in the dt computation       *
    *                                                            *
    **************************************************************/
   /* X direction */
   maximum = f_prime_max * (total_u_max/dx);

   /* Y direction */
   tmp = f_prime_max * (total_v_max/dy);
   if ( tmp > maximum )
      maximum = tmp;

   /* Z direction */
   tmp = f_prime_max * (total_w_max/dz) 
        + h_prime_max * fabs(g * (den0 - den1)) * (beta_max/dz);
   if ( tmp > maximum )
      maximum = tmp;

   return maximum;
}
