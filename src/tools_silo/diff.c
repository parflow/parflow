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
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *-----------------------------------------------------------------------------
 * $Revision: 1.1.1.1 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "diff.h"

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef max
#define max(a,b)  ((a) > (b) ? (a) : (b))
#endif


/*-----------------------------------------------------------------------
 * SigDiff:
 *   If (m >= 0), this routine prints the absolute differences
 *   of databox elements differing in more than m significant digits.
 *   If (m < 0), this routine prints the minimum number of agreeing
 *   significant digits.
 *
 *   We use the following to determine if two numbers, x1 and x2, have
 *   at least m significant digits:
 *
 *     xM = max(|x1|, |x2|)
 *
 *     if ( xM > a0 )
 *        "x1 and x2 have m significant digits"
 *     else
 *     {
 *       y = | x1 - x2 | / xM
 *
 *       if (y <= 0.5*10^{-m})
 *          "x1 and x2 have m significant digits"
 *     }
 *
 *   The number a0 is "absolute zero".  If xM is less than 0, it is
 *   treated as if it were exactly 0.
 *-----------------------------------------------------------------------*/

void      SigDiff(v1, v2, m, absolute_zero, result, fp)
Databox  *v1;
Databox  *v2;
int       m;
double    absolute_zero;
Tcl_DString *result;
FILE     *fp;
{
   double         *v1_p, *v2_p;

   double          sig_dig_rhs;
   double          adiff, amax, sdiff;
   double          max_adiff, max_sdiff;

   int             i,  j,  k;
   int             mi, mj, mk;
   int             nx, ny, nz;

   int             sig_digs;
   int             m_sig_digs, m_sig_digs_everywhere;

   /*-----------------------------------------------------------------------
    * check that dimensions are the same
    *-----------------------------------------------------------------------*/

   nx = DataboxNx(v1);
   ny = DataboxNy(v1);
   nz = DataboxNz(v1);

   /*-----------------------------------------------------------------------
    * diff the values and print the results
    *-----------------------------------------------------------------------*/

   if (m >= 0)
      sig_dig_rhs = 0.5 / pow(10.0, ((double) m));
   else
      sig_dig_rhs = 0.0;

   v1_p = DataboxCoeffs(v1);
   v2_p = DataboxCoeffs(v2);

   m_sig_digs_everywhere = TRUE;
   max_sdiff  = 0.0;
   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
	 for (i = 0; i < nx; i++)
	 {
	    adiff = fabs((*v1_p) - (*v2_p));
	    amax  = max(fabs(*v1_p), fabs(*v2_p));

	    m_sig_digs = TRUE;
	    if (amax > absolute_zero)
	    {
	       sdiff = adiff / amax;
	       if ( sdiff > sig_dig_rhs )
		  m_sig_digs = FALSE;
	    }

	    if (!m_sig_digs)
	    {
	       if (m >= 0)
               {

	          fprintf(fp, "Absolute difference at (% 3d, % 3d, % 3d): %e\n",
	             i, j, k, adiff);
               }
                  
	       if (sdiff > max_sdiff)
	       {
		  max_sdiff = sdiff;
		  max_adiff = adiff;
		  mi = i;
		  mj = j;
		  mk = k;
	       }

	       m_sig_digs_everywhere = FALSE;
	    }

	    v1_p++;
	    v2_p++;
	 }
      }
   }

   if (!m_sig_digs_everywhere)
   {
      /* compute min number of sig digs */
      sig_digs = 0;
      sdiff = max_sdiff;
      while (sdiff <= 0.5e-01)
      {
	 sdiff *= 10.0;
	 sig_digs++;
      }

      fprintf(fp, "Minimum significant digits at (% 3d, %3d, %3d) = %2d\n",
         mi, mj, mk, sig_digs);
      fprintf(fp, "Maximum absolute difference = %e\n", max_adiff);
   }

}


void         MSigDiff(v1, v2, m, absolute_zero, result)
Databox     *v1;
Databox     *v2;
int          m;
double       absolute_zero;
Tcl_DString *result;
{
   double         *v1_p, *v2_p;

   double          sig_dig_rhs;
   double          adiff, amax, sdiff;
   double          max_adiff, max_sdiff;

   int             i,  j,  k;
   int             mi, mj, mk;
   int             nx, ny, nz;

   int             sig_digs;
   int             m_sig_digs, m_sig_digs_everywhere;

   char            coord[256], num[256];


   Tcl_DStringInit(result);

   /*-----------------------------------------------------------------------
    * check that dimensions are the same
    *-----------------------------------------------------------------------*/

   nx = DataboxNx(v1);
   ny = DataboxNy(v1);
   nz = DataboxNz(v1);

   /*-----------------------------------------------------------------------
    * diff the values and print the results
    *-----------------------------------------------------------------------*/

   if (m >= 0)
      sig_dig_rhs = 0.5 / pow(10.0, ((double) m));
   else
      sig_dig_rhs = 0.0;

   v1_p = DataboxCoeffs(v1);
   v2_p = DataboxCoeffs(v2);

   m_sig_digs_everywhere = TRUE;
   max_sdiff  = 0.0;
   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
	 for (i = 0; i < nx; i++)
	 {
	    adiff = fabs((*v1_p) - (*v2_p));
	    amax  = max(fabs(*v1_p), fabs(*v2_p));

	    if ( max_adiff < adiff )
	       max_adiff = adiff;

	    m_sig_digs = TRUE;
	    if (amax > absolute_zero)
	    {
	       sdiff = adiff / amax;
	       if ( sdiff > sig_dig_rhs )
		  m_sig_digs = FALSE;
	    }

	    if (!m_sig_digs)
	    {
                  
	       if (sdiff > max_sdiff)
	       {
		  max_sdiff = sdiff;

		  mi = i;
		  mj = j;
		  mk = k;
	       }

	       m_sig_digs_everywhere = FALSE;
	    }

	    v1_p++;
	    v2_p++;
	 }
      }
   }

   if (!m_sig_digs_everywhere)
   {
      /* compute min number of sig digs */
      sig_digs = 0;
      sdiff = max_sdiff;
      while (sdiff <= 0.5e-01)
      {
	 sdiff *= 10.0;
	 sig_digs++;
      }

      /* Create a Tcl list of the form: {{mi mj mk sig_digs} max_adiff} */
      /* and append it to the result string.                            */

      sprintf(coord, "%d %d %d %d", mi, mj, mk, sig_digs);
      Tcl_DStringAppendElement(result, coord);
      sprintf(num, "%e", max_adiff);
      Tcl_DStringAppendElement(result, num);
         
   }

}


double    DiffElt(v1, v2, i, j, k, m, absolute_zero)
Databox  *v1;
Databox  *v2;
int       i, j, k;
int       m;
double    absolute_zero;
{
   double         *v1_p, *v2_p;

   double          sig_dig_rhs;
   double          adiff, amax, sdiff;
   double          max_sdiff;
   int             m_sig_digs;


   /*-----------------------------------------------------------------------
    * diff the values and print the results
    *-----------------------------------------------------------------------*/

   sig_dig_rhs = 0.5 / pow(10.0, ((double) m));

   v1_p = DataboxCoeff(v1, i, j, k);
   v2_p = DataboxCoeff(v2, i, j, k);

   max_sdiff  = 0.0;

   adiff = fabs((*v1_p) - (*v2_p));
   amax  = max(fabs(*v1_p), fabs(*v2_p));

   m_sig_digs = TRUE;
   if (amax > absolute_zero)
   {
      sdiff = adiff / amax;
      if ( sdiff > sig_dig_rhs )
         m_sig_digs = FALSE;
   }

   if (m_sig_digs)
   {

      return -1;

   }

   return adiff;
}
