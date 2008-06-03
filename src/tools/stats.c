/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Print various statistics
 *
 * (C) 1995 Regents of the University of California.
 *
 * $Revision: 1.4 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "stats.h"

/*-----------------------------------------------------------------------
 * Print various statistics
 *-----------------------------------------------------------------------*/

void        Stats(databox, min, max, mean, sum, variance, stdev)
Databox    *databox;
double     *min;
double     *max;
double     *mean;
double     *sum;
double     *variance;
double     *stdev;
{
   double  *dp;

   double   dtmp;

   int      i, j, k;
   int      nx, ny, nz, n;


   nx = DataboxNx(databox);
   ny = DataboxNy(databox);
   nz = DataboxNz(databox);
   n  = nx * ny * nz;

   /*---------------------------------------------------
    * Compute min, max, mean
    *---------------------------------------------------*/

   dp = DataboxCoeffs(databox);

   *min  = *dp;
   *max  = *dp;
   *mean = 0.0;
   *sum  = 0.0;

   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
	    if (*dp < *min)
	       *min = *dp;

	    if (*dp > *max)
	       *max = *dp;
    
	    *sum += *dp;
    
            dp++;
         }
      }
   }

   *mean = *sum / n;

   /*---------------------------------------------------
    * Compute variance, standard deviation
    *---------------------------------------------------*/

   dp = DataboxCoeffs(databox);

   *variance = 0.0;

   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
	    dtmp = (*dp - *mean);
	    *variance += dtmp*dtmp;
    
            dp++;
         }
      }
   }

   *variance /= n;
   *stdev = sqrt(*variance);

}
