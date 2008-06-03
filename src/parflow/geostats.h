/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Geostatistics class structures
 *
 *****************************************************************************/

#ifndef _GEOSTATS_HEADER
#define _GEOSTATS_HEADER

/* Structure for holding (scattered) conditioning data */

typedef struct
{
   int	  nc;		/* number of points in arrays */
   double *x,*y,*z;	/* location in real coordinates */
   double *v;		/* data value */

} RFCondData;

typedef struct
{
   double mean, sigma;
   double lambdaX, lambdaY, lambdaZ;
   int	  lognormal;

} Statistics;


#endif
