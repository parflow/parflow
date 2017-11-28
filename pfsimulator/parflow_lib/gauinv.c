/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 */
/*                                                                      %
 */
/* Copyright (C) 1992 Stanford Center for Reservoir Forecasting.  All   %
 */
/* rights reserved.  Distributed with: C.V. Deutsch and A.G. Journel.   %
 */
/* ``GSLIB: Geostatistical Software Library and User's Guide,'' Oxford  %
 */
/* University Press, New York, 1992.                                    %
 */
/*                                                                      %
 */
/* The programs in GSLIB are distributed in the hope that they will be  %
 */
/* useful, but WITHOUT ANY WARRANTY.  No author or distributor accepts  %
 */
/* responsibility to anyone for the consequences of using them or for   %
 */
/* whether they serve any particular purpose or work at all, unless he  %
 */
/* says so in writing.  Everyone is granted permission to copy, modify  %
 */
/* and redistribute the programs in GSLIB, but only under the condition %
 */
/* that this notice and the above copyright notice remain intact.       %
 */
/*                                                                      %
 */
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 */
/* -----------------------------------------------------------------------
 */

#include <math.h>

/* Subroutine */ int gauinv_(
                             double *p,
                             double *xp,
                             int *   ierr)
{
  /* Initialized data */

  static double lim = 1e-20;
  static double p0 = -.322232431088;
  static double p1 = -1.;
  static double p2 = -.342242088547;
  static double p3 = -.0204231210245;
  static double p4 = -4.53642210148e-5;
  static double q0 = .099348462606;
  static double q1 = .588581570495;
  static double q2 = .531103462366;
  static double q3 = .10353775285;
  static double q4 = .0038560700634;

  /* Local variables */
  static double y, pp;

/* -----------------------------------------------------------------------
 */

/* Computes the inverse of the standard normal cumulative distribution */
/* function with a numerical approximation from : Statistical Computing,
 */
/* by W.J. Kennedy, Jr. and James E. Gentle, 1980, p. 95. */



/* INPUT/OUTPUT: */

/*   p    = double precision cumulative probability value: dble(psingle)
 */
/*   xp   = G^-1 (p) in single precision */
/*   ierr = 1 - then error situation (p out of range), 0 - OK */


/* -----------------------------------------------------------------------
 */

/* Coefficients of approximation: */


/* Initialize: */

  *ierr = 1;
  *xp = (float)0.;
  pp = *p;
  if (*p > (float).5)
  {
    pp = 1 - pp;
  }
  if (*p < lim)
  {
    return 0;
  }
  *ierr = 0;
  if (*p == (float).5)
  {
    return 0;
  }

/* Approximate the function: */

  y = sqrt(log((float)1. / (pp * pp)));
  *xp = y + ((((y * p4 + p3) * y + p2) * y + p1) * y + p0) / ((((y * q4 +
                                                                 q3) * y + q2) * y + q1) * y + q0);
  if (*p == pp)
  {
    *xp = -(double)(*xp);
  }

/* Return with G^-1(p): */

  return 0;
} /* gauinv_ */



