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
 * Routines for generating line processes for the turning bands method.
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * LineProc
 *--------------------------------------------------------------------------*/

void  LineProc(Z, phi, theta, dzeta, izeta, nzeta, Kmax, dK)
double  *Z;
double   phi;
double   theta;
double   dzeta;
int      izeta;
int      nzeta;
double   Kmax;
double   dK;
{
   double   pi = acos(-1.0);

   int      M;
   double   dk;
   double   phij, kj, kj_fudge, S1, dW, zeta;

   int      i, j;


   /* compute M and dk */
   M  = (int)(Kmax / dK);
   dk = dK;

   /* initialize Z */
   for (i = 0; i < nzeta; i++)
      Z[i] = 0.0;

   /*-----------------------------------------------------------------------
    * compute Z
    *-----------------------------------------------------------------------*/

   for (j = 0; j < M; j++)
   {
      phij = 2.0*pi * Rand();
      kj   = dk * (j + 0.5);

      kj_fudge = kj + (dk/20.0) * (2.0*(Rand() - 0.5));

      S1 = (2.0*kj*kj) / (pi*(kj*kj+1.0)*(kj*kj+1.0));

      dW = 2.0*sqrt(S1*dk);

      for (i = 0; i < nzeta; i++)
      {
	 zeta = (izeta + i)*dzeta;
	 Z[i] += dW * cos(kj_fudge*zeta + phij);
      }
   }
}
