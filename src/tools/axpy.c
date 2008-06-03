/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.5 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Axpy
 *
 * (C) 1993 Regents of the University of California.
 *
 *-----------------------------------------------------------------------------
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "pftools.h"


/*-----------------------------------------------------------------------
 * Compute Y = alpha*X + Y
 *-----------------------------------------------------------------------*/

void       Axpy(double alpha, Databox *X,  Databox *Y)
{
   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;

   double         *xp, *yp;

   int             m, sx, sy, sz;
   

   nx = DataboxNx(X);
   ny = DataboxNy(X);
   nz = DataboxNz(X);

   x  = DataboxX(X);
   y  = DataboxY(X);
   z  = DataboxZ(X);

   dx = DataboxDx(X);
   dy = DataboxDy(X);
   dz = DataboxDz(X);

   xp  = DataboxCoeffs(X);
   yp  = DataboxCoeffs(Y);

   m = 0;
   sx = 1;
   sy = nx;
   sz = ny*nx;

   for (m = 0; m < (nx*ny*nz); m++)
   {
      yp[m] += alpha*xp[m];
   }

}


