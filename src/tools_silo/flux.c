/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * CompFlux
 *
 * (C) 1995 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *-----------------------------------------------------------------------------
 * $Revision: 1.1.1.1 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "flux.h"

#if 0
#define Mean(a, b) ( 0.5*((a) + (b)) )
#define Mean(a, b) ( sqrt((a) * (b)) )
#endif
#define Mean(a, b)    ( ((a) + (b)) ? ( (2.0*(a)*(b))/((a) + (b)) ) : 0 )

/*-----------------------------------------------------------------------
 * Compute net cell flux from conductivity and hydraulic head
 *-----------------------------------------------------------------------*/

Databox       *CompFlux(k, h)
Databox       *k, *h;
{
   Databox        *flux;

   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;

   double         *fluxp, *kp, *hp;

   double          qxp, qxm, qyp, qym, qzp, qzm;
   int             cell,
                   cell_xm1, cell_xp1,
                   cell_ym1, cell_yp1,
                   cell_zm1, cell_zp1;
   int             ii, jj, kk;




   nx = DataboxNx(k);
   ny = DataboxNy(k);
   nz = DataboxNz(k);

   x  = DataboxX(k);
   y  = DataboxY(k);
   z  = DataboxZ(k);

   dx = DataboxDx(k);
   dy = DataboxDy(k);
   dz = DataboxDz(k);

#if 0      /* ADD LATER */
   if ((dx != DataboxDx(h)) ||
       (dy != DataboxDy(h)) ||
       (dz != DataboxDz(h)))
   {
      Error("Spacings are not compatible\n");
      return NULL;
   }
#endif

   if ((flux = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
      return((Databox *)NULL); 

   kp = DataboxCoeffs(k);
   hp = DataboxCoeffs(h);
   fluxp = DataboxCoeffs(flux);

   cell     = 0;
   cell_xm1 = cell - 1;
   cell_xp1 = cell + 1;
   cell_ym1 = cell - nx;
   cell_yp1 = cell + nx;
   cell_zm1 = cell - nx*ny;
   cell_zp1 = cell + nx*ny;

   kp += nx*ny;
   hp += nx*ny;
   fluxp += nx*ny;
   for (kk = 1; kk < (nz-1); kk++)
   {
      kp += nx;
      hp += nx;
      fluxp += nx;
      for (jj = 1; jj < (ny-1); jj++)
      {
         kp++;
         hp++;
         fluxp++;
         for (ii = 1; ii < (nx-1); ii++)
         {
            qxp = - Mean(kp[cell_xp1], kp[cell])*(hp[cell_xp1] - hp[cell])/dx;
            qxm = - Mean(kp[cell], kp[cell_xm1])*(hp[cell] - hp[cell_xm1])/dx;
            qyp = - Mean(kp[cell_yp1], kp[cell])*(hp[cell_yp1] - hp[cell])/dy;
            qym = - Mean(kp[cell], kp[cell_ym1])*(hp[cell] - hp[cell_ym1])/dy;
            qzp = - Mean(kp[cell_zp1], kp[cell])*(hp[cell_zp1] - hp[cell])/dz;
            qzm = - Mean(kp[cell], kp[cell_zm1])*(hp[cell] - hp[cell_zm1])/dz;

            fluxp[cell] = (qxp-qxm)*dy*dz + (qyp-qym)*dx*dz + (qzp-qzm)*dx*dy;

            kp++;
            hp++;
            fluxp++;
         }
         kp++;
         hp++;
         fluxp++;
      }
      kp += nx;
      hp += nx;
      fluxp += nx;
   }

   return flux;
}
