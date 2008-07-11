/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * HHead, PHead
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: head.c,v $
 Revision 1.1.1.1  2006/02/14 23:05:52  kollet
 CLM.PF_1.0

 Revision 1.1.1.1  2006/02/14 18:51:22  kollet
 CLM.PF_1.0

 Revision 1.5  1996/08/10 06:35:12  mccombjr
 *** empty log message ***

 * Revision 1.4  1996/04/25  01:05:50  falgout
 * Added general BC capability.
 *
 * Revision 1.3  1995/12/21  00:56:38  steve
 * Added copyright
 *
 * Revision 1.2  1995/06/27  21:36:13  falgout
 * Added (X, Y, Z) coordinates to databox structure.
 *
 * Revision 1.1  1993/08/20  21:22:32  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "head.h"


/*-----------------------------------------------------------------------
 * compute the hydraulic head from the pressure head
 *-----------------------------------------------------------------------*/

Databox        *HHead(h, grid_type)
Databox        *h;
GridType        grid_type;
{
   Databox        *v;

   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;

   double         *hp, *vp;

   int             ji, k;

   double          zz, dz2;


   nx = DataboxNx(h);
   ny = DataboxNy(h);
   nz = DataboxNz(h);

   x  = DataboxX(h);
   y  = DataboxY(h);
   z  = DataboxZ(h);

   dx = DataboxDx(h);
   dy = DataboxDy(h);
   dz = DataboxDz(h);

   switch(grid_type)
   {
   case vertex: /* vertex centered */
      dz2 = 0.0;
      break;
   case cell:   /* cell centered */
      dz2 = dz/2;
      break;
   }

   if ((v = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
      return ((Databox *)NULL); 

   hp = DataboxCoeffs(h);
   vp = DataboxCoeffs(v);

   for (k = 0; k < nz; k++)
   {
      zz = z + ((double) k) * dz + dz2;
      for (ji = 0; ji < ny*nx; ji++)
	 *(vp++) = *(hp++) + zz;
   }

   return v;
}


/*-----------------------------------------------------------------------
 * compute the pressure from the hydraulic head
 *-----------------------------------------------------------------------*/

Databox        *PHead(h, grid_type)
Databox        *h;
GridType       grid_type;
{
   Databox        *v;

   int             nx, ny, nz;
   double          x,  y,  z;
   double          dx, dy, dz;

   double         *hp, *vp;

   int             ji, k;

   double          zz, dz2;


   nx = DataboxNx(h);
   ny = DataboxNy(h);
   nz = DataboxNz(h);

   x  = DataboxX(h);
   y  = DataboxY(h);
   z  = DataboxZ(h);

   dx = DataboxDx(h);
   dy = DataboxDy(h);
   dz = DataboxDz(h);

   switch(grid_type)
   {
   case vertex: /* vertex centered */
      dz2 = 0.0;
      break;
   case cell:   /* cell centered */
      dz2 = dz/2;
      break;
   }

   if ((v = NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)) == NULL)
      return((Databox *)NULL);

   hp = DataboxCoeffs(h);
   vp = DataboxCoeffs(v);

   for (k = 0; k < nz; k++)
   {
      zz = z + ((double) k) * dz + dz2;
      for (ji = 0; ji < ny*nx; ji++)
	 *(vp++) = *(hp++) - zz;
   }

   return v;
}


