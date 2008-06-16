/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.2 $
 *********************************************************************EHEADER*/
#include <stdlib.h>
#include "enlargebox.h"


Databox       *EnlargeBox(Databox *inbox,
			  int new_nx , int new_ny , int new_nz)
{
  Databox       *newbox;

  int             nx, ny, nz;
  double          x,  y,  z;
  double          dx, dy, dz;
  
  double         *new_ptr;
  double         *in_ptr;
  
  int             i, j, k;
  
  nx = DataboxNx(inbox);
  ny = DataboxNy(inbox);
  nz = DataboxNz(inbox);
  
  x  = DataboxX(inbox);
  y  = DataboxY(inbox);
  z  = DataboxZ(inbox);

  dx = DataboxDx(inbox);
  dy = DataboxDy(inbox);
  dz = DataboxDz(inbox);
  
  if(new_nx < nx){
     printf(" Error: new_nx must be greater than or equal to old size\n");
   }  

   if(new_ny < ny){
     printf(" Error: new_ny must be greater than or equal to old size\n");
   }  

   if(new_nz < nz){
     printf(" Error: new_nz must be greater than or equal to old size\n");
   }  

   if((newbox = NewDatabox(new_nx, new_ny, new_nz,
				x, y, z, 
				dx, dy, dz)) == NULL)
     return((Databox *)NULL);

   new_ptr = DataboxCoeffs(newbox);
   in_ptr = DataboxCoeffs(inbox);

   /* First just copy the old values into the new box */
   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
         for (i = 0; i < nx; i++)
         {
	    *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(inbox, i, j, k);
         }
      }
   }


   /* Copy the z plane  from the existing nz'th plane */
   for (k = nz; k < new_nz; k++)
      for (j = 0; j < ny; j++)
	 for (i = 0; i < nx; i++)
	    *DataboxCoeff(newbox, i, j, k) =    *DataboxCoeff(newbox, i, j, nz-1);

   /* Copy the y plane  from the existing ny'th plane */
   for (j = ny; j < new_ny; j++)
     for (k = 0; k < new_nz; k++)
       for (i = 0; i < nx; i++)
	 *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(newbox, i, ny-1, k);

   /* Copy the i planes from the existing nx'th plane */
   for (i = nx; i < new_nx; i++)
     for (j = 0; j < new_ny; j++)
       for (k = 0; k < new_nz; k++)
	 *DataboxCoeff(newbox, i, j, k) = *DataboxCoeff(newbox, nx-1, i, k);

   return newbox;
}
