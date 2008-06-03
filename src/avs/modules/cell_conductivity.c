/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * CellConductivity
 *
 * AVS module for creating a cell-centered conductivity field.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * CellConductivity
 *--------------------------------------------------------------------------*/

CellConductivity()
{
   int CellConductivity_compute();
   int p;


   AVSset_module_name("cell conductivity", MODULE_FILTER);

   AVScreate_input_port("Hydraulic Conductivity",
			"field 3D scalar 3-coord uniform float",
			REQUIRED);

   AVScreate_output_port("Field Output",
			 "field 3D scalar 3-coord uniform float");

   AVSset_compute_proc(CellConductivity_compute);
}

	
/*--------------------------------------------------------------------------
 * CellConductivity_compute
 *--------------------------------------------------------------------------*/

CellConductivity_compute(k, ck)
AVSfield_float  *k;
AVSfield_float **ck;
{
   int             nx, ny, nz;
   float           dx, dy, dz;

   float         *kp, *ckp;

   int             m1, m2, m3, m4, m5, m6, m7, m8;
   int             ii, jj, kk;

   int             dims[3];


   /* free old memory */
   if (*ck) 
      AVSfield_free((AVSfield *) *ck);

   /*-----------------------------------------------------------------------
    * get parameters
    *-----------------------------------------------------------------------*/

   nx = k -> dimensions[0];
   ny = k -> dimensions[1];
   nz = k -> dimensions[2];

   /* assuming equally spaced in each direction */
   dx = ((k -> max_extent[0]) - (k -> min_extent[0])) / (nx - 1);
   dy = ((k -> max_extent[1]) - (k -> min_extent[1])) / (ny - 1);
   dz = ((k -> max_extent[2]) - (k -> min_extent[2])) / (nz - 1);

   /*-----------------------------------------------------------------------
    * compute the cell conductivity field
    *-----------------------------------------------------------------------*/

   /* create the new AVSfield structure */
   dims[0] = (nx-1);
   dims[1] = (ny-1);
   dims[2] = (nz-1);
   *ck = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);

#if 0
   /* setup min and max extent */
   (*ck) -> min_extent[0] = 0.0;
   (*ck) -> max_extent[0] = (nx-2)*dx;
   (*ck) -> min_extent[1] = 0.0;
   (*ck) -> max_extent[1] = (ny-2)*dy;
   (*ck) -> min_extent[2] = 0.0;
   (*ck) -> max_extent[2] = (nz-2)*dz;

   /* setup coords (same and min and max extents) */
   (*ck) -> points[0] = 0.0;
   (*ck) -> points[1] = (nx-2)*dx;
   (*ck) -> points[2] = 0.0;
   (*ck) -> points[3] = (ny-2)*dy;
   (*ck) -> points[4] = 0.0;
   (*ck) -> points[5] = (nz-2)*dz;
#endif

   kp = k -> data;
   ckp = (*ck) -> data;

   m1 = 0;
   m2 = m1 + 1;
   m3 = m1 + nx;
   m4 = m3 + 1;
   m5 = m1 + ny*nx;
   m6 = m5 + 1;
   m7 = m5 + nx;
   m8 = m7 + 1;

   for (kk = 0; kk < (nz-1); kk++)
   {
      for (jj = 0; jj < (ny-1); jj++)
      {
	 for (ii = 0; ii < (nx-1); ii++)
	 {
	    *(ckp++) = (kp[m1] + kp[m2] + kp[m3] + kp[m4] +
			kp[m5] + kp[m6] + kp[m7] + kp[m8]) / 8.0;
	    
	    kp++;
	 }
	 kp++;
      }
      kp += nx;
   }

   return(1);
}
