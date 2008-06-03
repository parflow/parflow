/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * CellVelocity
 *
 * AVS module for computing a cell-centered velocity field.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * CellVelocity
 *--------------------------------------------------------------------------*/

CellVelocity()
{
   int CellVelocity_compute();
   int p;


   AVSset_module_name("cell velocity", MODULE_FILTER);

   AVScreate_input_port("Hydraulic Conductivity",
			"field 3D scalar 3-coord uniform float",
			REQUIRED);
   AVScreate_input_port("Hydraulic Head",
			"field 3D scalar 3-coord uniform float",
			REQUIRED);

   AVScreate_output_port("Field Output",
			 "field 3D scalar 3-coord uniform float");

   AVSset_compute_proc(CellVelocity_compute);
}

	
/*--------------------------------------------------------------------------
 * CellVelocity_compute
 *--------------------------------------------------------------------------*/

CellVelocity_compute(k, h, v)
AVSfield_float  *k;
AVSfield_float  *h;
AVSfield_float **v;
{
   double          sqrt(double);

   int             nx, ny, nz;
   float           dx, dy, dz;

   float          *kp, *hp, *vp;
   float           vx, vy, vz;

   int             m1, m2, m3, m4, m5, m6, m7, m8;
   int             ii, jj, kk;

   float          *coords;

   int             dims[3];


   /* free old memory */
   if (*v) 
      AVSfield_free((AVSfield *) *v);

   /*-----------------------------------------------------------------------
    * get parameters and check compatibility
    *-----------------------------------------------------------------------*/

   nx = k -> dimensions[0];
   ny = k -> dimensions[1];
   nz = k -> dimensions[2];

   /* assuming equally spaced in each direction */
   dx = ((k -> max_extent[0]) - (k -> min_extent[0])) / (nx - 1);
   dy = ((k -> max_extent[1]) - (k -> min_extent[1])) / (ny - 1);
   dz = ((k -> max_extent[2]) - (k -> min_extent[2])) / (nz - 1);

   /* check dimension compatibility */
   if ((nx != h -> dimensions[0]) ||
       (ny != h -> dimensions[1]) ||
       (nz != h -> dimensions[2]))
   {
      AVSerror("CellVelocity_compute: dimensions are not compatible");
      return(0);
   }

   /* check spacing compatibility */
   if ((k -> min_extent[0] != h -> min_extent[0]) ||
       (k -> min_extent[1] != h -> min_extent[1]) ||
       (k -> min_extent[2] != h -> min_extent[2]))
   {
      AVSerror("CellVelocity_compute: spacings are not compatible");
      return(0);
   }

   /*-----------------------------------------------------------------------
    * compute the velocity field
    *-----------------------------------------------------------------------*/

   /* create the new AVSfield structure */
   dims[0] = (nx-1);
   dims[1] = (ny-1);
   dims[2] = (nz-1);
   *v = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);

#if 0
   /* setup min and max extent */
   (*v) -> min_extent[0] = 0.0;
   (*v) -> max_extent[0] = (nx-2)*dx;
   (*v) -> min_extent[1] = 0.0;
   (*v) -> max_extent[1] = (ny-2)*dy;
   (*v) -> min_extent[2] = 0.0;
   (*v) -> max_extent[2] = (nz-2)*dz;

   /* setup coords (same and min and max extents) */
   (*v) -> points[0] = 0.0;
   (*v) -> points[1] = (nx-2)*dx;
   (*v) -> points[2] = 0.0;
   (*v) -> points[3] = (ny-2)*dy;
   (*v) -> points[4] = 0.0;
   (*v) -> points[5] = (nz-2)*dz;
#endif

   kp = k -> data;
   hp = h -> data;
   vp = (*v) -> data;

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
	    vx = - (sqrt(kp[m1]*kp[m2])*(hp[m2] - hp[m1]) +
		    sqrt(kp[m3]*kp[m4])*(hp[m4] - hp[m3]) +
		    sqrt(kp[m5]*kp[m6])*(hp[m6] - hp[m5]) +
		    sqrt(kp[m7]*kp[m8])*(hp[m8] - hp[m7])) / (4.0*dx);
	    vy = - (sqrt(kp[m1]*kp[m3])*(hp[m3] - hp[m1]) +
		    sqrt(kp[m2]*kp[m4])*(hp[m4] - hp[m2]) +
		    sqrt(kp[m5]*kp[m7])*(hp[m7] - hp[m5]) +
		    sqrt(kp[m6]*kp[m8])*(hp[m8] - hp[m6])) / (4.0*dy);
	    vz = - (sqrt(kp[m1]*kp[m5])*(hp[m5] - hp[m1] + dz) +
		    sqrt(kp[m3]*kp[m7])*(hp[m7] - hp[m3] + dz) +
		    sqrt(kp[m2]*kp[m6])*(hp[m6] - hp[m2] + dz) +
		    sqrt(kp[m4]*kp[m8])*(hp[m8] - hp[m4] + dz)) / (4.0*dz);
	    
	    *(vp++) = sqrt(vx*vx + vy*vy + vz*vz);
	    
	    kp++;
	    hp++;
	 }
	 kp++;
	 hp++;
      }
      kp += nx;
      hp += nx;
   }

   return(1);
}
