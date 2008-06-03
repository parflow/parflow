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
 * AVS module to sum values in the z dim.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>

	
/*--------------------------------------------------------------------------
 * CellVelocity_compute
 *--------------------------------------------------------------------------*/

int VertSum_compute(AVSfield_float  *input_field, AVSfield_float **output_field)
{
   int             nx, ny, nz;

   float          *in_p, *out_p;

   int             i, j, k;

   float          *coords;

   int             dims[2];


   /* free old memory */
   if (*output_field) 
      AVSfield_free((AVSfield *) *output_field);

   /*-----------------------------------------------------------------------
    * get parameters and check compatibility
    *-----------------------------------------------------------------------*/

   nx = input_field -> dimensions[0];
   ny = input_field -> dimensions[1];
   nz = input_field -> dimensions[2];

   /*-----------------------------------------------------------------------
    * compute the velocity field
    *-----------------------------------------------------------------------*/

   /* create the new AVSfield structure */
   dims[0] = nx;
   dims[1] = ny;
   *output_field = (AVSfield_float *)
      AVSdata_alloc("field 2D scalar 2-coord uniform float", dims);

   in_p = input_field -> data;
   out_p = (*output_field) -> data;

   for (k = 0; k < nz; k++)
      for (j = 0; j < ny; j++)
	 for (i = 0; i < nx; i++)
	    out_p[i + (j*nx)] += in_p[i + (j*nx) + (k*ny*nx)];
   
   return(1);
}


/*--------------------------------------------------------------------------
 * CellVelocity
 *--------------------------------------------------------------------------*/

int VertSum()
{
   int p;


   AVSset_module_name("vertical sum", MODULE_FILTER);

   AVScreate_input_port("Field Input",
			"field 3D scalar 3-coord uniform float",
			REQUIRED);

   AVScreate_output_port("Field Output",
			 "field 2D scalar 2-coord uniform float");

   AVSset_compute_proc(VertSum_compute);
}

