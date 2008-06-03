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
 * AVS module to set the boundaries to zero.
 *
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>

	
/*--------------------------------------------------------------------------
 * CellVelocity_compute
 *--------------------------------------------------------------------------*/

int ZeroBoundary_compute(AVSfield_float  *input_field, AVSfield_float **output_field)
{
   int             nx, ny, nz;

   float          *in_p, *out_p;

   int             i, j, k;

   float          *coords;

   int             dims[3];


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
   dims[2] = nz;
   *output_field = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);

   in_p = input_field -> data;
   out_p = (*output_field) -> data;

   for (k = 0; k < nz; k++)
   {
      for (j = 0; j < ny; j++)
      {
	 for (i = 0; i < nx; i++)
	 {
	    if( ((k == 0) || (k == (nz-1))) ||
	       ((j == 0) || (j == (ny-1))) ||
	       ((i == 0) || (i == (nx-1))) )
	    {
	       *out_p = 0.0;
	    }
	    else
	    {
	       *out_p = *in_p;
	    }
	    
	    out_p++;
	    in_p++;
	 }
      }
   }
   return(1);
}


/*--------------------------------------------------------------------------
 * CellVelocity
 *--------------------------------------------------------------------------*/

int ZeroBoundary()
{
   int p;


   AVSset_module_name("zero boundary", MODULE_FILTER);

   AVScreate_input_port("Field Input",
			"field 3D scalar 3-coord uniform float",
			REQUIRED);

   AVScreate_output_port("Field Output",
			 "field 3D scalar 3-coord uniform float");

   AVSset_compute_proc(ZeroBoundary_compute);
}

