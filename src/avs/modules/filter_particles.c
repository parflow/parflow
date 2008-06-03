/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * FilterParticles
 *
 * AVS module to remove particles with values below a certain threshold.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * some global macros
 *--------------------------------------------------------------------------*/

#define Index(i, j, k, NX, NY, NZ) ((k)*NY*NX + (j)*NX + (i))


/*--------------------------------------------------------------------------
 * FilterParticles
 *--------------------------------------------------------------------------*/

FilterParticles()
{
   int FilterParticles_compute();
   int p;


   AVSset_module_name("filter particles", MODULE_FILTER);

   AVScreate_input_port("Scatter Input",
			"field 1D scalar 3-coord irregular float",
			REQUIRED);

   AVScreate_output_port("Scatter Output",
			 "field 1D scalar 3-coord irregular float");

   p = AVSadd_float_parameter("min", 0.00000, FLOAT_UNBOUND, FLOAT_UNBOUND);
   AVSconnect_widget(p, "typein_real");

   AVSset_compute_proc(FilterParticles_compute);
}

	
/*--------------------------------------------------------------------------
 * FilterParticles_compute
 *--------------------------------------------------------------------------*/

FilterParticles_compute(scatter_in, scatter_out, min)
AVSfield_float  *scatter_in;
AVSfield_float **scatter_out;
float           *min;
{
   float  *in_data, *out_data;
   float  *in_x_coord,  *in_y_coord,  *in_z_coord;
   float  *out_x_coord, *out_y_coord, *out_z_coord;

   int     in_sz, out_sz;

   int    *index_array;
   int     i;

   int     dims[1];


   /* free old memory */
   if (*scatter_out) 
      AVSfield_free((AVSfield *) *scatter_out);

   /*-----------------------------------------------------------------------
    * filter the particles
    *-----------------------------------------------------------------------*/

   in_sz      = ((scatter_in) -> dimensions[0]);
   in_data    = ((scatter_in) -> data);
   in_x_coord = ((scatter_in) -> points);
   in_y_coord = in_x_coord + in_sz;
   in_z_coord = in_y_coord + in_sz;

   index_array = (int *) malloc(in_sz * sizeof(int));

   out_sz = 0;
   for (i = 0; i < in_sz; i++)
      if (in_data[i] > *min)
      {
	 index_array[out_sz] = i;
	 out_sz++;
      }

   /* create the new AVSfield structure */
   dims[0] = out_sz;
   *scatter_out = (AVSfield_float *)
      AVSdata_alloc("field 1D scalar 3-coord irregular float", dims);

   out_data    = ((*scatter_out) -> data);
   out_x_coord = ((*scatter_out) -> points);
   out_y_coord = out_x_coord + out_sz;
   out_z_coord = out_y_coord + out_sz;
   for (i = 0; i < out_sz; i++)
   {
      out_data[i]    = in_data[index_array[i]];
      out_x_coord[i] = in_x_coord[index_array[i]];
      out_y_coord[i] = in_y_coord[index_array[i]];
      out_z_coord[i] = in_z_coord[index_array[i]];
   }

   free(index_array);

   return(1);
}
