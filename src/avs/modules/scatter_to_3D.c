/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * ScatterTo3D
 *
 * AVS module for converting scatter data to 3D field data.
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
 * ScatterTo3D
 *--------------------------------------------------------------------------*/

ScatterTo3D()
{
   int ScatterTo3D_compute();
   int p;


   AVSset_module_name("scatter to 3D", MODULE_FILTER);

   AVScreate_input_port("3D Field",
			"field 3D 3-coord uniform float",
			OPTIONAL);
   AVScreate_input_port("Scatter Field",
			"field 1D scalar 3-coord irregular float",
			REQUIRED);

   AVScreate_output_port("Field Output",
			 "field 3D scalar 3-coord uniform float");

   p = AVSadd_parameter("NX", "integer", 21, 1, INT_UNBOUND);
   AVSconnect_widget(p, "typein_integer");

   p = AVSadd_parameter("NY", "integer", 21, 1, INT_UNBOUND);
   AVSconnect_widget(p, "typein_integer");

   p = AVSadd_parameter("NZ", "integer", 21, 1, INT_UNBOUND);
   AVSconnect_widget(p, "typein_integer");

   AVSset_compute_proc(ScatterTo3D_compute);
}

	
/*--------------------------------------------------------------------------
 * ScatterTo3D_compute
 *--------------------------------------------------------------------------*/

ScatterTo3D_compute(field_3d, scatter, out_field, NX, NY, NZ)
AVSfield_float  *field_3d;
AVSfield_float  *scatter;
AVSfield_float **out_field;
int              NX;
int              NY;
int              NZ;
{
   float  *x_coord_p, *y_coord_p, *z_coord_p;
   float  *fp, *scatter_p;

   int     i, n, index;

   int     dims[3];


   /* free old memory */
   if (*out_field) 
      AVSfield_free((AVSfield *) *out_field);

   /*-----------------------------------------------------------------------
    * get parameters and check compatibility
    *-----------------------------------------------------------------------*/

   /* set NX, NY, NZ parameters */
   if (AVSinput_changed("3D Field", 0))
   {
      NX = (field_3d -> dimensions[0]);
      NY = (field_3d -> dimensions[1]);
      NZ = (field_3d -> dimensions[2]);
      AVSmodify_parameter("NX",AVS_VALUE,NX,0,0);
      AVSmodify_parameter("NY",AVS_VALUE,NY,0,0);
      AVSmodify_parameter("NZ",AVS_VALUE,NZ,0,0);
   }

   /* create the new AVSfield structure */
   dims[0] = NX;
   dims[1] = NY;
   dims[2] = NZ;
   *out_field = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);

   fp = ((*out_field) -> data);
   for (i = 0; i < (dims[0]*dims[1]*dims[2]); i++)
      fp[i] = 0.0;

   n = (scatter -> dimensions[0]);
   scatter_p = (scatter -> data);
   x_coord_p = (scatter -> points);
   y_coord_p = x_coord_p + n;
   z_coord_p = y_coord_p + n;
   for (i = 0; i < n; i++)
   {
      index = Index(x_coord_p[i], y_coord_p[i], z_coord_p[i],
		    dims[0], dims[1], dims[2]);
      fp[index] = scatter_p[i];
   }

   return(1);
}
