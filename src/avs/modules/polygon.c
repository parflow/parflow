#ifdef SCCS
static char sccsid[]="@(#)polygon.c	8.3 AVS 92/10/16";
#endif
/*
			Copyright (c) 1989,1990 by
			Advanced Visual Systems Inc.
			All Rights Reserved
	
	This software comprises unpublished confidential information of
	Advanced Visual Systems Inc. and may not be used, copied or made
	available to anyone, except in accordance with the license
	under which it is furnished.
	
	This file is under sccs control at AVS in:
	/nfs/caffeine/root/src1/sccs/avs/examples/s.polygon.c
	
*/
#include <stdio.h>
#include <avs/avs.h>
#include <avs/field.h>
#include <avs/geom.h>

/******************************************************************************/

/* 
 * This is a C example to create a geometric object.  In this example,
 * the original data is kept in an ascii description file.  
 * This module converts disjoint polygon information into geom format.  
 * It assumes that the polygons have no normals or colors
 * (but could be easily modified to include either or both).
 * The vertices of the polygons can either be shared by all of the polygons
 * (in which case they will be smooth shaded), or unshared (flat shaded).
 * The format is:
 *  
 *	<type> (either "facet" or "smooth")
 *	n_verts_polygon_1(decimal)
 *	vert[1]X(float) vert[1]Y(float) vert[1]Z(float)
 *	...
 *	n_verts_polygon_2(decimal)
 *	..
 *	<EOF>
 * 
 *  This example is based on the avs-1 geometric filter "polygon.c".  It
 *  can be used with the data file in: /usr/avs/filter/example/cube.polygon
 */  

/* 
 * The function AVSinit_modules is called from the main() routine supplied
 * by AVS.  In it, we call AVSmodule_from_desc with the name of our 
 * description routine.
 */

/*  
 * The routine "polygon_to_geom" is the description routine.  It provides
 * AVS some necessary information such as: name, input and output ports, 
 * parameters etc. 
 */
polygon_to_geom()
{
	int polygon_compute();	/* declare the compute function (below) */
	int out_port;	/* temporaries to hold the port numbers */
	int parm;

	/* Set the module name and type */
	AVSset_module_name("polygon_to_geom", MODULE_DATA);

	/* There are no input ports for this module */

	/* Create an output port for the result */
	out_port = AVScreate_output_port(/*name*/ "Geometry", /*type*/"geom");

	/* Add one paramter: the filename of the polygon object */
	parm = AVSadd_parameter("polygon filename", "string", NULL, NULL, NULL);

	/* Tells AVS to use a file browser as a widget for this module */
   	AVSconnect_widget(parm,"browser");

	/* Tell avs what subroutine to call to do the compute */
	AVSset_compute_proc(polygon_compute);
}

/*
 * polygon_compute is the compute routine.  It is called whenever AVS 
 * is given a new filename for us to compute.  The arguments are: the 
 * output geometry (passed by reference), and the filename parameter.  
 * Note that the order is always inputs, outputs, parameters.
 */

#define MAXVERTS	100  		/* Maximum vertices in a polygon */

polygon_compute(output, filename)
GEOMedit_list *output;
char *filename;
{
   int i;
   char type[100];
   float verts[MAXVERTS][3];
   int nvs, shared, connect;
   GEOMobj *obj;
   FILE *fp;

   /* Return failure (non-zero) if the filename is NULL or we can't open it */
   if (filename == NULL) return(0); 
   if ((fp = fopen(filename,"r")) == NULL) return(0); 

   fscanf(fp,"%s",type);
   if (!strcmp(type,"facet")) { 
      shared = GEOM_NOT_SHARED;
      connect = GEOM_NO_CONNECTIVITY;
   }
   else {
      shared = GEOM_SHARED;
      connect = 0;
   }

   /* Create our GEOM object to add polygons to */
   obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);

   while (fscanf(fp,"%d",&nvs) == 1) {
      for (i = 0; i < nvs; i++)
	 fscanf(fp,"%f%f%f",&verts[i][0],&verts[i][1],&verts[i][2]);
      GEOMadd_disjoint_polygon(obj,verts,NULL,NULL,nvs,shared,0);
   }
   GEOMgen_normals(obj,0); /* Create the normals for the object */

   /* 
    * This converts the representation of the object from a polyhedral
    * representation to a connected triangle list representation which
    * is much more efficient for most hardware to deal with.  It is 
    * not strictly necessary.
    */
   GEOMcvt_polyh_to_polytri(obj,GEOM_SURFACE|GEOM_WIREFRAME|connect);

   /* 
    * Now we communicate this object to AVS: 
    * first we initialize the list of changes for this time to NULL 
    */
   *output = GEOMinit_edit_list(*output);

   /* Now we replace the geometry for the object named "polygon" to this obj */
   GEOMedit_geometry(*output,"polygon",obj);

   /* Then we free up our reference to this object */
   GEOMdestroy_obj(obj); 

   /* 
    * Make sure that module returns success when finished (otherwise
    * AVS might think that it failed and won't execute downstream modules) 
    */
   return(1);
}
