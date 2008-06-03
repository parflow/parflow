/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#define MAXVERTS 4

#include <stdio.h>
#include <avs/avs.h>
#include <avs/field.h>
#include <avs/geom.h>

void DrawIt(GEOMobj *obj, float x, float y, float z_lower, float z_upper,
	    float radius)
{
   int nvs=4;
   float vert[MAXVERTS][3];
   int shared;

#if 0
   shared = GEOM_SHARED;
#else
   shared = GEOM_NOT_SHARED;
#endif

   /* Face 1 */
   vert[0][0] = x-radius; 
   vert[0][1] = y-radius; 
   vert[0][2] = z_upper;
   
   vert[1][0] = x-radius; 
   vert[1][1] = y-radius; 
   vert[1][2] = z_lower;
   
   vert[2][0] = x+radius; 
   vert[2][1] = y-radius; 
   vert[2][2] = z_lower;
   
   vert[3][0] = x+radius; 
   vert[3][1] = y-radius; 
   vert[3][2] = z_upper;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
   
   /* Face 2 */
   vert[0][0] = x+radius; 
   vert[0][1] = y-radius; 
   vert[0][2] = z_upper;
   
   vert[1][0] = x+radius; 
   vert[1][1] = y-radius; 
   vert[1][2] = z_lower;
   
   vert[2][0] = x+radius; 
   vert[2][1] = y+radius; 
   vert[2][2] = z_lower;
   
   vert[3][0] = x+radius; 
   vert[3][1] = y+radius; 
   vert[3][2] = z_upper;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
   
   /* Face 3 */
   vert[0][0] = x+radius; 
   vert[0][1] = y+radius; 
   vert[0][2] = z_upper;
   
   vert[1][0] = x+radius; 
   vert[1][1] = y+radius; 
   vert[1][2] = z_lower;
   
   vert[2][0] = x-radius; 
   vert[2][1] = y+radius; 
   vert[2][2] = z_lower;
   
   vert[3][0] = x-radius; 
   vert[3][1] = y+radius; 
   vert[3][2] = z_upper;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
   /* Face 4 */
   vert[0][0] = x-radius; 
   vert[0][1] = y+radius; 
   vert[0][2] = z_upper;
   
   vert[1][0] = x-radius; 
   vert[1][1] = y+radius; 
   vert[1][2] = z_lower;
   
   vert[2][0] = x-radius; 
   vert[2][1] = y-radius; 
   vert[2][2] = z_lower;
   
   vert[3][0] = x-radius; 
   vert[3][1] = y-radius; 
   vert[3][2] = z_upper;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
   /* Face 5 */
   vert[0][0] = x-radius; 
   vert[0][1] = y-radius; 
   vert[0][2] = z_upper;
   
   vert[1][0] = x+radius; 
   vert[1][1] = y-radius; 
   vert[1][2] = z_upper;
   
   vert[2][0] = x+radius; 
   vert[2][1] = y+radius; 
   vert[2][2] = z_upper;

   vert[3][0] = x-radius; 
   vert[3][1] = y+radius; 
   vert[3][2] = z_upper;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
   /* Face 6 */
   vert[0][0] = x-radius; 
   vert[0][1] = y-radius; 
   vert[0][2] = z_lower;
   
   vert[1][0] = x+radius; 
   vert[1][1] = y-radius; 
   vert[1][2] = z_lower;
   
   vert[2][0] = x+radius; 
   vert[2][1] = y+radius; 
   vert[2][2] = z_lower;
   
   vert[3][0] = x-radius; 
   vert[3][1] = y+radius; 
   vert[3][2] = z_lower;
   
   GEOMadd_disjoint_polygon(obj,vert,NULL,NULL,nvs,shared,0);
}

compute_well_geom(GEOMedit_list *output, 
		  char *filename,
		  float *well_diam,
		  float *pipe_diam)
{
   int i;
   int c;

   float x, y, z_lower, z_upper;

   double BackgroundX, BackgroundY, BackgroundZ; 
   int BackgroundNX, BackgroundNY, BackgroundNZ; 
   double BackgroundDX, BackgroundDY, BackgroundDZ; 

   int NumPhases, NumComp, NumWells;

   int WellNum;
   
   int WellNameLength;
   char WellName[2048];

   double WellX_l, WellY_l, WellZ_l;
   double WellX_u, WellY_u, WellZ_u;

   double WellDiamIgnored;

   int WellType, WellAction;

   int num_zones;

   int connect=0;

   GEOMobj *obj;
   FILE *fp=NULL;

   /* Create our GEOM object to add polygons to */
   obj = GEOMcreate_obj(GEOM_POLYHEDRON,NULL);
   
   /* Return failure (non-zero) if the filename is NULL or we can't open it */

   if (filename == NULL)
      return(0); 
   if ((fp = fopen(filename,"r")) == NULL) 
      return(0); 

   fscanf(fp, "%lf%lf%lf", &BackgroundX, &BackgroundY, &BackgroundZ);
   fscanf(fp, "%d%d%d", &BackgroundNX, &BackgroundNY, &BackgroundNZ);
   fscanf(fp, "%lf%lf%lf", &BackgroundDX, &BackgroundDY, &BackgroundDZ);

   fscanf(fp, "%d%d%d", &NumPhases, &NumComp, &NumWells);

   for(i=0; i < NumWells; i++)
   {      
      /* The basic well location */

      /* Need to get the CR at the end of the line */
      fscanf(fp, "%d%d%c", &WellNum, &WellNameLength, WellName);

      for(c = 0; c < WellNameLength; c++)
      {
	 fscanf(fp, "%c", WellName+c);
      }

      fscanf(fp, "%lf%lf%lf", &WellX_l, &WellY_l, &WellZ_l);

      fscanf(fp, "%lf%lf%lf", &WellX_u, &WellY_u, &WellZ_u);

      fscanf(fp, "%lf", &WellDiamIgnored);

      fscanf(fp, "%d%d", &WellType, &WellAction);

      /* Draw Screened area */
      x = (int) (((WellX_l - BackgroundX)/ BackgroundDX) + 0.5)+0.5;
      y = (int) (((WellY_l - BackgroundY)/ BackgroundDY) + 0.5)+0.5;
      z_lower = (int)(((WellZ_l - BackgroundZ)/ BackgroundDZ) + 0.5);
      z_upper = (int)(((WellZ_u - BackgroundZ)/ BackgroundDZ) + 0.5);

      DrawIt(obj, x, y, z_lower, z_upper, *well_diam);

      /* Draw pipe */
      DrawIt(obj, x, y, z_upper, (float)(BackgroundNZ+1), *pipe_diam);
   }
   
   fclose(fp);

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
   GEOMedit_geometry(*output,"Wells",obj);

   /* Then we free up our reference to this object */
   GEOMdestroy_obj(obj); 

   /* 
    * Make sure that module returns success when finished (otherwise
    * AVS might think that it failed and won't execute downstream modules) 
    */
   return(1);
}

int WellGeom()
{
   int out_port;	/* temporaries to hold the port numbers */
   int in_port;
   int parm;
   
   /* Set the module name and type */
   AVSset_module_name("well_to_geom", MODULE_DATA);
   
   /* Input ports for this module */

   /* Create an output port for the result */
   out_port = AVScreate_output_port(/*name*/ "Geometry", /*type*/"geom");
   
   /* Add one paramter: the filename of the polygon object */
   parm = AVSadd_parameter("well filename", "string", NULL, NULL, NULL);
   /* Tells AVS to use a file browser as a widget for this module */
   AVSconnect_widget(parm,"browser");

   AVSconnect_widget(AVSadd_float_parameter("Well Diameter", 0.5,
					    FLOAT_UNBOUND, 
					    FLOAT_UNBOUND), "typein_real");

   AVSconnect_widget(AVSadd_float_parameter("Pipe Diameter", 0.2, 
					    FLOAT_UNBOUND,
					    FLOAT_UNBOUND), "typein_real");
   
   /* Tell avs what subroutine to call to do the compute */
   AVSset_compute_proc(compute_well_geom);
}

