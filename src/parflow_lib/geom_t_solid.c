/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for the Geometry class.
 *
 *****************************************************************************/

#include "parflow.h"
#include "geometry.h"


/*--------------------------------------------------------------------------
 * GeomNewTSolid
 *--------------------------------------------------------------------------*/

GeomTSolid  *GeomNewTSolid(surface, patches, num_patches, num_patch_triangles)
GeomTIN     *surface;
int        **patches;              /* arrays of surface triangle indices */
int          num_patches;
int         *num_patch_triangles;
{
    GeomTSolid   *new;


    new = talloc(GeomTSolid, 1);

    (new -> surface)             = surface;
    (new -> patches)             = patches;
    (new -> num_patches)         = num_patches;
    (new -> num_patch_triangles) = num_patch_triangles;

    return new;
}


/*--------------------------------------------------------------------------
 * GeomFreeTSolid
 *--------------------------------------------------------------------------*/

void         GeomFreeTSolid(solid)
GeomTSolid  *solid;
{
   int  p;


   GeomFreeTIN(solid -> surface);
   for (p = 0; p < (solid -> num_patches); p++)
      tfree(solid -> patches[p]);
   tfree(solid -> patches);
   tfree(solid -> num_patch_triangles);

   tfree(solid);
}


/*--------------------------------------------------------------------------
 * GeomReadTSolids
 *--------------------------------------------------------------------------*/

int            GeomReadTSolids(solids_data_ptr, geom_input_name)
GeomTSolid  ***solids_data_ptr;
char *geom_input_name;
{
   GeomTSolid      **solids_data = NULL;

   GeomTIN          *surface;
   GeomVertex      **vertices;
   GeomVertexArray  *vertex_array;
   GeomTriangle    **triangles;
   int               nS, nT, nV;

   int             **patches;
   int               num_patches;
   int              *num_patch_triangles;

   int               s, t, v, p;
		    
   char              *solids_filename;
   amps_File         solids_file;
   int               version_number;

   amps_Invoice      invoice;
   
   char key[IDB_MAX_KEY_LEN];


   /*------------------------------------------------------
    * Read the solids file name and open the file
    *------------------------------------------------------*/

   sprintf(key, "GeomInput.%s.FileName", geom_input_name);
   solids_filename = GetString(key);

   if ((solids_file = amps_SFopen(solids_filename, "r")) == NULL)
   {
      InputError("Error: can't open solids file %s%s\n", solids_filename,
		 "");
   }
   /*------------------------------------------------------
    * Check the file version number
    *------------------------------------------------------*/

   /* Input the number of vertices */
   invoice = amps_NewInvoice("%i", &version_number);
   amps_SFBCast(amps_CommWorld, solids_file, invoice);
   amps_FreeInvoice(invoice);

   if (version_number != PFSOL_GEOM_T_SOLID_VERSION)
   {
      if(!amps_Rank(amps_CommWorld))
	 amps_Printf("Error: need input file version %d\n", 
		     PFSOL_GEOM_T_SOLID_VERSION);
      exit(1);
   }
   
   /*------------------------------------------------------
    * Read in the solid information
    *------------------------------------------------------*/

   /* Input the number of vertices */
   invoice = amps_NewInvoice("%i", &nV);
   amps_SFBCast(amps_CommWorld, solids_file, invoice);
   amps_FreeInvoice(invoice);
   
   vertices     = ctalloc(GeomVertex *, nV);
   
   /* Read in all the vertices */
   for (v = 0; v < nV; v++) 
   {
      vertices[v] = ctalloc(GeomVertex, 1);
   
      invoice = amps_NewInvoice("%d%d%d",
				&GeomVertexX(vertices[v]),
				&GeomVertexY(vertices[v]),
				&GeomVertexZ(vertices[v]));
      amps_SFBCast(amps_CommWorld, solids_file, invoice);
      amps_FreeInvoice(invoice);
   }
   
   vertex_array = GeomNewVertexArray(vertices, nV);

   /* Input the number of solids */
   invoice = amps_NewInvoice("%i", &nS);
   amps_SFBCast(amps_CommWorld, solids_file, invoice);
   amps_FreeInvoice(invoice);
   
   solids_data = ctalloc(GeomTSolid *, nS);

   /* Read in the solids */
   for (s = 0; s < nS; s++)
   {
      /* Input the number of triangles */
      invoice = amps_NewInvoice("%i", &nT);
      amps_SFBCast(amps_CommWorld, solids_file, invoice);
      amps_FreeInvoice(invoice);

      triangles = ctalloc(GeomTriangle *, nT);

      /* Read in the triangles */
      for (t = 0; t < nT; t++) 
      {
	 triangles[t] = ctalloc(GeomTriangle, 1);

         invoice = amps_NewInvoice("%i%i%i",
				   &GeomTriangleV0(triangles[t]),
				   &GeomTriangleV1(triangles[t]),
				   &GeomTriangleV2(triangles[t]));
         amps_SFBCast(amps_CommWorld, solids_file, invoice);
         amps_FreeInvoice(invoice);
      }

      surface = GeomNewTIN(vertex_array, triangles, nT);

      /* Input the number of patches */
      invoice = amps_NewInvoice("%i", &num_patches);
      amps_SFBCast(amps_CommWorld, solids_file, invoice);
      amps_FreeInvoice(invoice);

      patches             = ctalloc(int *, num_patches);
      num_patch_triangles = ctalloc(int, num_patches);

      for (p = 0; p < num_patches; p++)
      {
	 /* Input the number of patches */
	 invoice = amps_NewInvoice("%i", &num_patch_triangles[p]);
	 amps_SFBCast(amps_CommWorld, solids_file, invoice);
	 amps_FreeInvoice(invoice);

	 patches[p] = talloc(int, num_patch_triangles[p]);

	 /* Read in the triangle indices */
	 for (t = 0; t < num_patch_triangles[p]; t++)
	 {
	    invoice = amps_NewInvoice("%i", &patches[p][t]);
	    amps_SFBCast(amps_CommWorld, solids_file, invoice);
	    amps_FreeInvoice(invoice);
	 }
      }

      solids_data[s] =
	 GeomNewTSolid(surface, patches, num_patches, num_patch_triangles);
   }

   amps_SFclose(solids_file);

   *solids_data_ptr = solids_data;

   return (nS);
}


/*--------------------------------------------------------------------------
 * GeomTSolidFromBox
 *--------------------------------------------------------------------------*/

GeomTSolid  *GeomTSolidFromBox(xl, yl, zl, xu, yu, zu)
double       xl;
double       yl;
double       zl;
double       xu;
double       yu;
double       zu;
{
   GeomTSolid       *solid_data;

   GeomTIN          *surface;
   GeomVertex      **vertices;
   GeomVertexArray  *vertex_array;
   GeomTriangle    **triangles;

   int             **patches;
   int               num_patches;
   int              *num_patch_triangles;

   int               i, j, k, v, p;


   /*------------------------------------------------------
    * Set up vertex_array
    *------------------------------------------------------*/

   vertices = ctalloc(GeomVertex *, 8);
   
   v = 0;
   for (k = 0; k < 2; k++)
      for (j = 0; j < 2; j++)
	 for (i = 0; i < 2; i++)
	 {
	    vertices[v] = ctalloc(GeomVertex, 1);
   
	    switch(i)
	    {
	    case 0:
	       GeomVertexX(vertices[v]) = xl;
	       break;
	    case 1:
	       GeomVertexX(vertices[v]) = xu;
	       break;
	    }

	    switch(j)
	    {
	    case 0:
	       GeomVertexY(vertices[v]) = yl;
	       break;
	    case 1:
	       GeomVertexY(vertices[v]) = yu;
	       break;
	    }

	    switch(k)
	    {
	    case 0:
	       GeomVertexZ(vertices[v]) = zl;
	       break;
	    case 1:
	       GeomVertexZ(vertices[v]) = zu;
	       break;
	    }

	    v++;
	 }
   
   vertex_array = GeomNewVertexArray(vertices, 8);

   /*------------------------------------------------------
    * Set up triangles
    *------------------------------------------------------*/

   triangles = ctalloc(GeomTriangle *, 12);

   /* x lower face */
   triangles[0] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[0]) = 2;
   GeomTriangleV1(triangles[0]) = 0;
   GeomTriangleV2(triangles[0]) = 4;
   triangles[1] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[1]) = 4;
   GeomTriangleV1(triangles[1]) = 6;
   GeomTriangleV2(triangles[1]) = 2;

   /* x upper face */
   triangles[2] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[2]) = 1;
   GeomTriangleV1(triangles[2]) = 3;
   GeomTriangleV2(triangles[2]) = 7;
   triangles[3] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[3]) = 7;
   GeomTriangleV1(triangles[3]) = 5;
   GeomTriangleV2(triangles[3]) = 1;

   /* y lower face */
   triangles[4] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[4]) = 0;
   GeomTriangleV1(triangles[4]) = 1;
   GeomTriangleV2(triangles[4]) = 5;
   triangles[5] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[5]) = 5;
   GeomTriangleV1(triangles[5]) = 4;
   GeomTriangleV2(triangles[5]) = 0;

   /* y upper face */
   triangles[6] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[6]) = 3;
   GeomTriangleV1(triangles[6]) = 2;
   GeomTriangleV2(triangles[6]) = 6;
   triangles[7] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[7]) = 6;
   GeomTriangleV1(triangles[7]) = 7;
   GeomTriangleV2(triangles[7]) = 3;

   /* z lower face */
   triangles[8] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[8]) = 2;
   GeomTriangleV1(triangles[8]) = 3;
   GeomTriangleV2(triangles[8]) = 1;
   triangles[9] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[9]) = 1;
   GeomTriangleV1(triangles[9]) = 0;
   GeomTriangleV2(triangles[9]) = 2;

   /* z upper face */
   triangles[10] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[10]) = 4;
   GeomTriangleV1(triangles[10]) = 5;
   GeomTriangleV2(triangles[10]) = 7;
   triangles[11] = ctalloc(GeomTriangle, 1);
   GeomTriangleV0(triangles[11]) = 7;
   GeomTriangleV1(triangles[11]) = 6;
   GeomTriangleV2(triangles[11]) = 4;


   /*------------------------------------------------------
    * Set up patches
    *------------------------------------------------------*/

   num_patches = 6;
   patches             = ctalloc(int *, num_patches);
   num_patch_triangles = ctalloc(int, num_patches);
   for (p = 0; p < num_patches; p++)
   {
      num_patch_triangles[p] = 2;
      patches[p] = talloc(int, 2);

      patches[p][0] = 2*p;
      patches[p][1] = 2*p + 1;
   }

   /*------------------------------------------------------
    * Set up solid
    *------------------------------------------------------*/

   surface = GeomNewTIN(vertex_array, triangles, 12);

   solid_data =
      GeomNewTSolid(surface, patches, num_patches, num_patch_triangles);

   return solid_data;
}

