/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "gms.h"

#include <stdio.h>
#include <string.h>

/*--------------------------------------------------------------------------
 * gms_WriteTINs
 *--------------------------------------------------------------------------*/

void           gms_WriteTINs(TINs, nTINs, filename)
gms_TIN      **TINs;
int            nTINs;
char          *filename;
{
   FILE	        *file;

   Vertex      **vertices;
   int      	 nvertices;
	    
   Triangle    **triangles;
   int      	 ntriangles;
	    
   int      	 T, v, t;


   /*-----------------------------------------------------------------------
    * Print the TIN output file
    *-----------------------------------------------------------------------*/
   
   /* open the output file */
   file = fopen(filename, "w");
   
   /* print some heading info */
   fprintf(file, "TIN\n");

   for (T = 0; T < nTINs; T++)
   {
      /* Get vertices and triangles from the TIN structure */
      vertices   = (TINs[T] -> vertices);
      nvertices  = (TINs[T] -> nvertices);
      triangles  = (TINs[T] -> triangles);
      ntriangles = (TINs[T] -> ntriangles);
   
      /* print some TIN heading info */
      fprintf(file, "BEGT\n");
      if (strlen(TINs[T] -> TIN_name))
	 fprintf(file, "TNAM %s\n", (TINs[T] -> TIN_name));
      fprintf(file, "MAT %d\n",  (TINs[T] -> mat_id));
      
      /* print out the vertices */
      if (nvertices)
      {
	 fprintf(file, "VERT %d\n", nvertices);
	 for (v = 0; v < nvertices; v++)
	 {
	    fprintf(file,"%.15e %.15e %.15e  0\n",
		    (vertices[v] -> x),
		    (vertices[v] -> y),
		    (vertices[v] -> z));
	 }
      }
      
      /* print out the triangles */
      if (ntriangles)
      {
	 fprintf(file, "TRI %d\n", ntriangles);
	 for (t = 0; t < ntriangles; t++)
	 {
	    fprintf(file,"%d %d %d\n",
		    (triangles[t] -> v0) + 1,
		    (triangles[t] -> v1) + 1,
		    (triangles[t] -> v2) + 1);
	 }
      }
      
      /* print some TIN closing info */
      fprintf(file, "ENDT\n");
   }

   /* close the output file */
   fclose(file);
}


