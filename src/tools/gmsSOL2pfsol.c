/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.6 $
 *********************************************************************EHEADER*/

#include "gms.h"

#include <stdio.h>


/*--------------------------------------------------------------------------
 * Main routine
 *--------------------------------------------------------------------------*/

int main (argc, argv)
int argc;
char **argv;
{
   FILE	        *outfile;

   gms_Solid   **solids;
   gms_Solid   **new_solids;
   gms_Solid   **tmp_solids;
   int      	 nsolids;
   int      	 new_nsolids;
	    
   Vertex      **vertices;
   int      	 nvertices;
	    
   Triangle    **triangles;
   int      	 ntriangles;
	    
   int      	*new_to_old;       /* maps new indices to old indices */
   int      	*old_to_new;       /* maps old indices to new indices */
	    
   int      	 s, v, t, i;


   if (argc < 3) 
   {
      fprintf(stderr, "Usage:  gmsSOL2pfsol <gms solid files> <pfsol file>\n");
      exit(1);
   }

   /*-----------------------------------------------------------------------
    * Read in the gms SOLID files
    *-----------------------------------------------------------------------*/
   
   nsolids = 0;
   for (i = 1; i <= (argc-2); i++)
   {
      /* read the solids in next input file */
      gms_ReadSolids(&new_solids, &new_nsolids, argv[i]);

      /* add the new solids to the solids array */
      tmp_solids = solids;
      solids = ctalloc(gms_Solid *, (nsolids + new_nsolids));
      for (s = 0; s < nsolids; s++)
	 solids[s] = tmp_solids[s];
      for (s = 0; s < new_nsolids; s++)
	 solids[nsolids+s] = new_solids[s];
      nsolids += new_nsolids;
      tfree(tmp_solids);
      tfree(new_solids);
   }
   
   /*-----------------------------------------------------------------------
    * Concatenate the vertices of the solids
    *-----------------------------------------------------------------------*/
   
   nvertices = 0;
   for (s = 0; s < nsolids; s++)
      nvertices += (solids[s] -> nvertices);
   
   vertices = ctalloc(Vertex *, nvertices);
   v = 0;
   for (s = 0; s < nsolids; s++)
      for (i = 0; i < (solids[s] -> nvertices); i++, v++)
	 vertices[v] = (solids[s] -> vertices[i]);
   
   /*-----------------------------------------------------------------------
    * Sort the vertices (z first, then y, then x; i.e. x varies fastest)
    * and get the new_to_old array.
    *-----------------------------------------------------------------------*/
   
   new_to_old = SortVertices(vertices, nvertices, 1);

   /*-----------------------------------------------------------------------
    * Eliminate duplicate vertices and create old_to_new index map array
    *-----------------------------------------------------------------------*/
   
   old_to_new = ctalloc(int, nvertices);
   i = 0;
   for (v = 0; v < nvertices; v++)
   {
      if (((vertices[v] -> x) != (vertices[i] -> x)) ||
	  ((vertices[v] -> y) != (vertices[i] -> y)) ||
	  ((vertices[v] -> z) != (vertices[i] -> z)))
      {
	 i++;
	 vertices[i] = vertices[v];
      }
      old_to_new[new_to_old[v]] = i;
   }
   nvertices = (i+1);
   
   /*-----------------------------------------------------------------------
    * Print out the `.pfsol' file
    *-----------------------------------------------------------------------*/
   
   /* open the output file */
   outfile = fopen(argv[argc-1], "w");
   
   /* print out the version number */
   fprintf(outfile,"1\n");
   
   /* print out nvertices */
   fprintf(outfile,"%d\n", nvertices);
   
   /* print out the vertices */
   for (v = 0; v < nvertices; v++)
      fprintf(outfile,"%.15e %.15e %.15e\n",
	      (vertices[v] -> x),
	      (vertices[v] -> y),
	      (vertices[v] -> z));
   
   /* print out nsolids */
   fprintf(outfile,"%d\n", nsolids);
   
   /* print out the solid information */
   v = 0;
   for (s = 0; s < nsolids; s++)
   {
      /* print solid_name and mat_id to stdout */
      printf("solid %d: name = %s, material id = %d\n",
	     s, (solids[s] -> solid_name), (solids[s] -> mat_id));

      triangles  = (solids[s] -> triangles);
      ntriangles = (solids[s] -> ntriangles);

      /* print out ntriangles */
      fprintf(outfile,"%d\n", ntriangles);

      /* print out the triangles, making sure to remap the vertex indices */
      for (t = 0; t < ntriangles; t++)
      {
	 fprintf(outfile,"%d %d %d\n",
		 old_to_new[v + (triangles[t] -> v0)],
		 old_to_new[v + (triangles[t] -> v1)],
		 old_to_new[v + (triangles[t] -> v2)]);
      }

      /* print number of patches = 0 */
      fprintf(outfile,"0\n");

      v += (solids[s] -> nvertices);
   }

   /* close the output file */
   fclose(outfile);

   return(0);
}
