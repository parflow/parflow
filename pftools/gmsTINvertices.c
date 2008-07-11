/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
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
   gms_TIN     **TINs;
   gms_TIN     **new_TINs;
   gms_TIN     **tmp_TINs;
   int      	 nTINs;
   int      	 new_nTINs;

   Vertex      **vertices;
   int      	 nvertices;
	    
   int      	 T, v, i;


   if (argc < 3) 
   {
      fprintf(stderr,
	      "Usage:  gmsTINvertices <TIN input files> <TIN output file>\n");
      exit(1);
   }

   /*-----------------------------------------------------------------------
    * Read in the gms TIN files
    *-----------------------------------------------------------------------*/
   
   nTINs = 0;
   for (i = 1; i <= (argc-2); i++)
   {
      /* read the TINs in next input file */
      gms_ReadTINs(&new_TINs, &new_nTINs, argv[i]);

      /* add the new TINs to the TINs array */
      tmp_TINs = TINs;
      TINs = ctalloc(gms_TIN *, (nTINs + new_nTINs));
      for (T = 0; T < nTINs; T++)
	 TINs[T] = tmp_TINs[T];
      for (T = 0; T < new_nTINs; T++)
	 TINs[nTINs+T] = new_TINs[T];
      nTINs += new_nTINs;
      tfree(tmp_TINs);
      tfree(new_TINs);
   }
   
   /*-----------------------------------------------------------------------
    * Concatenate the vertices of the TINs and set the z-component to 0
    *-----------------------------------------------------------------------*/
   
   nvertices = 0;
   for (T = 0; T < nTINs; T++)
      nvertices += (TINs[T] -> nvertices);
   
   vertices = ctalloc(Vertex *, nvertices);
   v = 0;
   for (T = 0; T < nTINs; T++)
      for (i = 0; i < (TINs[T] -> nvertices); i++, v++)
      {
	 vertices[v] = (TINs[T] -> vertices[i]);
	 (vertices[v] -> z) = 0.0;
      }
   
   /*-----------------------------------------------------------------------
    * Sort the vertices (y first, then x; i.e. x varies fastest)
    *-----------------------------------------------------------------------*/
   
   SortXYVertices(vertices, nvertices, 0);

   /*-----------------------------------------------------------------------
    * Eliminate duplicate xy vertices
    *-----------------------------------------------------------------------*/
   
   i = 0;
   for (v = 0; v < nvertices; v++)
   {
      if (((vertices[v] -> x) != (vertices[i] -> x)) ||
	  ((vertices[v] -> y) != (vertices[i] -> y)))
      {
	 i++;
	 vertices[i] = vertices[v];
      }
   }
   nvertices = (i+1);
   
   /*-----------------------------------------------------------------------
    * Create the output TIN structure
    *-----------------------------------------------------------------------*/

   new_TINs = ctalloc(gms_TIN *, 1);
   new_TINs[0] = ctalloc(gms_TIN, 1);

   (new_TINs[0] -> vertices)  = vertices;
   (new_TINs[0] -> nvertices) = nvertices;
   
   /*-----------------------------------------------------------------------
    * Write the output file
    *-----------------------------------------------------------------------*/

   gms_WriteTINs(new_TINs, 1, argv[argc-1]);

   return(0);
}
