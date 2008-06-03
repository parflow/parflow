/******************************************************************************
 * ReadParticles
 *
 * AVS module to read parflow-binary files.
 *
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: read_particles.c,v $
 Revision 1.1.1.1  2006/02/14 23:05:50  kollet
 CLM.PF_1.0

 Revision 1.1.1.1  2006/02/14 18:51:21  kollet
 CLM.PF_1.0

 Revision 1.1  1994/05/12 22:40:37  falgout
 Initial revision

 * Revision 1.6  1994/02/03  19:24:42  falgout
 * Modified to build into one executable.
 *
 * Revision 1.5  1994/01/11  22:34:45  falgout
 * Changed to float.
 *
 * Revision 1.4  1994/01/11  01:16:01  falgout
 * Changed to uniform fields.
 *
 * Revision 1.3  1993/08/20  16:15:22  falgout
 * Corrected computation of max_extent.
 *
 * Revision 1.2  1993/08/03  16:34:25  falgout
 * Made rectilinear.
 *
 * Revision 1.1  1993/07/29  22:08:36  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * ReadParticles
 *--------------------------------------------------------------------------*/

ReadParticles()
{
   int ReadParticles_compute();
   int p;


   AVSset_module_name("read particles", MODULE_DATA);

   AVScreate_output_port("Particle Set 0",
			 "field 1D 2-vector 3-coord irregular float");

   p = AVSadd_parameter("Read Volume Browser", "string", 0, 0, "");

   AVSconnect_widget(p, "browser");

   AVSset_compute_proc(ReadParticles_compute);
}

	
/*--------------------------------------------------------------------------
 * ReadParticles_compute
 *--------------------------------------------------------------------------*/

ReadParticles_compute(parts, filename)
AVSfield_float **parts;
char            *filename;
{
   FILE   *fp;

   int     i, n;
   int     x, y, z;

   double  value[2];

   float  *x_coord_p, *y_coord_p, *z_coord_p;
   float  *parts_p;

   int dims[1];


   /* no filename yet */
   if (!filename)
      return(1);
    
   /* open filename */
   if (!(fp = fopen(filename, "r")))
   {
      AVSerror("ReadParticles_compute: can't open data file %s", filename);
      return(0);
   }

   /* free old memory */
   if (*parts) 
      AVSfield_free((AVSfield *) *parts);

   /*-----------------------------------------------------------------------
    * read the data
    *-----------------------------------------------------------------------*/

   /* read in header info */
   fscanf(fp, "%d", &n);

   /* create the new AVSfield structure */
   dims[0] = n;
   *parts = (AVSfield_float *)
      AVSdata_alloc("field 1D 2-vector 3-coord irregular float", dims);

   /* read in the field data */
   parts_p   = ((*parts) -> data);
   x_coord_p = ((*parts) -> points);
   y_coord_p = x_coord_p + n;
   z_coord_p = y_coord_p + n;
   for (i = 0; i < n; i++)
   {
      fscanf(fp, "%d%d%d%lf%lf", &x, &y, &z, &value[0], &value[1]);

      x_coord_p[i] = (float) x;
      y_coord_p[i] = (float) y;
      z_coord_p[i] = (float) z;
      parts_p[2*i + 0] = (float) value[0];
      parts_p[2*i + 1] = (float) value[1];
   }

   /*-----------------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------------*/

   fclose(fp);
   
   return(1);
}
