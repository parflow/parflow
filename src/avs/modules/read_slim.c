/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * ReadSlim
 *
 * AVS module to read slim particle files.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * ReadSlim
 *--------------------------------------------------------------------------*/

ReadSlim()
{
   int ReadSlim_compute();
   int p;


   AVSset_module_name("read slim", MODULE_DATA);

   AVScreate_output_port("Scattered Output",
			 "field 1D scalar 3-coord irregular float");

   p = AVSadd_parameter("Read Volume Browser", "string", NULL, "", "");
   AVSconnect_widget(p, "browser");

   p = AVSadd_parameter("Contaminant", "choice", "1", "1!2", "!");
   AVSconnect_widget(p, "radio_buttons");

   AVSset_compute_proc(ReadSlim_compute);
}

	
/*--------------------------------------------------------------------------
 * ReadSlim_compute
 *--------------------------------------------------------------------------*/

ReadSlim_compute(field, filename, contaminant)
AVSfield_float **field;
char            *filename;
char            *contaminant;
{
   FILE   *fp;

   int     i, n;

   float  *x_coord_p, *y_coord_p, *z_coord_p;
   float  *field_p;

   float   f_junk;

   int dims[1];


   /* no filename yet */
   if (!filename)
      return(1);
    
   /* open filename */
   if (!(fp = fopen(filename, "r")))
   {
      AVSerror("ReadSlim_compute: can't open data file %s", filename);
      return(0);
   }

   /* free old memory */
   if (*field) 
      AVSfield_free((AVSfield *) *field);

   /*-----------------------------------------------------------------------
    * read the data
    *-----------------------------------------------------------------------*/

   /* read in header info */
   fscanf(fp, "%d", &n);

   /* create the new AVSfield structure */
   dims[0] = n;
   *field = (AVSfield_float *)
      AVSdata_alloc("field 1D scalar 3-coord irregular float", dims);

   /* read in the field data */
   field_p   = ((*field) -> data);
   x_coord_p = ((*field) -> points);
   y_coord_p = x_coord_p + n;
   z_coord_p = y_coord_p + n;
   switch( AVSchoice_number( "Contaminant", contaminant ) )
   {
   case 1:
      for (i = 0; i < n; i++)
      {
	 fscanf(fp, "%f%f%f%f%f",
		(x_coord_p + i), (y_coord_p + i), (z_coord_p + i),
		(field_p + i), &f_junk);
      }
      break;

   case 2:
      for (i = 0; i < n; i++)
      {
	 fscanf(fp, "%f%f%f%f%f",
		(x_coord_p + i), (y_coord_p + i), (z_coord_p + i),
		&f_junk, (field_p + i));
      }
      break;
   }

   /*-----------------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------------*/

   fclose(fp);
   
   return(1);
}
