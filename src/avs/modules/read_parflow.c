/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * ReadParFlow
 *
 * AVS module to read `pfb' and `pfsb' files.
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include <avs.h>
#include <field.h>

#include <../../tools/tools_io.h>


/*--------------------------------------------------------------------------
 * ReadParFlow
 *--------------------------------------------------------------------------*/

ReadParFlow()
{
   int ReadParFlow_compute();
   int p;


   AVSset_module_name("read parflow", MODULE_DATA);

   p = AVScreate_output_port("Uniform Output",
                             "field 3D scalar 3-coord uniform float");

   p = AVScreate_output_port("Scattered Output",
                             "field 1D scalar 3-coord irregular float");

   p = AVSadd_parameter("File Browser", "string",
			NULL, "", ".pfb.pfsb");
   AVSconnect_widget(p, "browser");

   p = AVSadd_parameter("Scan Files", "choice",
			"all", "all:pfb:pfsb", ":");
   AVSconnect_widget(p, "choice");

   AVSset_compute_proc(ReadParFlow_compute);
}

	
/*--------------------------------------------------------------------------
 * ReadParFlow_compute
 *--------------------------------------------------------------------------*/

ReadParFlow_compute(field, sfield, filename, scan_files)
AVSfield_float **field;
AVSfield_float **sfield;
char            *filename;
char            *scan_files;
{
   FILE     *fp;

   char     *file_ext;


   /*-----------------------------------------------------------------------
    * Scan files if `scan_files' parameter has changed
    *-----------------------------------------------------------------------*/

   if (AVSparameter_changed("Scan Files"))
   {
      if (!strcmp(scan_files,"all"))
	 AVSmodify_parameter("File Browser", AVS_MAXVAL,
			     NULL, "", ".pfb.pfsb");

      else if (!strcmp(scan_files,"pfb"))
	 AVSmodify_parameter("File Browser", AVS_MAXVAL,
			     NULL, "", ".pfb");

      else if (!strcmp(scan_files,"pfsb"))
	 AVSmodify_parameter("File Browser", AVS_MAXVAL,
			     NULL, "", ".pfsb");
   }
    
   /*-----------------------------------------------------------------------
    * Return if no filename yet
    *-----------------------------------------------------------------------*/

   if (!filename)
   {
      AVSmark_output_unchanged("Uniform Output");
      AVSmark_output_unchanged("Scattered Output");
      return(1);
   }
    
   /*-----------------------------------------------------------------------
    * Read the data
    *-----------------------------------------------------------------------*/

   if (!(fp = fopen(filename, "r")))
   {
      AVSerror("ReadParFlow_compute: can't open data file %s", filename);
      return(0);
   }

   if (file_ext = strrchr(filename, '.'))
   {
      if (!strcmp(file_ext,".pfb"))
	 ReadPFBData(field, fp);

      else if (!strcmp(file_ext,".pfsb"))
	 ReadPFSBData(field, sfield, fp);
   }

   fclose(fp);

   return(1);
}


/*--------------------------------------------------------------------------
 * ReadPFBData
 *--------------------------------------------------------------------------*/

ReadPFBData(field, fp)
AVSfield_float **field;
FILE            *fp;
{
   double          X,  Y,  Z;
   int             NX, NY, NZ;
   double          DX, DY, DZ;
   int             num_subgrids;

   int             ix,  iy,  iz;
   int             nx, ny, nz;
   int             rx, ry, rz;

   int             nsg, i, j, k;
   int             dims[3];

   float          *field_p;

   double          value;


   /*-----------------------------------------------------------------------
    * Read the data
    *-----------------------------------------------------------------------*/

   /* free old data */
   if (*field)
   {
      AVSfield_free((AVSfield *) *field);
      *field = NULL;
   }

   /* read in header info */
   tools_ReadDouble(fp, &X, 1);
   tools_ReadDouble(fp, &Y, 1);
   tools_ReadDouble(fp, &Z, 1);

   tools_ReadInt(fp, &NX, 1);
   tools_ReadInt(fp, &NY, 1);
   tools_ReadInt(fp, &NZ, 1);

   tools_ReadDouble(fp, &DX, 1);
   tools_ReadDouble(fp, &DY, 1);
   tools_ReadDouble(fp, &DZ, 1);

   tools_ReadInt(fp, &num_subgrids, 1);

   /* create the new AVSfield structure */
   dims[0] = NX;
   dims[1] = NY;
   dims[2] = NZ;
   *field = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);
   if ( *field == NULL )
   {
      AVSerror("ReadPFBData: Allocation of output field (field) failed.");
      return(0);
   }

   /* zero out the field, just to be sure */
   field_p = ((*field) -> data);
   for (i = 0; i < (dims[0]*dims[1]*dims[2]); i++)
      field_p[i] = 0.0;

   /* read in the sub-grid data and put it into the avs field */
   for (nsg = num_subgrids; nsg--;)
   {

     tools_ReadInt(fp, &ix, 1);
     tools_ReadInt(fp, &iy, 1);
     tools_ReadInt(fp, &iz, 1);

     tools_ReadInt(fp, &nx, 1);
     tools_ReadInt(fp, &ny, 1);
     tools_ReadInt(fp, &nz, 1);

     tools_ReadInt(fp, &rx, 1);
     tools_ReadInt(fp, &ry, 1);
     tools_ReadInt(fp, &rz, 1);

      for (k = 0; k < nz; k++)
         for (j = 0; j < ny; j++)
            for (i = 0; i < nx; i++)
            {
	      tools_ReadDouble(fp, &value, 1);
	      I3D(*field,(ix + i),(iy + j),(iz + k)) = (float) value;
            }

   }

   return(1);
}


/*--------------------------------------------------------------------------
 * ReadPFSBData
 *--------------------------------------------------------------------------*/

ReadPFSBData(field, sfield, fp)
AVSfield_float **field;
AVSfield_float **sfield;
FILE            *fp;
{
   double       X,  Y,  Z;
   int          NX, NY, NZ;
   double       DX, DY, DZ;
   int          num_subgrids;

   int          ix, iy, iz;
   int          nx, ny, nz;
   int          rx, ry, rz;

   int          i, j, k, m, n, total_number, current_number, nsg, index;
   int          sdims[1], dims[3];

   float       *sfield_p, *field_p;
   float       *x_coord_p, *y_coord_p, *z_coord_p;

   double       value;

   typedef struct
   {
      int    num_data_values;
      float  *x_coord, *y_coord, *z_coord, *data_value;

   } SubgridData;

   SubgridData *data;


   /*-----------------------------------------------------------------------
    * Read the data
    *-----------------------------------------------------------------------*/

   /* set the extent to zero */
   total_number = 0;

   /* read in header info */
   tools_ReadDouble(fp, &X, 1);
   tools_ReadDouble(fp, &Y, 1);
   tools_ReadDouble(fp, &Z, 1);

   tools_ReadInt(fp, &NX, 1);
   tools_ReadInt(fp, &NY, 1);
   tools_ReadInt(fp, &NZ, 1);

   tools_ReadDouble(fp, &DX, 1);
   tools_ReadDouble(fp, &DY, 1);
   tools_ReadDouble(fp, &DZ, 1);

   tools_ReadInt(fp, &num_subgrids, 1);

   data = (SubgridData *) malloc(num_subgrids * sizeof(SubgridData));

   /* read in the field data */
   for (nsg = 0; nsg < num_subgrids; nsg++)
   {
     tools_ReadInt(fp, &ix, 1);
     tools_ReadInt(fp, &iy, 1);
     tools_ReadInt(fp, &iz, 1);

     tools_ReadInt(fp, &nx, 1);
     tools_ReadInt(fp, &ny, 1);
     tools_ReadInt(fp, &nz, 1);

     tools_ReadInt(fp, &rx, 1);
     tools_ReadInt(fp, &ry, 1);
     tools_ReadInt(fp, &rz, 1);

     tools_ReadInt(fp, &n, 1);

     (data + nsg) -> num_data_values = n;
     (data + nsg) -> x_coord    = (float *) malloc(n * sizeof(float));
     (data + nsg) -> y_coord    = (float *) malloc(n * sizeof(float));
     (data + nsg) -> z_coord    = (float *) malloc(n * sizeof(float));
     (data + nsg) -> data_value = (float *) malloc(n * sizeof(float));

     for (m = 0; m < n; m++)
       {
	 tools_ReadInt(fp, &i, 1);
	 tools_ReadInt(fp, &j, 1);
	 tools_ReadInt(fp, &k, 1);

	 tools_ReadDouble(fp, &value, 1);

         *(((data + nsg) -> x_coord) + m)    = (float) i;
         *(((data + nsg) -> y_coord) + m)    = (float) j;
         *(((data + nsg) -> z_coord) + m)    = (float) k;
         *(((data + nsg) -> data_value) + m) = (float) value;
         total_number++;
      }

   }

   /*-----------------------------------------------------------------------
    * Create the scattered field, `sfield'
    *-----------------------------------------------------------------------*/

   /* free old data */
   if (*sfield)
   {
      AVSfield_free((AVSfield *) *sfield);
      *sfield = NULL;
   }

   /* allocate `sfield' */
   sdims[0] = total_number;
   *sfield = (AVSfield_float *)
      AVSdata_alloc("field 1D scalar 3-coord irregular float", sdims);
   if ( *sfield == NULL )
   {
      AVSerror("ReadPFSBData: Allocation of output field (sfield) failed.");
      return(0);
   }

   /* copy the data into `sfield' */
   sfield_p  = ((*sfield) -> data);
   x_coord_p = ((*sfield) -> points);
   y_coord_p = x_coord_p + total_number;
   z_coord_p = y_coord_p + total_number;
   current_number = 0;
   for(nsg = 0; nsg < num_subgrids; nsg++)
   {
      for(m = 0; m < (data + nsg) -> num_data_values; m++)
      {
         x_coord_p[current_number] = *(((data + nsg) -> x_coord) + m);
         y_coord_p[current_number] = *(((data + nsg) -> y_coord) + m);
         z_coord_p[current_number] = *(((data + nsg) -> z_coord) + m);
         sfield_p[current_number]  = *(((data + nsg) -> data_value) + m);
         current_number++;
      }
   }

   /* free up the `SubgridData' structure `data' */
   for (nsg = 0; nsg < num_subgrids; nsg++)
   {
      free((data + nsg) -> x_coord);
      free((data + nsg) -> y_coord);
      free((data + nsg) -> z_coord);
      free((data + nsg) -> data_value);
   }
   free(data);

   /*-----------------------------------------------------------------------
    * Create the uniform field, `field'
    *-----------------------------------------------------------------------*/

   /* free old data */
   if (*field)
   {
      AVSfield_free((AVSfield *) *field);
      *field = NULL;
   }

   /* allocate `field' */
   dims[0] = NX;
   dims[1] = NY;
   dims[2] = NZ;
   *field = (AVSfield_float *)
      AVSdata_alloc("field 3D scalar 3-coord uniform float", dims);

   /* zero out the field, just to be sure */
   field_p = ((*field) -> data);
   for (i = 0; i < (dims[0]*dims[1]*dims[2]); i++)
      field_p[i] = 0.0;

   /* copy `sfield' into `field' */
   n = ((*sfield) -> dimensions[0]);
   sfield_p  = ((*sfield) -> data);
   x_coord_p = ((*sfield) -> points);
   y_coord_p = x_coord_p + n;
   z_coord_p = y_coord_p + n;
   for (i = 0; i < n; i++)
   {
      index = z_coord_p[i]*NY*NX + y_coord_p[i]*NX + x_coord_p[i];
      field_p[index] = sfield_p[i];
   }

   return(1);
}

