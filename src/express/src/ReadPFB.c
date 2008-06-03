#include "user.h"
#include <stdio.h>
#include <string.h>

#include <avs/err.h>
#include <avs/fld.h>

#include <../../tools/tools_io.h>

#define ERR_RETURN(A) {ERRerror("read",1,ERR_ORIG, A); return(0);}


/* read a .pfb file into a uniform grid */
int
ReadPFB(OMobj_id ReadPFB_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *filename = NULL;
   OMobj_id Uniform_Grid_id;
   int Uniform_Grid_ndim, *Uniform_Grid_dims, Uniform_Grid_nspace, Uniform_Grid_npoints;
   float *Uniform_Grid_points;
   int Uniform_Grid_ncomp, Uniform_Grid_comp_count, Uniform_Grid_veclen;
   int    Uniform_Grid_data_type, Uniform_Grid_ndata;
   float *Uniform_Grid_data;

   /* my declarations */
   FILE* fp;

   double          X,  Y,  Z;
   int             NX, NY, NZ;
   double          DX, DY, DZ;
   int             num_subgrids;
 
   int             ix,  iy,  iz;
   int             nx, ny, nz;
   int             rx, ry, rz;
 
   int             nsg, i, j, k;
 
   double          value;

   char*	   label;



   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get filename's value */
   if (OMget_name_str_val(ReadPFB_id, OMstr_to_name("filename"), &filename, 0) != 1)
      filename = NULL;

   if (filename == NULL)
      ERR_RETURN("ReadPFB: no filename given");
   if (strcmp(strrchr(filename,'.'),".pfb") != 0)
      ERR_RETURN("ReadPFB: not a .pfb file");

   label = strrchr(filename,'/');
   if (label != NULL)
      label++;
   if (label == NULL)
      label = filename;
   
   /* Call does not exist:
      FLDset_node_data_label (Uniform_Grid_id, 0, label); */


   /***********************/
   /* Function's Body     */
   /***********************/
   /*
      ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: ReadPFB generated from method: ReadPFB.ReadFile\n");
   */


   /***********************/
   /*  Set output values  */
   /***********************/
    /* Set Uniform_Grid uniform mesh */

   /*  Get mesh id */
   Uniform_Grid_id = OMfind_subobj(ReadPFB_id, OMstr_to_name("Uniform_Grid"), OM_OBJ_RW);

   /* Set mesh dimensionality, Uniform_Grid_ndim can be 1,2 or 3 */
   Uniform_Grid_ndim = 3;
   FLDset_ndim (Uniform_Grid_id, Uniform_Grid_ndim);

   fp = fopen(filename,"r");
   if (fp == NULL)
      ERR_RETURN("ReadPFB: cannot open file");

   

   /* Set mesh dims array */
   Uniform_Grid_dims = (int *)ARRalloc(NULL, DTYPE_INT, 
                           Uniform_Grid_ndim, NULL);


   /*** fill in dims array with your values ***/

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

   Uniform_Grid_dims[0] = NX;
   Uniform_Grid_dims[1] = NY;
   Uniform_Grid_dims[2] = NZ;

   FLDset_dims (Uniform_Grid_id, Uniform_Grid_dims);
   FLDset_nnodes (Uniform_Grid_id, NX*NY*NZ);
   if (Uniform_Grid_dims)
      ARRfree((char *)Uniform_Grid_dims);

   /* Set mesh nspace, Uniform_Grid_nspace can be 1,2 or 3 */
   Uniform_Grid_nspace = 3;
   FLDset_nspace (Uniform_Grid_id, Uniform_Grid_nspace);

   /* Set mesh extents */
   Uniform_Grid_npoints = 2*Uniform_Grid_nspace;
   /* added by me
   Uniform_Grid_npoints = 2; */
   /* added by me
   FLDset_npoints(Uniform_Grid_id,Uniform_Grid_npoints); */

   Uniform_Grid_points = (float *)ARRalloc(NULL, DTYPE_FLOAT, 
                               Uniform_Grid_npoints, NULL);

   /*** fill in points array with values for 2 points: low left and high right corners of the mesh ***/
   Uniform_Grid_points[0] = X;
   Uniform_Grid_points[1] = Y;
   Uniform_Grid_points[2] = Z;
   Uniform_Grid_points[3] = X+(NX-1)*DX;
   Uniform_Grid_points[4] = Y+(NY-1)*DY;
   Uniform_Grid_points[5] = Z+(NZ-1)*DZ;

   FLDset_points (Uniform_Grid_id, Uniform_Grid_points, Uniform_Grid_npoints, 
                  OM_SET_ARRAY_FREE);
   /* set  Uniform_Grid node data */

   /* Get field id */
   Uniform_Grid_id = OMfind_subobj(ReadPFB_id, OMstr_to_name("Uniform_Grid"), OM_OBJ_RW);

   /* Set number of node data components */
   Uniform_Grid_ncomp = 1;
   FLDset_node_data_ncomp (Uniform_Grid_id, Uniform_Grid_ncomp);

   /* For each node data component set veclen, type and data arry itself */
   for (Uniform_Grid_comp_count=0; Uniform_Grid_comp_count < Uniform_Grid_ncomp; Uniform_Grid_comp_count++) {

      /* Set veclen, assign Uniform_Grid_veclen before next call */
      FLDget_node_data_veclen (Uniform_Grid_id,Uniform_Grid_comp_count,&Uniform_Grid_veclen);
      /* just a scalar at each point */
      Uniform_Grid_veclen = 1;
      FLDset_node_data_veclen (Uniform_Grid_id,Uniform_Grid_comp_count,Uniform_Grid_veclen);

      /* Set data array */
      /* data_type should be set to one of the following: 
         DTYPE_BYTE, DTYPE_CHAR, DTYPE_SHORT, 
         DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE) */

      Uniform_Grid_data_type = DTYPE_FLOAT;

      /* allocate Uniform_Grid_data array first */
      /* assume float array and Uniform_Grid_ndata is set to number of nodes */
      Uniform_Grid_ndata = NX*NY*NZ;

      Uniform_Grid_data = (float *)ARRalloc(NULL, Uniform_Grid_data_type, 
                                 Uniform_Grid_veclen*Uniform_Grid_ndata, NULL);

      if (Uniform_Grid_comp_count == 0) {
	 /* read in the sub-grid data and put it into the field */
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
		   int I, J, K;

		   I = ix+i;
		   J = iy+j;
		   K = iz+k;

		   tools_ReadDouble(fp, &value, 1);
		   /* Uniform_Grid_data[I*NY*NZ + J*NZ + K] = (float)value; */
		   Uniform_Grid_data[K*NX*NY + J*NX + I] = (float)value;
		 }
       
	 }
      }

      FLDset_node_data (Uniform_Grid_id, Uniform_Grid_comp_count, (char *)Uniform_Grid_data, Uniform_Grid_data_type,
                        Uniform_Grid_ndata*Uniform_Grid_veclen, OM_SET_ARRAY_FREE);

      /*  Other useful calls:

          FLDset_node_data_id()
          FLDset_node_null_data()
          FLDset_node_data_label()
      */
   }
   

   fclose(fp);

   /*************************/
   /*  Free input variables */
   /*************************/
   if (filename)
      free(filename);

   return(1);
}
