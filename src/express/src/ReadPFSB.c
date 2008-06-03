#include "user.h"
#include <stdio.h>
#include <string.h>

#include <avs/err.h>
#include <avs/fld.h>

#include <../../tools/tools_io.h>
 
#define ERR_RETURN(A) {ERRerror("read",1,ERR_ORIG, A); return(0);}


/* data for each subgrid */
typedef struct
{
   int    nnz;
   float  *x, *y, *z, *data;
} SubgridData;


/* Reads data from the .pfsb file, allocates an array of SubgridData structs,
   and puts the data into them.  Also calculates and passes useful global
   data. */
SubgridData* read_data(FILE *fp, int *subgrids, double *XYZ, int *N, double *D,
		       int *nnz)
{
   double d[3];
   int i[9];
   int s;
   SubgridData *subgrid;

   *nnz = 0;

   tools_ReadDouble(fp, XYZ, 3);
   tools_ReadInt(fp, N, 3);
   tools_ReadDouble(fp, D, 3);

   tools_ReadInt(fp, subgrids, 1);

   subgrid = (SubgridData *) malloc(*subgrids * sizeof(SubgridData));

   for (s = 0; s < *subgrids; s++) {
      int subgrid_nnz;
      int n;

      tools_ReadInt(fp, i, 9);
      tools_ReadInt(fp, &subgrid_nnz, 1);
      subgrid[s].nnz = subgrid_nnz;
      *nnz += subgrid_nnz;

      subgrid[s].x = (float *) malloc(subgrid_nnz * sizeof(float));
      subgrid[s].y = (float *) malloc(subgrid_nnz * sizeof(float));
      subgrid[s].z = (float *) malloc(subgrid_nnz * sizeof(float));
      subgrid[s].data = (float *) malloc(subgrid_nnz * sizeof(float));

      for (n = 0; n < subgrid_nnz; n++) {
	 int ijk[3];
	 double data;

	 tools_ReadInt(fp, ijk, 3);
	 subgrid[s].x[n] = (float) (XYZ[0] + ijk[0]*D[0]);
	 subgrid[s].y[n] = (float) (XYZ[1] + ijk[1]*D[1]);
	 subgrid[s].z[n] = (float) (XYZ[2] + ijk[2]*D[2]);

	 tools_ReadDouble(fp, &data, 1);
	 subgrid[s].data[n] = (float) data;
      }
   }

   return (subgrid);
}


/* read a .pfsb file into a structured grid.  Also includes an integer array,
   'orig_dims', which contains the grid dimensions from the file.  (this is
   used in the sparse_to_unif module) */
int
ReadPFSB(OMobj_id ReadPFSB_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *filename = NULL;
   OMobj_id NonUniform_Grid_id;
   int NonUniform_Grid_ndim, *NonUniform_Grid_dims, NonUniform_Grid_nspace, NonUniform_Grid_nnodes;
   float *NonUniform_Grid_coord;
   int NonUniform_Grid_ncomp, NonUniform_Grid_comp_count, NonUniform_Grid_veclen;
   int    NonUniform_Grid_data_type, NonUniform_Grid_ndata;
   float *NonUniform_Grid_data;

   /* my variables */
   double       XYZ[3];
   int          N[3];
   double       D[3];

   float	min_extent[3], max_extent[3];
   int		*orig_dims;

   int		nnz, subgrids;
   int		n, s;
   int		nnz_count;
   int		i;
 
   SubgridData *subgrid;
 
   FILE* fp;


   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get filename's value */
   if (OMget_name_str_val(ReadPFSB_id, OMstr_to_name("filename"), &filename, 0) != 1)
      filename = NULL;
   if (filename == NULL)
      ERR_RETURN("ReadPFSB: filename not given");
   if (strcmp(strrchr(filename,'.'),".pfsb") != 0)
      ERR_RETURN("ReadPFSB: not a .pfsb file");



   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: ReadPFSB generated from method: ReadPFSB.ReadFile\n"); */


   /***********************/
   /*  Set output values  */
   /***********************/
   /* Set NonUniform_Grid structured mesh */

   /* Get mesh id */
   NonUniform_Grid_id = OMfind_subobj(ReadPFSB_id, OMstr_to_name("NonUniform_Grid"), OM_OBJ_RW);

   /* Set mesh dimensionality, NonUniform_Grid_ndim can be 1,2 or 3 */
   NonUniform_Grid_ndim = 1;
   FLDset_ndim (NonUniform_Grid_id, NonUniform_Grid_ndim);


   fp = fopen(filename,"r");
   if (fp == NULL)
      ERR_RETURN("ReadPFSB: cannot open file");

   /*-----------------------------------------------------------------------*
    * Read the data                                                         *
    *-----------------------------------------------------------------------*/
 
   subgrid = read_data(fp,&subgrids,XYZ,N,D,&nnz);
   fclose(fp);

   /* Set mesh dims array */
   NonUniform_Grid_dims = (int *)ARRalloc(NULL, DTYPE_INT, NonUniform_Grid_ndim, NULL);

   /*** fill in dims array with your values ***/
   NonUniform_Grid_dims[0] = nnz;

   FLDset_dims (NonUniform_Grid_id, NonUniform_Grid_dims);
   if (NonUniform_Grid_dims)
      ARRfree((char *)NonUniform_Grid_dims);

   /* get orig_dims array to put grid dimensions in */
   orig_dims = (int *)OMret_name_array_ptr(NonUniform_Grid_id,OMstr_to_name("orig_dims"),OM_GET_ARRAY_WR,NULL,NULL);
   if (orig_dims == NULL)
      ERR_RETURN("ReadPFSB: can't get orig_dims array");

   orig_dims[0] = N[0];
   orig_dims[1] = N[1];
   orig_dims[2] = N[2];

   if (orig_dims)
      ARRfree(orig_dims);

   /* Set mesh nspace, NonUniform_Grid_nspace can be 1,2 or 3 */
   NonUniform_Grid_nspace = 3;
   FLDset_nspace (NonUniform_Grid_id, NonUniform_Grid_nspace);

   FLDset_npoints (NonUniform_Grid_id, 2);

   /* Set mesh coordinates */
   /* first allocate NonUniform_Grid_coord array */
   FLDget_nnodes (NonUniform_Grid_id, &NonUniform_Grid_nnodes);
   
   NonUniform_Grid_coord = (float *)ARRalloc(NULL, DTYPE_FLOAT, 
                              NonUniform_Grid_nspace*NonUniform_Grid_nnodes, NULL);

   /*** fill in NonUniform_Grid_coord array with X[,Y,Z] values at each node ***/
   nnz_count = 0;
   for (s = 0; s < subgrids; s++) {
      for (n = 0; n < subgrid[s].nnz; n++) {
	 NonUniform_Grid_coord[nnz_count*3  ] = subgrid[s].x[n];
	 NonUniform_Grid_coord[nnz_count*3+1] = subgrid[s].y[n];
	 NonUniform_Grid_coord[nnz_count*3+2] = subgrid[s].z[n];
	 /*
	    NonUniform_Grid_coord[nnz_count] = subgrid[s].x[n];
	    NonUniform_Grid_coord[NonUniform_Grid_nnodes+nnz_count] = subgrid[s].y[n];
	    NonUniform_Grid_coord[2*NonUniform_Grid_nnodes+nnz_count] = subgrid[s].z[n];
	 */
	 nnz_count++;
      }
   }

   FLDset_coord (NonUniform_Grid_id, NonUniform_Grid_coord, NonUniform_Grid_nspace*NonUniform_Grid_nnodes, 
                 OM_SET_ARRAY_FREE);

   /* set extents to be those of the true grid size */
   for (i = 0; i < 3; i++) {
      min_extent[i] = (float)XYZ[i];
      max_extent[i] = (float)(XYZ[i] + (N[i]-1)*D[i]);
   }
   FLDset_coord_extent(NonUniform_Grid_id, min_extent, max_extent,3);

   
   /* set  NonUniform_Grid node data */

   /* Get field id */
   NonUniform_Grid_id = OMfind_subobj(ReadPFSB_id, OMstr_to_name("NonUniform_Grid"), OM_OBJ_RW);

   /* Set number of node data components */
   NonUniform_Grid_ncomp = 1;
   FLDset_node_data_ncomp (NonUniform_Grid_id, NonUniform_Grid_ncomp);

   /* For each node data component set veclen, type and data arry itself */
   for (NonUniform_Grid_comp_count=0; NonUniform_Grid_comp_count < NonUniform_Grid_ncomp; NonUniform_Grid_comp_count++) {

      /* Set veclen, assign NonUniform_Grid_veclen before next call */
      FLDget_node_data_veclen (NonUniform_Grid_id,NonUniform_Grid_comp_count,&NonUniform_Grid_veclen);

      NonUniform_Grid_veclen = 1;
      FLDset_node_data_veclen (NonUniform_Grid_id,NonUniform_Grid_comp_count,NonUniform_Grid_veclen);

      /* Set data array */
      /* data_type should be set to one of the following: 
         DTYPE_BYTE, DTYPE_CHAR, DTYPE_SHORT, 
         DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE) */

      NonUniform_Grid_data_type = DTYPE_FLOAT;

      /* allocate NonUniform_Grid_data array first */
      /* assume float array and NonUniform_Grid_ndata is set to number of nodes */
      NonUniform_Grid_ndata = nnz;

      NonUniform_Grid_data = (float *)ARRalloc(NULL, NonUniform_Grid_data_type, 
                                 NonUniform_Grid_veclen*NonUniform_Grid_ndata, NULL);

      if (NonUniform_Grid_comp_count == 0) {
	 nnz_count = 0;
	 for (s = 0; s < subgrids; s++) {
	    for (n = 0; n < subgrid[s].nnz; n++) {
	       NonUniform_Grid_data[nnz_count++] = subgrid[s].data[n];
	    }
	 }
      }

      FLDset_node_data (NonUniform_Grid_id, NonUniform_Grid_comp_count, (char *) NonUniform_Grid_data, NonUniform_Grid_data_type,
                        NonUniform_Grid_ndata*NonUniform_Grid_veclen, OM_SET_ARRAY_FREE);

      /*  Other useful calls:

          FLDset_node_data_id()
          FLDset_node_null_data()
          FLDset_node_data_label()
      */
   }
   

   /* free subgrid array */
   for (s = 0; s < subgrids; s++) {
      free(subgrid[s].x);
      free(subgrid[s].y);
      free(subgrid[s].z);
      free(subgrid[s].data);
   }
   free(subgrid);
      

   /*************************/
   /*  Free input variables */
   /*************************/
   if (filename)
      free(filename);

   return(1);
}
