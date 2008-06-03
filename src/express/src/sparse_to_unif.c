#include "user.h"
#include <string.h>
#include <avs/err.h>

#define ERR_RETURN(A) {ERRerror("read",1,ERR_ORIG, A);return(0);}
#define my_rint(f) ((int)((f)+0.5))


/* This function converts a sparse (.pfsb) field to a uniform field.  If the
   variable orig_dims is present in the sparse field, that is used for the
   uniform field dimensions. */

int
sparse_to_unif(OMobj_id sparse_to_unif_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   OMobj_id in_id;
   int in_ndim, *in_dims, in_nspace, in_nnodes;
   float *in_coord;
   int in_ncomp, in_comp_count, in_veclen;
   int    in_data_type, in_ndata;
   float  *in_data = NULL;
   double  zeroval;
   OMobj_id out_id;
   int out_ndim, *out_dims, out_nspace, out_npoints;
   float *out_points;
   int out_ncomp, out_comp_count, out_veclen;
   int    out_data_type, out_ndata;
   float *out_data;
   int	 *orig_dims;
   float min_extent[3], max_extent[3];
   float D[3]; /* dx, dy, dz; to be calculated */
   int	 i;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get in structured mesh */

   /* Get mesh id */
   in_id = OMfind_subobj(sparse_to_unif_id, OMstr_to_name("in"), OM_OBJ_RD);

   /* Get mesh dims array */
   FLDget_dims (in_id, &in_dims, &in_ndim);

   /* Get mesh nspace */
   FLDget_nspace (in_id, &in_nspace);

   /* Get mesh coordinates */
   if (FLDget_coord (in_id, &in_coord, &in_nnodes, OM_GET_ARRAY_RD) != 1)
      ERR_RETURN("can't get coords for in_id");

   /* Get in's node data */

   /* Get field id */
   in_id = OMfind_subobj(sparse_to_unif_id, OMstr_to_name("in"), OM_OBJ_RD);

   /* Get number of node data components */
   FLDget_node_data_ncomp (in_id, &in_ncomp);

   /* For each node data component get veclen, type and data arry itself */
   for (in_comp_count=0; in_comp_count < in_ncomp; in_comp_count++) {

      if (in_comp_count == 0) {
	 /* Get veclen */
	 if (FLDget_node_data_veclen (in_id, in_comp_count, &in_veclen) != 1)
	    ERR_RETURN("unable to get in_id node data");

	 /* Get data array and data_type which is one of the following: 
	    DTYPE_BYTE, DTYPE_CHAR, DTYPE_SHORT, 
	    DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE */

	 if (FLDget_node_data (in_id, in_comp_count, &in_data_type, &in_data,
			   &in_ndata, OM_GET_ARRAY_RD) != 1)
	    ERR_RETURN("unable to get in_id node data");

	 /* in_data array is freed at end of function */
      }
   }
   
   /* Get dimensions of original grid */
   orig_dims = (int *)OMret_name_array_ptr(in_id,OMstr_to_name("orig_dims"),OM_GET_ARRAY_RD,NULL,NULL);

   if (orig_dims == NULL) {
      printf("orig_dims not found in group, assuming 10x10x10\n");

      orig_dims = (int *)ARRalloc(NULL,DTYPE_INT,3,NULL);
      if (orig_dims == NULL)
	 ERR_RETURN("can't allocate memory for orig_dims");

      orig_dims[0] = orig_dims[1] = orig_dims[2] = 10;
   }


   /* get min and max extents of the original grid */
   if (FLDget_coord_extent(in_id,min_extent,max_extent) != 1)
      ERR_RETURN("can't get coord extent for in_id");

   D[0] = (max_extent[0]-min_extent[0])/(orig_dims[0]-1);
   D[1] = (max_extent[1]-min_extent[1])/(orig_dims[1]-1);
   D[2] = (max_extent[2]-min_extent[2])/(orig_dims[2]-1);


   /* Get zeroval's value */
   if (OMget_name_real_val(sparse_to_unif_id, OMstr_to_name("zeroval"), &zeroval) != 1)
      zeroval = 0.0;


   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: sparse_to_unif generated from method: sparse_to_unif.update\n"); */


   /***********************/
   /*  Set output values  */
   /***********************/
    /* Set out uniform mesh */

   /*  Get mesh id */
   out_id = OMfind_subobj(sparse_to_unif_id, OMstr_to_name("out"), OM_OBJ_RW);

   /* Set mesh dimensionality, out_ndim can be 1,2 or 3 */
   out_ndim = 3;
   FLDset_ndim (out_id, out_ndim);

   /* Set mesh dims array */
   out_dims = (int *)ARRalloc(NULL, DTYPE_INT, 
                           out_ndim, NULL);

   /*** fill in dims array with your values ***/
   for (i = 0; i < out_ndim; i++)
      out_dims[i] = orig_dims[i];

   FLDset_dims (out_id, out_dims);
   /* out_dims freed at end of function */

   /* Set mesh nspace, out_nspace can be 1,2 or 3 */
   out_nspace = 3;
   FLDset_nspace (out_id, out_nspace);

   /* Set mesh extents */
   out_npoints = 2*out_nspace;
   out_points = (float *)ARRalloc(NULL, DTYPE_FLOAT, 
                               out_npoints, NULL);
   
   out_points[0] = min_extent[0];
   out_points[1] = min_extent[1];
   out_points[2] = min_extent[2];
   out_points[3] = max_extent[0];
   out_points[4] = max_extent[1];
   out_points[5] = max_extent[2];

   FLDset_points (out_id, out_points, out_npoints, 
                  OM_SET_ARRAY_FREE);
   /* set  out node data */

   /* Get field id */
   out_id = OMfind_subobj(sparse_to_unif_id, OMstr_to_name("out"), OM_OBJ_RW);

   /* Set number of node data components */
   out_ncomp = 1;
   FLDset_node_data_ncomp (out_id, out_ncomp);

   /* For each node data component set veclen, type and data arry itself */
   for (out_comp_count=0; out_comp_count < out_ncomp; out_comp_count++) {

      if (out_comp_count == 0) {
	 /* Set veclen, assign out_veclen before next call */
	 /* FLDget_node_data_veclen (out_id,out_comp_count,&out_veclen); */
	 out_veclen = 1;
	 FLDset_node_data_veclen (out_id,out_comp_count,out_veclen);

	 /* Set data array */
	 /* data_type should be set to one of the following: 
	    DTYPE_BYTE, DTYPE_CHAR, DTYPE_SHORT, 
	    DTYPE_INT, DTYPE_FLOAT, DTYPE_DOUBLE) */

	 out_data_type = DTYPE_FLOAT;

	 out_ndata = out_dims[0]*out_dims[1]*out_dims[2];

	 /* allocate out_data array first */
	 /* assume float array and out_ndata is set to number of nodes */

	 out_data = (float *)ARRalloc(NULL, out_data_type, 
				    out_veclen*out_ndata, NULL);

	 if (out_data == NULL)
	    ERR_RETURN("can't allocate memory for out_data");

	 /* "zero" out the data array */
	 for (i = 0; i < out_veclen*out_ndata; i++)
	    out_data[i] = zeroval;

	 for (i = 0; i < in_nnodes; i += 3) {
	    int I,J,K;

	    I = (int)my_rint((in_coord[i+0]-min_extent[0])/D[0]);
	    J = (int)my_rint((in_coord[i+1]-min_extent[1])/D[1]);
	    K = (int)my_rint((in_coord[i+2]-min_extent[2])/D[2]);

	    out_data[K*out_dims[0]*out_dims[1] + J*out_dims[0] + I] = in_data[i/3];
	    /*
	       out_data[I*out_dims[1]*out_dims[2] + J*out_dims[2] + K] = in_data[i/3];
	    */
	 }
	 
	 FLDset_node_data (out_id, out_comp_count, out_data, out_data_type,
			   out_ndata*out_veclen, OM_SET_ARRAY_FREE);
      }

   }


   /*************************/
   /*  Free input variables */
   /*************************/
   if (in_data)
      ARRfree((char *)in_data);
   if (in_dims)
      ARRfree((char *)in_dims);
   if (in_coord)
      ARRfree((char *)in_coord);
   if (out_dims)
      ARRfree((char *)out_dims);

   return(1);
}
