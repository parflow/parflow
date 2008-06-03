#include "user.h"

#define ERR_RETURN(A) {ERRerror("update",1,ERR_ORIG, A); return(0);}


/* This is a crude and non-robust way to set an object's transform from the
   given perpendicular axis to the XY plane.  It is used in Orthograph to
   ensure plots are in the right plane.*/
int
calculate(OMobj_id Mesh_plot_2D_xform_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   int  axis;
   OMobj_id in_id;
   OMobj_id Xform_id;
   OMobj_id inmat_id;
   OMobj_id outmat_id;
   OMobj_id id;
   float *inmat = NULL;
   float *outmat = NULL;
   int i, j;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get axis's value */ 
   if (OMget_name_int_val(Mesh_plot_2D_xform_id, OMstr_to_name("axis"), &axis) != 1) 
      axis = -1;

   if (axis < 0 || axis > 2)
      ERR_RETURN("invalid axis variable");

   /* Get in mesh */

   /* Get mesh id */
   in_id = OMfind_subobj(Mesh_plot_2D_xform_id, OMstr_to_name("in"), OM_OBJ_RD);
   if (OMis_null_obj(in_id))
      ERR_RETURN("can't find in in object");
   Xform_id = OMfind_subobj(in_id, OMstr_to_name("xform"), OM_OBJ_RD);
   if (OMis_null_obj(Xform_id))
      ERR_RETURN("can't find xform in in");
   inmat_id = OMfind_subobj(Xform_id, OMstr_to_name("mat"), OM_OBJ_RD);
   if (OMis_null_obj(inmat_id))
      ERR_RETURN("can't find (input) mat in xform");

   inmat = (float *)OMret_array_ptr(inmat_id,OM_GET_ARRAY_RD,NULL,NULL);
   if (inmat == NULL)
      ERR_RETURN("can't get inmat array");

   id = OMfind_subobj(Mesh_plot_2D_xform_id, OMstr_to_name("Xform"), OM_OBJ_RW);
   if (OMis_null_obj(id))
      ERR_RETURN("can't find Xform in object");
   id = OMfind_subobj(id, OMstr_to_name("xform"), OM_OBJ_RW);
   if (OMis_null_obj(id))
      ERR_RETURN("can't find xform in Xform");
   outmat_id = OMfind_subobj(id, OMstr_to_name("mat"), OM_OBJ_RW);
   if (OMis_null_obj(outmat_id))
      ERR_RETURN("can't find (output) mat in xform");

   outmat = (float *)OMret_array_ptr(outmat_id,OM_GET_ARRAY_WR,NULL,NULL);
   if (outmat == NULL)
      ERR_RETURN("can't get outmat array");

   for (i = 0; i < 4*4; i++)
      outmat[i] = 0.0;

   if (axis == 0) {
      outmat[0] = inmat[1];
      outmat[5] = inmat[6];
      outmat[10] = inmat[8];
   }
   else if (axis == 1) {
      outmat[0] = inmat[0];
      outmat[5] = inmat[6];
      outmat[10] = -inmat[9];
   }
   else if (axis == 2) {
      for (i = 0; i < 4*4; i++)
	 outmat[i] = inmat[i];
   }
   outmat[15] = inmat[15];

   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: calculate generated from method: Mesh_plot_2D_xform.update\n"); */


   /*************************/
   /*  Free input variables */
   /*************************/
   if (inmat)
       ARRfree((char *)inmat);
   if (outmat)
       ARRfree((char *)outmat);

   return(1);
}
