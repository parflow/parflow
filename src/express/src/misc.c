#include <stdio.h>
#include "user.h"

/* some miscellaneous functions... one to be exact */

/* this was for debugging matrices, to be able to see the whole matrix at
   once */
int
copy_mat(OMobj_id copy_mat_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   int  in_size = 0;
   float *in = NULL; 
   int  out_size = 0;
   float *out = NULL; 
   int i;

   /***********************/
   /*  Get input values   */
   /***********************/
   in = (float *)OMret_name_array_ptr(copy_mat_id, OMstr_to_name("in"), OM_GET_ARRAY_RD,
			&in_size, NULL);


   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: copy_mat generated from method: copy_mat.update\n"); */
   for (i = 0; i < in_size; i++) {
      printf("%f",in[i]);
      if (i % 4 == 3)
	 printf("\n");
      else
	 printf(" ");
   }
   printf("\n");


   /***********************/
   /*  Set output values  */
   /***********************/
   /*
    *   set number of elements in array: 
    *   out_size = ...
    *   allocate array out:
    */
   out_size = in_size;
   out = (float *)ARRalloc(NULL, DTYPE_FLOAT, out_size, NULL);
   /*
    *  fill in array out
    */
   for (i = 0; i < in_size; i++) {
      out[i] = in[i];
   }
   OMset_name_array(copy_mat_id, OMstr_to_name("out"), DTYPE_FLOAT, (void *)out, 
                    out_size, OM_SET_ARRAY_FREE);
   /*
    * alternatively, if the dimensions of out array are set at this point 
    * you can use call that allocates memory:
    * out = (float *)OMret_name_array_ptr(copy_mat_id,
    *   	   	OMstr_to_name("out"), OM_GET_ARRAY_RW,
    *			&out_size, NULL);
    */

   /*************************/
   /*  Free input variables */
   /*************************/
   if (in != NULL) 
      ARRfree((char *)in);

   return(1);
}
