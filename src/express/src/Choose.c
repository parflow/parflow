#include "user.h"
#include <string.h>

#include <avs/err.h>

#define ERR_RETURN(A) {if (input_file) free(input_file); ERRerror("read",1,ERR_ORIG, A); return(0);}


/* Outputs an integer which tells whether the input file is a .pfsb file or a
   .pfb file (0 == .pfsb, 1 == .pfb).  Sets the corresponding output string
   for that type of file, and unsets the non-corresponding string. */
int
choose_type(OMobj_id choose_file_id, OMevent_mask event_mask, int seq_num)
{
   char *input_file = NULL;
   char *ext = NULL;
   char *pfsb_file = NULL;
   char *pfb_file = NULL;
   OMobj_id choice_id;
   int status;

   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: choose_type generated from method: choose_file_type.choose\n"); */

   if (OMget_name_str_val(choose_file_id, OMstr_to_name("input_file"), &input_file, 0) != 1)
      input_file = NULL;
   if (input_file == NULL) {
      /* ERR_RETURN("read_parflow: no filename given"); */
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfb_file"), NULL) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfb_file to null");
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfsb_file"), NULL) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfsb_file to null");
      /* Code to set choice to unset state.  Not used because choice needs
	 to be set for field types to be propagated out of the chooser.

	 choice_id = OMfind_subobj(choose_file_id,OMstr_to_name("choice"),OM_OBJ_RW);
	 if (OMis_null_obj(choice_id))
	    ERR_RETURN("read_parflow: unable to find int variable choice");
	 if (OMset_obj_val(choice_id, OMnull_obj, 0) != 1)
	    ERR_RETURN("read_parflow: unable to set int variable choice");
      */
      if (OMset_name_int_val(choose_file_id, OMstr_to_name("choice"), 1) != 1)
	 ERR_RETURN("read_parflow: unable to set int variable choice");
      return(1);
   }

   ext = strrchr(input_file,'.');
   if (ext == NULL)
      ERR_RETURN("read_parflow: unknown file type");

   if (strcmp(ext,".pfsb") == 0) {
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfb_file"), NULL) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfb_file to null");
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfsb_file"), input_file) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfsb_file");
      if (OMset_name_int_val(choose_file_id, OMstr_to_name("choice"), 0) != 1)
	 ERR_RETURN("read_parflow: unable to set int variable choice");
   }
   else if (strcmp(ext,".pfb") == 0) {
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfsb_file"), NULL) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfsb_file to null");
      if (OMset_name_str_val(choose_file_id, OMstr_to_name("pfb_file"), input_file) != 1)
	 ERR_RETURN("read_parflow: unable to set string variable pfb_file");
      if (OMset_name_int_val(choose_file_id, OMstr_to_name("choice"), 1) != 1)
	 ERR_RETURN("read_parflow: unable to set int variable choice");
   }
   else
      ERR_RETURN("read_parflow: unknown file type");

   free(input_file);
   return(1);
}
