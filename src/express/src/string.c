#include <string.h>
#include <avs/om.h>
#include <avs/om_att.h>
#include "user.h"

/* Some helpful string functions */

/* string_concat is not really necessary, as you can just as well use the
   V expression "string1 + string2" */
int
string_concat(OMobj_id strcat_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *string1 = NULL;
   char  *string2 = NULL;
   char  *concat = NULL;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get string1's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string1"), &string1, 0) != 1)
      string1 = NULL;

   /* Get string2's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string2"), &string2, 0) != 1)
      string2 = NULL;


   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: string_concat generated from method: strcat.update\n"); */


   /***********************/
   /*  Set output values  */
   /***********************/
   /* Set concat's value */
   if (string1 && string2) {
      concat = malloc(strlen(string1)+strlen(string2)+1);
      strcpy(concat,string1);
      strcat(concat,string2);
      OMset_name_str_val(strcat_id, OMstr_to_name("concat"), concat);
      if (concat)
	 free(concat);
   }
   else {
      OMset_name_str_val(strcat_id, OMstr_to_name("concat"), NULL);
   }

   /*************************/
   /*  Free input variables */
   /*************************/
   if (string1)
      free(string1);

   if (string2)
      free(string2);

   return(1);
}


/* splits string1 at the first occurrence of string2[0].  outstring1 contains
   the "left" part of string1 up to the character before string2[0], and
   outstring2 containts the "right" part of string1 after string2[0] */
int
string_split_fwd(OMobj_id strcat_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *string1 = NULL;
   char  *string2 = NULL;
   char  *outstring1 = NULL;
   char  *outstring2 = NULL;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get string1's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string1"), &string1, 0) != 1)
      string1 = NULL;

   /* Get string2's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string2"), &string2, 0) != 1)
      string2 = NULL;


   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: string_chr generated from method: strchr.update\n"); */


   /***********************/
   /*  Set output values  */
   /***********************/
   /* Set outstring's value */
   if (string1 && string2) {
      outstring1 = string1;
      outstring2 = strchr(string1,string2[0]);
      if (outstring2 && *outstring2)
	 *outstring2++ = '\0'; /* pretty tricky, eh? */
   }
   else {
      outstring1 = NULL;
      outstring2 = NULL;
   }
   OMset_name_str_val(strcat_id, OMstr_to_name("outstring1"), outstring1);
   OMset_name_str_val(strcat_id, OMstr_to_name("outstring2"), outstring2);

   /*************************/
   /*  Free input variables */
   /*************************/
   if (string1)
      free(string1);

   if (string2)
      free(string2);

   return(1);
}


/* splits string1 at the last occurrence of string2[0].  outstring1 contains
   the "left" part of string1 up to the character before string2[0], and
   outstring2 containts the "right" part of string1 after string2[0] */
int
string_split_rev(OMobj_id strcat_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *string1 = NULL;
   char  *string2 = NULL;
   char  *outstring1 = NULL;
   char  *outstring2 = NULL;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get string1's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string1"), &string1, 0) != 1)
      string1 = NULL;

   /* Get string2's value */
   if (OMget_name_str_val(strcat_id, OMstr_to_name("string2"), &string2, 0) != 1)
      string2 = NULL;


   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: string_chr generated from method: strchr.update\n"); */


   /***********************/
   /*  Set output values  */
   /***********************/
   /* Set outstring's value */
   if (string1 && string2) {
      outstring1 = string1;
      outstring2 = strrchr(string1,string2[0]);
      if (outstring2 && *outstring2)
	 *outstring2++ = '\0'; /* pretty tricky, eh? */
      else if (!outstring2) {
	 outstring2 = outstring1;
	 outstring1 = NULL;
      }
   }
   else {
      outstring1 = NULL;
      outstring2 = NULL;
   }
   OMset_name_str_val(strcat_id, OMstr_to_name("outstring1"), outstring1);
   OMset_name_str_val(strcat_id, OMstr_to_name("outstring2"), outstring2);

   /*************************/
   /*  Free input variables */
   /*************************/
   if (string1)
      free(string1);

   if (string2)
      free(string2);

   return(1);
}


/* splits string1 at all occurrences of string2, and outputs an array of
   these substrings and the number of them. */
int
string_split_all(OMobj_id strsplit_id, OMevent_mask event_mask, int seq_num)
{
   /***********************/
   /*  Declare variables  */
   /***********************/
   char  *string1 = NULL;
   char  *string2 = NULL;
   char  **array = NULL;
   OMobj_id array_id;
   int  array_size = 0, array_count;
   int	strings;
   int	i;
   char	*ptr, *nextptr;

   /***********************/
   /*  Get input values   */
   /***********************/
   /* Get string1's value */
   if (OMget_name_str_val(strsplit_id, OMstr_to_name("string1"), &string1, 0) != 1)
      string1 = NULL;

   /* Get string2's value */
   if (OMget_name_str_val(strsplit_id, OMstr_to_name("string2"), &string2, 0) != 1)
      string2 = NULL;

   if (OMget_name_int_val(strsplit_id, OMstr_to_name("strings"), &strings) != 1)
      strings = 0;

   /***********************/
   /* Function's Body     */
   /***********************/
   /* ERRverror("",ERR_NO_HEADER | ERR_INFO,"I'm in function: string_split generated from method: strsplit.update\n"); */

   array_id = OMfind_subobj(strsplit_id, OMstr_to_name("array"), OM_OBJ_RW);

   if (string1 && string2) {
      array_size = 0;
      for (ptr = string1; ptr; ptr = strstr(ptr,string2)) {
	 if (array_size != 0)
	    ptr += strlen(string2);
	 array_size++;
      }

      array = (char **)malloc(array_size*sizeof(char*));
      array_count = 0;
      for (ptr = string1; ptr; ) {
	 if (array_count != 0)
	    ptr += strlen(string2);
	 nextptr = strstr(ptr,string2);

	 if (nextptr) {
	    char c = *nextptr;

	    *nextptr = '\0';
	    array[array_count++] = strdup(ptr);
	    *nextptr = c;
	 }
	 else {
	    array[array_count++] = strdup(ptr);
	 }

	 ptr = nextptr;
      }

      /* OMset_array_size(array_id, array_size); */
      if (OMset_name_int_val(strsplit_id, OMstr_to_name("strings"), array_size) != 1) {
	 printf("unable to set strings");
	 return(1);
      }
      for (array_count = 0; array_count < array_size;  array_count++) {
	 OMset_str_array_val(array_id, array_count, array[array_count]);
      }
   }
   else {
      if (OMset_name_int_val(strsplit_id, OMstr_to_name("strings"), 0) != 1) {
	 printf("unable to set strings");
	 return(1);
      }
      /* I have no idea how to unset the array dimension; the OMget_obj_prop
	 call returns a null object.

	 OMobj_id dims_id;
	 int dims_type;

	 dims_id = OMget_obj_prop(array_id, OM_prop_dimensions, 0);
	 if (OMis_null_obj(dims_id)) {
	    printf("dims_id is null\n");
	    return(0);
	 }

	 OMget_data_type(dims_id, &dims_type);

	 if (dims_type == OM_TYPE_INT)
	    printf("you got it!\n");
	 else
	    printf("sorry, try again later\n");
      */

      /* OMset_obj_val(dims_id, OMnull_obj, 0); */
   }

   for (array_count = 0; array_count < array_size;  array_count++)
      free(array[array_count]);

   free(array);


   /*************************/
   /*  Free input variables */
   /*************************/
   if (string1)
      free(string1);

   if (string2)
      free(string2);

   return(1);
}
