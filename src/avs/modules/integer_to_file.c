/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * IntegerToFile
 *
 * AVS module to convert integers to file names.
 *
 *****************************************************************************/

#include <stdio.h>
#include <string.h>

#include <avs.h>
#include <field.h>

/*--------------------------------------------------------------------------
 * IntegerToFile
 *--------------------------------------------------------------------------*/

IntegerToFile()
{
   int IntegerToFile_compute();
   int p;


   AVSset_module_name("integer to file", MODULE_FILTER);

   AVScreate_input_port("Integer", "integer", REQUIRED);

   AVScreate_output_port("File", "string");

   p = AVSadd_parameter("File Prefix", "string", "", "", "");
   AVSconnect_widget(p, "browser");

   p = AVSadd_parameter("Format", "string",
			"%s%04d.ext", "", "");
   AVSconnect_widget(p, "typein");

   AVSset_compute_proc(IntegerToFile_compute);
}

	
/*--------------------------------------------------------------------------
 * IntegerToFile_compute
 *--------------------------------------------------------------------------*/

IntegerToFile_compute(integer, file, file_prefix, format_string)
int       integer;
char    **file;
char     *file_prefix;
char     *format_string;
{
   int    sz;


   if ( AVSinput_changed("Integer", 0) )
   {
      /* free old memory */
      if (*file)
	 free(*file);

      /* assume that the integer part of `format_string'  */
      /* specifies no more than 16 digits                 */
      sz = strlen(file_prefix) + 16;

      *file = malloc(sz * sizeof(char));
      sprintf(*file, format_string, file_prefix, integer);
   }
   else
   {
      AVSmark_output_unchanged("File");
   }

   return(1);
}
