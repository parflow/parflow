/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * NewLogging, FreeLogging, OpenLogFile, CloseLogFile
 *
 * Routines for printing vectors to standard out.
 *
 *****************************************************************************/

#include "parflow.h"
#include "version.h"

/*--------------------------------------------------------------------------
 * NewLogging
 *--------------------------------------------------------------------------*/

void   NewLogging()
{
   FILE  *log_file;

   char   filename[MAXPATHLEN];

   GlobalsLoggingLevel = 1;

   if(!amps_Rank(amps_CommWorld))
   {
      sprintf(filename, "%s.%s", GlobalsOutFileName, "log");

      if ((log_file = fopen(filename, "w")) == NULL)
      {
	 InputError("Error: can't open output file %s%s\n", filename, "");
      }
      
      fprintf(log_file, "*****************************************************************************\n");
      fprintf(log_file, "ParFlow Output Log\n");
      fprintf(log_file, "\t\t%s\n", ParFlowRevision);
      fprintf(log_file, "\tCompiled on    : %s %s\n", __DATE__, __TIME__);
#ifdef CFLAGS
      fprintf(log_file, "\tWith C flags   : %s\n", CFLAGS);
#endif

#ifdef FFLAGS
      fprintf(log_file, "\tWith F77 flags : %s\n", FFLAGS);
#endif
      fprintf(log_file, "*****************************************************************************\n");
      
      fclose(log_file);
   }
}


/*--------------------------------------------------------------------------
 * FreeLogging
 *--------------------------------------------------------------------------*/

void   FreeLogging()
{
}


/*--------------------------------------------------------------------------
 * OpenLogFile
 *--------------------------------------------------------------------------*/

FILE *OpenLogFile(module_name)
char       *module_name;
{
   FILE *log_file;

   char       filename[256];


   sprintf(filename, "%s.%s", GlobalsOutFileName, "log");

   if(!amps_Rank(amps_CommWorld))
   {
      log_file = amps_SFopen(filename, "a");

      fprintf(log_file, "==============================================\n");
      fprintf(log_file, "BEGIN Log Info (%s)\n", module_name);
      fprintf(log_file, "\n");

      return  log_file;
   }
   else
      return NULL;
}


/*--------------------------------------------------------------------------
 * CloseLogFile
 *--------------------------------------------------------------------------*/

int  CloseLogFile(log_file)
FILE *log_file;
{
   if(!amps_Rank(amps_CommWorld))
   {
      fprintf(log_file, "\n");
      fprintf(log_file, "END Log Info\n");
      fprintf(log_file, "==============================================\n");

      return  fclose(log_file);
   }
   else
      return 0;
}


