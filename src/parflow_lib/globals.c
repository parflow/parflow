/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routines for manipulating global structures.
 *
 *****************************************************************************/

#define PARFLOW_GLOBALS

#include "parflow.h"
#include "globals.h"


/*--------------------------------------------------------------------------
 * NewGlobals
 *--------------------------------------------------------------------------*/

void   NewGlobals(run_name)
char  *run_name;
{
   globals = ctalloc(Globals, 1);

   sprintf(GlobalsRunName,     "%s",    run_name);
   sprintf(GlobalsInFileName,  "%s.%s", run_name, "pfidb");
   sprintf(GlobalsOutFileName, "%s.%s", run_name, "out");
}


/*--------------------------------------------------------------------------
 * FreeGlobals
 *--------------------------------------------------------------------------*/

void  FreeGlobals()
{
   tfree(globals);
}

/*--------------------------------------------------------------------------
 * LogGlobals
 *--------------------------------------------------------------------------*/

void  LogGlobals()
{
   IfLogging(0)
   {
      FILE *log_file;

      log_file = OpenLogFile("Globals");

      fprintf(log_file, "Run Name: %s\n",
                   GlobalsRunName);
      fprintf(log_file, "Logging Level = %d\n",
                   GlobalsLoggingLevel);
      fprintf(log_file, "Num processes = %d\n",
                   GlobalsNumProcs);
      fprintf(log_file, "Process grid = (%d,%d,%d)\n",
                   GlobalsNumProcsX,
                   GlobalsNumProcsY,
                   GlobalsNumProcsZ);

      CloseLogFile(log_file);
   }
}


