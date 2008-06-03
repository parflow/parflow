/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <stdio.h>
#include <sys/param.h>
#include <stdlib.h>

#ifdef AMPS_UNISTD
#include <unistd.h>
#else
#include <sys/unistd.h>
#endif

#include <sys/times.h>

#include "amps.h"

#ifdef AMPS_MALLOC_DEBUG
char amps_malloclog[MAXPATHLEN];
#endif

#ifndef AMPS_CPU_TICKS_PER_SEC
long AMPS_CPU_TICKS_PER_SEC;
#endif

int amps_size;
int amps_rank;

#ifdef AMPS_F2CLIB_FIX
int MAIN__()
{
}
#endif

/*===========================================================================*/
/**

Every {\em AMPS} program must call this function to initialize the
message passing environment.  This must be done before any other
{\em AMPS} calls.  \Ref{amps_Init} does a synchronization on all the
nodes and the host (if it exists).  {\bf argc} and {\bf argv} should be
the {\bf argc} and {\bf argv} that were passed to {\bf main}.  On some
ports command line arguments can be used to control the underlying
message passing system.  For example, on {\em Chameleon} the
{\bf -trace} flag can be use to create a log of communication events.

{\large Example:}
\begin{verbatim}
int main( int argc, char *argv)
{
   amps_Init(argc, argv);
   
   amps_Printf("Hello World");

   amps_Finalize();
}
\end{verbatim}

{\large Notes:}

@memo Initialize AMPS
@param argc Command line argument count [IN/OUT]
@param argv Command line argument array [IN/OUT]
@return 
*/
int amps_Init(int *argc, char **argv[])
{
#ifdef AMPS_MPI_SETHOME
   char *temp_path;
   int length;
#endif

   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int namelen;

   MPI_Init(argc, argv);
   MPI_Comm_size(MPI_COMM_WORLD, &amps_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &amps_rank);

#ifdef AMPS_STDOUT_NOBUFF
   setbuf (stdout, NULL);
#endif

#ifdef AMPS_BSD_TIME
   amps_clock_init();
#endif

#ifdef AMPS_MPI_SETHOME
   if( !amps_rank )
   {
      if( (temp_path = getenv("AMPS_EXE_DIR")) == NULL)
      {
	 printf("AMPS Error: can't get AMPS_EXE_DIR envirnment variabl\n");
	 exit(1);
      }

      length = strlen(temp_path)+1;
   }

   MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if(amps_rank)
   {
      temp_path = malloc(length);
   }

   MPI_Bcast(temp_path, length, MPI_CHAR, 0, MPI_COMM_WORLD);

   if(chdir(temp_path))
      printf("AMPS Error: can't set working directory to %s", temp_path);

   if(amps_rank)
   {
      free(temp_path);
   }

#endif

#ifdef AMPS_MALLOC_DEBUG
    malloc_logpath = amps_malloclog;
    sprintf(malloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif

#ifdef TIMING
#ifndef CRAY_TIME
   AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
#endif
#endif

#ifdef AMPS_PRINT_HOSTNAME
   MPI_Get_processor_name(processor_name,&namelen);

   printf("Process %d on %s\n", amps_rank, processor_name);
#endif

   return 0;
}  


