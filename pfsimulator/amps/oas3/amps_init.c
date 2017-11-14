/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/

#include <stdio.h>
#include <sys/param.h>
#include <sys/param.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/times.h>

#include "amps.h"
MPI_Comm oas3Comm;
int dummy1_oas3 = 0;

int amps_mpi_initialized = FALSE;

#ifdef AMPS_MALLOC_DEBUG
char amps_malloclog[MAXPATHLEN];
#endif

#ifndef CRAY_TIME
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
 *
 * Every {\em AMPS} program must call this function to initialize the
 * message passing environment.  This must be done before any other
 * {\em AMPS} calls.  \Ref{amps_Init} does a synchronization on all the
 * nodes and the host (if it exists).  {\bf argc} and {\bf argv} should be
 * the {\bf argc} and {\bf argv} that were passed to {\bf main}.  On some
 * ports command line arguments can be used to control the underlying
 * message passing system.  For example, on {\em Chameleon} the
 * {\bf -trace} flag can be use to create a log of communication events.
 *
 * {\large Example:}
 * \begin{verbatim}
 * int main( int argc, char *argv)
 * {
 * amps_Init(argc, argv);
 *
 * amps_Printf("Hello World");
 *
 * amps_Finalize();
 * }
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Initialize AMPS
 * @param argc Command line argument count [IN/OUT]
 * @param argv Command line argument array [IN/OUT]
 * @return
 */
int amps_Init(int *argc, char **argv[])
{
#ifdef AMPS_MPI_SETHOME
  char *temp_path;
  int length;
#endif

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int namelen;

/*   MPI_Init(argc, argv);*/
  CALL_oas_pfl_init(&dummy1_oas3);
  amps_mpi_initialized = TRUE;
//CPS get the local commnunicator handle
//   oas3Comm = MPI_Comm_f2c(__oas_pfl_vardef_MOD_localcomm);//for GNU compioers
  oas3Comm = MPI_Comm_f2c(oas_pfl_vardef_mp_localcomm_);   //for Intel Compilers

  MPI_Comm_size(oas3Comm, &amps_size);
  MPI_Comm_rank(oas3Comm, &amps_rank);

#ifdef AMPS_STDOUT_NOBUFF
  setbuf(stdout, NULL);
#endif

#ifdef CASC_HAVE_GETTIMEOFDAY
  amps_clock_init();
#endif

#ifdef AMPS_MPI_SETHOME
  if (!amps_rank)
  {
    if ((temp_path = getenv("AMPS_EXE_DIR")) == NULL)
    {
      printf("AMPS Error: can't get AMPS_EXE_DIR envirnment variabl\n");
      exit(1);
    }

    length = strlen(temp_path) + 1;
  }

  MPI_Bcast(&length, 1, MPI_INT, 0, oas3Comm);

  if (amps_rank)
  {
    temp_path = malloc(length);
  }

  MPI_Bcast(temp_path, length, MPI_CHAR, 0, oas3Comm);

  if (chdir(temp_path))
    printf("AMPS Error: can't set working directory to %s", temp_path);

  if (amps_rank)
  {
    free(temp_path);
  }
#endif

#ifdef AMPS_MALLOC_DEBUG
  dmalloc_logpath = amps_malloclog;
  sprintf(dmalloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif

#ifdef TIMING
#ifndef CRAY_TIME
  AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
#endif
#endif

#ifdef AMPS_PRINT_HOSTNAME
  MPI_Get_processor_name(processor_name, &namelen);

  printf("Process %d on %s\n", amps_rank, processor_name);
#endif

  return 0;
}


/*===========================================================================*/
/**
 *
 * Initialization when ParFlow is being invoked by another application.
 * This must be done before any other {\em AMPS} calls.
 *
 * {\large Example:}
 * \begin{verbatim}
 * int main( int argc, char *argv)
 * {
 * amps_EmbeddedInit();
 *
 * amps_Printf("Hello World");
 *
 * amps_Finalize();
 * }
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Initialize AMPS
 * @return
 */
int amps_EmbeddedInit(void)
{
  MPI_Comm_size(oas3Comm, &amps_size);
  MPI_Comm_rank(oas3Comm, &amps_rank);

#ifdef AMPS_STDOUT_NOBUFF
  setbuf(stdout, NULL);
#endif

#ifdef CASC_HAVE_GETTIMEOFDAY
  amps_clock_init();
#endif

#ifdef AMPS_MALLOC_DEBUG
  dmalloc_logpath = amps_malloclog;
  sprintf(dmalloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif

#ifdef TIMING
#ifndef CRAY_TIME
  AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
#endif
#endif

  return 0;
}


