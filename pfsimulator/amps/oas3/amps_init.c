/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
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

#include "amps.h"

#include <stdio.h>
#include <sys/param.h>
#include <sys/times.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>

/* Global flag indicating if AMPS has been initialized */
int amps_mpi_initialized = FALSE;

#ifdef AMPS_MALLOC_DEBUG
char amps_malloclog[MAXPATHLEN];
#endif

int amps_size;
int amps_rank;
int amps_node_rank;
int amps_node_size;
int amps_write_rank;
int amps_write_size;
MPI_Comm amps_CommNode = MPI_COMM_NULL;
MPI_Comm amps_CommWrite = MPI_COMM_NULL;

MPI_Comm oas3Comm = MPI_COMM_NULL;
int dummy1_oas3 = 0;

#ifdef AMPS_F2CLIB_FIX
int MAIN__()
{
}
#endif

/*Adler32 function to calculate hash based on node name and length */
const int MOD_ADLER = 65521;

uint32_t Adler32(unsigned char *data, size_t len) /* where data is the location of the data in physical memory and
                                                   * len is the length of the data in bytes */
{
  uint32_t a = 1, b = 0;
  size_t index;

/* Process each byte of the data in order */
  for (index = 0; index < len; ++index)
  {
    a = (a + data[index]) % MOD_ADLER;
    b = (b + a) % MOD_ADLER;
  }

  return (b << 16) | a;
}

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

  CALL_oas_pfl_init(&dummy1_oas3);
  amps_mpi_initialized = TRUE;
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  oas3Comm = MPI_Comm_f2c(oas_pfl_vardef_mp_localcomm_);  //for Intel compilers
#else
  oas3Comm = MPI_Comm_f2c(__oas_pfl_vardef_MOD_localcomm);  //for GNU compilers
#endif

  MPI_Comm_size(oas3Comm, &amps_size);
  MPI_Comm_rank(oas3Comm, &amps_rank);

  /* Create communicator with one rank per compute node */
#if MPI_VERSION >= 3
  MPI_Comm_split_type(oas3Comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &amps_CommNode);
#else
  /* Split the node level communicator based on Adler32 hash keys of processor name */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  MPI_Get_processor_name(processor_name, &namelen);
  uint32_t checkSum = Adler32((unsigned char*)processor_name, namelen);
  /* Comm split only accepts non-negative numbers */
  /* Not super great for hashing purposes but hoping MPI-3 code will be used on most cases */
  checkSum &= INT_MAX;
  MPI_Comm_split(oas3Comm, checkSum, amps_rank, &amps_CommNode);
#endif

  MPI_Comm_rank(amps_CommNode, &amps_node_rank);
  MPI_Comm_size(amps_CommNode, &amps_node_size);
  int color;
  if (amps_node_rank == 0)
  {
    color = 0;
  }
  else
  {
    color = 1;
  }
  MPI_Comm_split(oas3Comm, color, amps_rank, &amps_CommWrite);
  if (amps_node_rank == 0)
  {
    MPI_Comm_size(amps_CommWrite, &amps_write_size);
  }


#ifdef AMPS_STDOUT_NOBUFF
  setbuf(stdout, NULL);
#endif

  amps_clock_init();

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

#ifdef AMPS_PRINT_HOSTNAME
  MPI_Get_processor_name(processor_name, &namelen);

  printf("Process %d on %s\n", amps_rank, processor_name);
#endif

  return 0;
}

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

  amps_clock_init();

#ifdef AMPS_MALLOC_DEBUG
  dmalloc_logpath = amps_malloclog;
  sprintf(dmalloc_logpath, "malloc.log.%04d", amps_Rank(amps_CommWorld));
#endif

  return 0;
}

/**
 * Compare oas3Comm to communicator for embedded parflow application.
 *
 * ParFlow converts communicator context from f2c and compares to oas3Comm.
 *
 * {\large Example:}
 * \begin{verbatim}
 * int main( int argc, char *argv)
 * {
 * <MPI Initialized>
 * amps_EmbeddedInitFComm(MPI_COMM_PARFLOW);
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
 * @param comm MPI Fortran communicator context to use for ParFlow
 * @return
 */
int amps_EmbeddedInitFComm(MPI_Fint *f_handle)
{
  MPI_Comm comm;

  comm = MPI_Comm_f2c(*f_handle);
  return amps_EmbeddedInitComm(comm);
}

/**
 * Compare oas3Comm to communicator for embedded parflow application.
 *
 * ParFlow will continue to use oas3Comm but checks for congruency.
 *
 * {\large Example:}
 * \begin{verbatim}
 * int main( int argc, char *argv)
 * {
 * <MPI Initialized>
 * amps_EmbeddedInit(MPI_COMM_WORLD);
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
 * @param comm MPI communicator context to use for ParFlow
 * @return
 */
int amps_EmbeddedInitComm(MPI_Comm comm)
{
  int result;

  MPI_Comm_compare(oas3Comm, comm, &result);
  if (result != MPI_CONGRUENT)
  {
    printf("AMPS Error: oas3Comm is not congruent with embedded usage\n");
    exit(1);
  }

  return amps_EmbeddedInit();
}
