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

#undef main

int amps_size;

_declspec(thread) int amps_rank;

HANDLE *amps_sync_ready;
HANDLE *amps_sync_done;

HANDLE *locks;

HANDLE *sema;

HANDLE bcast_lock;
HANDLE *bcast_locks;

HANDLE *bcast_sema;


AMPS_ShMemBuffer **buffers;
AMPS_ShMemBuffer **buffers_end;

AMPS_ShMemBuffer **buffers_local;
AMPS_ShMemBuffer **buffers_local_end;

AMPS_ShMemBuffer **bcast_buffers;
AMPS_ShMemBuffer **bcast_buffers_end;

AMPS_ShMemBuffer **bcast_buffers_local;
AMPS_ShMemBuffer **bcast_buffers_local_end;


amps_Comm amps_CommWorld = 21;

int amps_Init(argc, argv)
int *argc;
char **argv[];
{
  return 0;
}

unsigned amps_main(void *arg)
{
  unsigned result = 0;

  char **argv = arg;
  int argc;

  amps_rank = (int)argv[0];

  argc = (int)argv[1];

  /* Invoke users main here */
  AMPS_USERS_MAIN(argc, &argv[2]);

  /* this is alloced in main thread but freed here */
  free(arg);

  return result;
}

int main(argc, argv)
int argc;
char *argv[];
{
  int i, j;

  char **pass_argv;

  HANDLE *thread_handle;
  unsigned ThreadId;

  if (argc < 2)
  {
    printf("Error: specify the number of nodes\n");
    exit(1);
  }

  amps_size = atoi(argv[1]);

  thread_handle = malloc(sizeof(HANDLE) * amps_size);

  amps_sync_ready = malloc(sizeof(HANDLE) * amps_size);
  amps_sync_done = malloc(sizeof(HANDLE) * amps_size);

  locks = malloc(sizeof(HANDLE) * amps_size);

  sema = malloc(sizeof(HANDLE) * amps_size);

  bcast_locks = malloc(sizeof(HANDLE) * amps_size);
  bcast_sema = malloc(sizeof(HANDLE) * amps_size);

  buffers = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  buffers_end = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  buffers_local = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  buffers_local_end = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));

  bcast_buffers = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  bcast_buffers_end = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  bcast_buffers_local = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));
  bcast_buffers_local_end = calloc(amps_size, sizeof(AMPS_ShMemBuffer *));

  for (i = 0; i < amps_size; i++)
  {
    locks[i] = CreateMutex(NULL, FALSE, NULL);
    sema[i] = CreateSemaphore(0, 0, AMPS_MAX_MESGS, 0);

    bcast_locks[i] = CreateMutex(NULL, FALSE, NULL);
    bcast_sema[i] = CreateSemaphore(0, 0, AMPS_MAX_MESGS, 0);
  }

  for (i = 1; i < amps_size; i++)
  {
    amps_sync_ready[i] = CreateEvent(0,
                                     FALSE,
                                     FALSE,
                                     0);

    amps_sync_done[i] = CreateEvent(0,
                                    FALSE,
                                    FALSE,
                                    0);
  }

  for (i = 0; i < amps_size; i++)
  {
    pass_argv = (char**)malloc((argc + 2) * sizeof(char *));

    pass_argv[0] = (char*)i;
    pass_argv[1] = (char*)argc - 1;
    pass_argv[2] = argv[0];

    for (j = 2; j < argc; j++)
      pass_argv[j + 1] = argv[j];

    thread_handle[i] = (HANDLE)_beginthreadex(NULL, 0,
                                              (LPTHREAD_START_ROUTINE)amps_main,
                                              (void*)pass_argv,
                                              0,
                                              &ThreadId);
  }

  /* Wait for all threads to finish up */

  WaitForMultipleObjects(amps_size, thread_handle, TRUE, INFINITE);

  for (i = 1; i < amps_size; i++)
  {
    CloseHandle(thread_handle[i]);
    CloseHandle(amps_sync_ready[i]);
    CloseHandle(amps_sync_done[i]);
  }

  free(thread_handle);


  free(amps_sync_ready);
  free(amps_sync_done);

  free(locks);
  free(sema);
  free(bcast_locks);
  free(bcast_sema);

  free(buffers);
  free(buffers_end);
  free(buffers_local);
  free(buffers_local_end);

  free(bcast_buffers);
  free(bcast_buffers_end);
  free(bcast_buffers_local);
  free(bcast_buffers_local_end);


  return 0;
}


void *_amps_CTAlloc(int count, char *filename, int line)
{
  void *ptr;

  if (count)
    if ((ptr = calloc(count, 1)) == NULL)
    {
      amps_Printf("Error: out of memory in <%s> at line %d\n",
                  filename, line);
      exit(1);
      return NULL;
    }
    else
      return ptr;
  else
    return NULL;
}

void *_amps_TAlloc(int count, char *filename, int line)
{
  void *ptr;

  if (count)
    if ((ptr = malloc(count)) == NULL)
    {
      amps_Printf("Error: out of memory in <%s> at line %d\n",
                  filename, line);
      exit(1);
      return NULL;
    }
    else
      return ptr;
  else
    return NULL;
}


