/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <malloc.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/param.h>
#include <sys/unistd.h>
#include <sys/times.h>

#include "amps.h"

int amps_tid;
int amps_rank;
int amps_size;

void *redWrk;
void *redWrk_buf;
#pragma _CRI cache_align barSync, bcaSync, redSync
long redSync[_SHMEM_REDUCE_SYNC_SIZE];
long barSync[_SHMEM_BARRIER_SYNC_SIZE];
long bcaSync[_SHMEM_BCAST_SYNC_SIZE];

int amps_shared_mem_size;

char *amps_shared_mem;

char *amps_shared_mem_ptr;

void *amps_talloc(int size)
{

  if(      (amps_shared_mem_ptr += size) > 
     (amps_shared_mem + amps_shared_mem_size) )
    {
      amps_Printf("AMPS Error: out of shared memory space\n");
      exit(1);
    }
  
  return amps_shared_mem_ptr - size;
}

void *amps_calloc(int count, int size)
{

  memset(amps_shared_mem_ptr, 0, size*count);
  if( (amps_shared_mem_ptr += size*count) > 
     (amps_shared_mem + amps_shared_mem_size) )
    {
      amps_Printf("AMPS Error: out of shared memory space\n");
      exit(1);
    }

  return amps_shared_mem_ptr - (size*count);
}
 
void shmem_init()
{
    int i, n;

    char *amps_size;

    if( (amps_size = getenv("AMPS_SHMEM_SIZE")) == NULL)
      {
	amps_Printf("AMPS Error: Can't find envirnment variable AMPS_SHMEM_SIZE\n");
	exit(1);
      }
    

    amps_shared_mem_size = atoi(amps_size);
    

    if ( (amps_shared_mem = shmalloc(amps_shared_mem_size))== NULL)
      {
	amps_Printf("AMPS Error: Failed to allocate shared memory buffer\n");
	exit(1);
      }

    amps_shared_mem_ptr = (char *)amps_shared_mem;
 
    /* Initialize reduction synchronization and work data arrays */
    
    for (i=0; i<_SHMEM_REDUCE_SYNC_SIZE; i++) {
        redSync[i] = _SHMEM_SYNC_VALUE;
    }
 
    n = 8 * 
      (((amps_ReduceBufSize/2)+1 > _SHMEM_REDUCE_MIN_WRKDATA_SIZE) ?
       (amps_ReduceBufSize/2)+1 : _SHMEM_REDUCE_MIN_WRKDATA_SIZE);
	
 
    redWrk = (void *)shmalloc(n);
    
    redWrk_buf = (void *)shmalloc(8*amps_ReduceBufSize);
 
    /* Initialize barrier synchronization and work data arrays */
 
    for (i=0; i<_SHMEM_BARRIER_SYNC_SIZE; i++) {
        barSync[i] = _SHMEM_SYNC_VALUE;
    }
 
    /* Initialize broadcast synchronization and work data arrays */
 
    for (i=0; i<_SHMEM_BCAST_SYNC_SIZE; i++) {
        bcaSync[i] = _SHMEM_SYNC_VALUE;
    }
 
    barrier();
}


int amps_Init(argc, argv)
int *argc;
char **argv[];
{

#if 0
    pvm_setopt(PvmAutoErr, 2);
#endif

    amps_rank = pvm_get_PE(pvm_mytid());

    pvm_barrier(NULL,-1);

    amps_size = pvm_gsize(amps_CommWorld);

    shmem_init();

    return 0;
}  


