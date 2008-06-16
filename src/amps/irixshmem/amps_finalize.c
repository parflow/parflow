/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <malloc.h>
#include <sys/param.h>

#include "amps.h"

extern char amps_arena_name[];

int amps_memcheck()
{
  struct mallinfo usmalloc_info;

  usmalloc_info = usmallinfo(amps_arena);

  if(!amps_rank)
    {
      amps_Printf("============= Shared Memory ===========================\n");
      amps_Printf("arena    total space in arena                %d\n", usmalloc_info.arena);
      amps_Printf("ordblks  number of ordinary blocks           %d\n", usmalloc_info.ordblks);
      amps_Printf("smblkds  number of small blocks              %d\n", usmalloc_info.smblks);
      amps_Printf("hblks    number of holding blocks            %d\n", usmalloc_info.hblks);
      amps_Printf("hblkhd   space in holding block headers      %d\n", usmalloc_info.hblkhd);
      amps_Printf("usmblks  space in small blocks in use        %d\n", usmalloc_info.usmblks);
      amps_Printf("fsmblks  space in free small blocks          %d\n", usmalloc_info.fsmblks);
      amps_Printf("uordblks space in ordinary blocks in use     %d\n", usmalloc_info.uordblks);
      amps_Printf("fordblks space in free ordinary blocks       %d\n", usmalloc_info.fordblks);
      amps_Printf("keepcost space penalty if keep option        %d\n", usmalloc_info.keepcost);
    }

}

int amps_Finalize()
{

    int i;


    amps_Sync(amps_CommWorld);

    amps_FreeInvoice(amps_SyncInvoice);

    amps_FreeBufferFreeList();

    if(!amps_rank)
    {
       for(i = 0; i < amps_size; i++)
       {
	  usfreelock( amps_shmem_info -> locks[i], amps_arena);
	  usfreesema( amps_shmem_info -> sema[i], amps_arena);
	  usfreelock( amps_shmem_info -> bcast_locks[i], amps_arena);
	  usfreesema( amps_shmem_info -> bcast_sema[i], amps_arena);
       }
       
       usfreelock(amps_shmem_info -> bcast_lock, amps_arena);
       free_barrier(amps_shmem_info -> sync);
       usfree(amps_shmem_info, amps_arena);

       usdetach(amps_arena);
    }
    else
       usdetach(amps_arena);

#ifdef AMPS_MALLOC_DEBUG
  /* check out the heap and shut everything down if we are in debug mode */
  malloc_verify(0);
  malloc_shutdown();
#endif

    return 0;
 }
