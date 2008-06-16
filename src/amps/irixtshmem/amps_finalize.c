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


#ifdef AMPS_MALLOC_DEBUG
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

  usmalloc_info = mallinfo();

  if(!amps_rank)
    {
      amps_Printf("============= Heap   Memory ===========================\n");
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

#endif

int amps_Finalize()
{
    amps_Sync(amps_CommWorld);

    amps_FreeInvoice(amps_SyncInvoice);

    amps_FreeBufferFreeList();

    return 0;
 }
