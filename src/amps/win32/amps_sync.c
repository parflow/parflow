/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <amps.h>

int amps_Sync(amps_Comm comm)
{
  int i;
  
  if(amps_rank)
    {
      SetEvent(amps_sync_ready[amps_rank]);
      WaitForSingleObject(amps_sync_done[amps_rank], INFINITE);
    }
  else
    {
      for(i=1; i < amps_size; i++)
	WaitForSingleObject(amps_sync_ready[i], INFINITE);
      for(i=1; i < amps_size; i++)
	SetEvent(amps_sync_done[i]);
    }
  
  return 0;
}
