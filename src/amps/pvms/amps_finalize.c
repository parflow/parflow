/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int amps_Finalize()
{

   if(amps_size > 1)
   {
      pvm_barrier(amps_CommWorld, amps_size);

      pvm_lvgroup(amps_CommWorld);
      free(amps_tids);
   }

   amps_CommWorld = NULL;


   pvm_exit();

#ifdef AMPS_MALLOC_DEBUG
  /* check out the heap and shut everything down if we are in debug mode */
  malloc_verify(0);
  malloc_shutdown();
#endif
   return 0;
}
