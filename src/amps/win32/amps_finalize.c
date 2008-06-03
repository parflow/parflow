/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <malloc.h>

#include "amps.h"

int amps_Finalize()
{

  amps_Sync(amps_CommWorld);

  return 0;
}


