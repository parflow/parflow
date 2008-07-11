/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

#ifndef AMPS_NEWHANDLE_SPECIALIZED
amps_Handle amps_NewHandle(comm, id, invoice, package)
amps_Comm comm;
int id;
amps_Invoice invoice;
amps_Package package;
{
   amps_Handle handle;

   handle = (amps_Handle)malloc(sizeof(amps_HandleObject));

   handle -> comm     = comm;
   handle -> id       = id;
   handle -> invoice  = invoice;
   handle -> package  = package;
   
   handle -> type = (invoice != NULL) ? 1 : 0;

   return handle;
}
#endif
