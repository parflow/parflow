/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

/**

A shared file is closed by invoking the \Ref{amps_SFclose} function.
This routine must be called by all members of the communicator that
opened the file.

{\large Example:}
\begin{verbatim}
amps_File file;
amps_Invoice invoice;

file = amps_SFopen(filename, "r");

amps_SFBCast(amps_CommWorld, file, invoice);

amps_SFclose(file);
\end{verbatim}

{\large Notes:}

@memo Close a shared file
@param file Shared file handle
@return Error code */
int amps_SFclose(amps_File file)
{
   if(amps_Rank(amps_CommWorld))
      return 0;
   else
      return fclose(file);
}
