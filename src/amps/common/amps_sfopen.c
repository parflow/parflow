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

There are three types files in {\em AMPS}, shared, fixed and
distributed.  The \Ref{amps_SFOpen} command opens a shared file.
Shared files are opened and operated on by an entire context of nodes
specified by the {\bf comm} communicator.  Shared files provide a
simple way to distribute information from an input file to a group of
nodes.  This routine must be called by all members of the communicator
and all node members must call the same shared I/O routines in the
same order on the opened file.  The returned \Ref{amps_File} must be
closed by \Ref{amps_SFclose}.

A {\bf NULL} return value indicates the open failed.

{\large Example:}
\begin{verbatim}
amps_File file;
amps_Invoice invoice;

file = amps_SFopen(filename, "r");

amps_SFBCast(amps_CommWorld, file, invoice);

amps_SFclose(file);
\end{verbatim}

{\large Notes:}

@memo Open a shared file
@param filename Name of file to open [IN]
@param type Mode to open file [IN]
@return shared fille handle
*/
amps_File amps_SFopen(char *filename,char *type)
{

   if(amps_Rank(amps_CommWorld))
      return (amps_File)1;
   else
      return fopen(filename, type);
}
