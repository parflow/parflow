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

The \Ref{amps_Fopen} command is used to open a distributed file.  In
the distributed file model there is a disk (possibly virtual)
associated with each node.  This file is written to and read from
using the \Ref{amps_Fprintf} and \Ref{amps_Fscanf} commands just as
you would for a normal UNIX file.  You must ensure that the
distributed file exists on all nodes before attempting to open it for
reading.  The arguments to \Ref{amps_File} are the same as for {\bf
fopen}: {\bf filename} is the name of the file to and {\bf type} is
the mode to open.

\Ref{amps_Fopen} returns NULL if the the file open fails.

{\large Example:}
\begin{verbatim}
amps_File file;

char *filename;
double d;

file = amps_Fopen(filename,"w");

amps_Fprintf(file, "%lf", d);

amps_Fclose(file);
\end{verbatim}

{\large Notes:}

There should be some commands to take a normal file and distribute it
to the nodes.  This presents a problem since {\em AMPS} does not
know how you want to split up the file to the nodes.  This is being 
worked on; basically the {\em AMPS} I/O calls are a hack until some standard
can be arrived at.  The functionality is sufficient for {\em ParFlow} at
the current time.

@memo Open a distributed file
@param filename Filename of the file to operate on [IN]
@param type Mode options [IN]
@return File handle*/
amps_File amps_Fopen(char *filename, char *type)
{
   char temp[255];

   sprintf(temp, "%s.%05d", filename, amps_Rank(amps_CommWorld));

   return fopen(temp, type);
}
