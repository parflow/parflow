/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#include "amps.h"

/**
 *
 * The \Ref{amps_Fopen} command is used to open a distributed file.  In
 * the distributed file model there is a disk (possibly virtual)
 * associated with each node.  This file is written to and read from
 * using the \Ref{amps_Fprintf} and \Ref{amps_Fscanf} commands just as
 * you would for a normal UNIX file.  You must ensure that the
 * distributed file exists on all nodes before attempting to open it for
 * reading.  The arguments to \Ref{amps_File} are the same as for {\bf
 * fopen}: {\bf filename} is the name of the file to and {\bf type} is
 * the mode to open.
 *
 * \Ref{amps_Fopen} returns NULL if the the file open fails.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_File file;
 *
 * char *filename;
 * double d;
 *
 * file = amps_Fopen(filename,"w");
 *
 * amps_Fprintf(file, "%lf", d);
 *
 * amps_Fclose(file);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * There should be some commands to take a normal file and distribute it
 * to the nodes.  This presents a problem since {\em AMPS} does not
 * know how you want to split up the file to the nodes.  This is being
 * worked on; basically the {\em AMPS} I/O calls are a hack until some standard
 * can be arrived at.  The functionality is sufficient for {\em ParFlow} at
 * the current time.
 *
 * @memo Open a distributed file
 * @param filename Filename of the file to operate on [IN]
 * @param type Mode options [IN]
 * @return File handle*/

#ifndef AMPS_FOPEN_SPECIALIZED

amps_File amps_Fopen(char *filename, char *type)
{
  char temp[255];

  sprintf(temp, "%s.%05d", filename, amps_Rank(amps_CommWorld));

  return fopen(temp, type);
}

#endif
