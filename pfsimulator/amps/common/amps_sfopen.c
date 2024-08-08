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
 * There are three types files in {\em AMPS}, shared, fixed and
 * distributed.  The \Ref{amps_SFOpen} command opens a shared file.
 * Shared files are opened and operated on by an entire context of nodes
 * specified by the {\bf comm} communicator.  Shared files provide a
 * simple way to distribute information from an input file to a group of
 * nodes.  This routine must be called by all members of the communicator
 * and all node members must call the same shared I/O routines in the
 * same order on the opened file.  The returned \Ref{amps_File} must be
 * closed by \Ref{amps_SFclose}.
 *
 * A {\bf NULL} return value indicates the open failed.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_File file;
 * amps_Invoice invoice;
 *
 * file = amps_SFopen(filename, "r");
 *
 * amps_SFBCast(amps_CommWorld, file, invoice);
 *
 * amps_SFclose(file);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Open a shared file
 * @param filename Name of file to open [IN]
 * @param type Mode to open file [IN]
 * @return shared fille handle
 */
#ifndef AMPS_SFOPEN_SPECIALIZED
amps_File amps_SFopen(const char *filename, const char *type)
{
  if (amps_Rank(amps_CommWorld))
    return (amps_File)1;
  else
    return fopen(filename, type);
}
#endif
