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
 * A shared file is closed by invoking the \Ref{amps_SFclose} function.
 * This routine must be called by all members of the communicator that
 * opened the file.
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
 * @memo Close a shared file
 * @param file Shared file handle
 * @return Error code */

#ifndef AMPS_SFCLOSE_SPECIALIZED
int amps_SFclose(amps_File file)
{
  if (amps_Rank(amps_CommWorld))
    return 0;
  else
    return fclose(file);
}
#endif
