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
#include <stdarg.h>

#include "amps.h"

/*===========================================================================*/
/**
 *
 * This routine is used to print information to the standard output device.
 * Where the output actually ends up is dependent on the underlying message
 * passing system.  On some systems it will appear on the console where the
 * program was run; on others it will be placed in a file.  The arguments
 * are the same as for the standard C {\bf printf} function.
 *
 * {\large Example:}
 * \begin{verbatim}
 * int main( int argc, char *argv)
 * {
 * amps_Init(argc, argv);
 *
 * amps_Printf("Hello World");
 *
 * amps_Finalize();
 * }
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * Where the output ends up and the order of the output
 * is dependent on the underlying message passing system.
 *
 * @memo Print to stdout
 * @param fmt Output format string [IN]
 * @param ... Output optional variables
 * @return void
 */

void amps_Printf(const char *fmt, ...)
{
  va_list argp;

  fprintf(stdout, "Node %d: ", amps_rank);
  va_start(argp, fmt);

#ifdef SGS_LAM
  /* In order to capture output in tcl can't have output kept
   * open so direct to stderr and send stdout to /dev/null
   * LAM is doing something with stdout */
  vfprintf(stderr, fmt, argp);
#else
  vfprintf(stdout, fmt, argp);
#endif

  va_end(argp);
}

