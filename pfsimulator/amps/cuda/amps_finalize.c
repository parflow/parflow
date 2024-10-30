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

extern int amps_mpi_initialized;

/*===========================================================================*/
/**
 *
 * Every {\em AMPS} program must call this function to exit from the
 * message passing environment.  This must be the last call to an
 * {\em AMPS} routine.  \Ref{amps_Finalize} might synchronize the
 * of the node programs;  this might be necessary to correctly free up
 * memory resources and communication structures.
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
 *
 * There is currently no way to forcibly kill another node.  Exiting do
 * to an error condition is problematic.
 *
 * @memo Exit AMPS environment
 * @return Error code
 */

int amps_Finalize()
{
  if (amps_mpi_initialized)
  {
    MPI_Comm_free(&amps_CommNode);
    MPI_Comm_free(&amps_CommWrite);
    MPI_Comm_free(&amps_CommWorld);

    MPI_Finalize();
  }

  if (amps_device_globals.combuf_recv_size != 0)
    CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_recv));
  if (amps_device_globals.combuf_send_size != 0)
    CUDA_ERRCHK(cudaFree(amps_device_globals.combuf_send));

  for (int i = 0; i < amps_device_globals.streams_created; i++)
  {
    CUDA_ERRCHK(cudaStreamDestroy(amps_device_globals.stream[i]));
  }

  return 0;
}
