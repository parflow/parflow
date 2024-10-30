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

int amps_xsend(
               amps_Comm    comm,
               int          dest,
               amps_Invoice invoice,
               char *       buffer)
{
  amps_create_mpi_cont_send_type(comm, invoice);

  MPI_Type_commit(&invoice->mpi_type);

  MPI_Send(buffer, 1, invoice->mpi_type, dest, 0, amps_CommWorld);

  MPI_Type_free(&invoice->mpi_type);

  return 0;
}

/*===========================================================================*/
/**
 *
 * Sends a message to the node with {\bf rank} in the {\bf comm} context.
 * The contents of the message are described by the {\bf invoice} as
 * indicated in \Ref{amps_NewInvoice}.  After a return
 * from send it is invalid to access any overlaid variables.  This
 * version of send is possibly blocking.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int me, i;
 * double d;
 *
 * me = amps_Rank(amps_CommWorld);
 *
 * invoice = amps_NewInvoice("%i%d", &i, &d);
 *
 * amps_Send(amps_CommWorld, me+1, invoice);
 *
 * amps_Recv(amps_CommWorld, me-1, invoice);
 *
 * amps_FreeInvoice(invoice);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * Since {\em AMPS} is layered on other message passing systems there is
 * some loosely defined behavior.  \Ref{amps_Send} is to be implemented
 * in a way that starts and completes a send operation on the source
 * side.  Since some systems have synchronization or do queuing on the
 * sending side this function can block until a receive is posted on the
 * node specified by {\bf rank}.
 *
 * @memo Blocking send
 * @param comm Communication context [IN]
 * @param dest Rank of destination node [IN]
 * @param invoice Data to communicate [IN]
 * @return Error code
 */
int amps_Send(amps_Comm comm, int dest, amps_Invoice invoice)
{
  amps_create_mpi_type(comm, invoice);

  MPI_Type_commit(&invoice->mpi_type);

  MPI_Send(MPI_BOTTOM, 1, invoice->mpi_type, dest, 0, amps_CommWorld);

  MPI_Type_free(&invoice->mpi_type);

  return 0;
}

