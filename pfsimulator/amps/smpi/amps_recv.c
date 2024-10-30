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

char *amps_recvb(src, size)
int src;
int *size;
{
  char *buf;

  MPI_Status status;

  MPI_Probe(src, 0, amps_CommWorld, &status);

  MPI_Get_count(&status, MPI_BYTE, size);

  buf = malloc(*size);

  MPI_Recv(buf, *size, MPI_BYTE, src, 0, amps_CommWorld, &status);

  return buf;
}

/*===========================================================================*/
/**
 *
 * \Ref{amps_Recv} is a blocking receive operation.  It receives a message
 * from the node with {\bf rank} within the {\bf comm} context.  This
 * operation will not return until the receive operation has been
 * completed.  The received data is unpacked into the the data locations
 * specified in the {\bf invoice}.  After the return it is legal to access
 * overlayed variables (which must be freed with \Ref{amps_Clear}).
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
 * @memo Blocking receive
 * @param comm Communication context [IN]
 * @param source Node rank to receive from [IN]
 * @param invoice Data to receive [IN/OUT]
 * @return Error code
 */

int amps_Recv(amps_Comm comm, int source, amps_Invoice invoice)
{
  char *buffer;
  int size;
  MPI_Status status;

  AMPS_CLEAR_INVOICE(invoice);

  MPI_Probe(source, 0, amps_CommWorld, &status);

  MPI_Get_count(&status, MPI_BYTE, &size);

  buffer = malloc(size);

/*
 * amps_Printf("Recv buffer: %x from node %d\n", buffer, source);
 */

  MPI_Recv(buffer, size, MPI_BYTE, source, 0, amps_CommWorld, &status);

  amps_unpack(comm, invoice, buffer);

  AMPS_PACK_FREE_LETTER(comm, invoice, buffer);

  return 0;
}
