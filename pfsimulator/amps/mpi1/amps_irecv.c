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

/*===========================================================================*/
/**
 * \Ref{amps_IRecv} is a non-blocking receive operation which returns a
 * \Ref{amps_Handle}.  It receives a message from the node with {\bf rank}
 * within the {\bf comm} context.  This operation will attempt to post a
 * receive and then return even if the receive operation is not completed.
 * The {\bf handle} is passed to \Ref{amps_Test} to test for completion
 * and to \Ref{amps_Wait} to block waiting for completion to occur.
 * After a test succeeds or the wait returns it is legal to access
 * variables within the {\bf invoice}.  Any attempt to access variables
 * used in the {\bf invoice} between the posting of the \Ref{amps_IRecv}
 * command and the successful \Ref{amps_Test} or \Ref{amps_Wait} is not
 * permitted (since these variables might be changing).  Every \Ref{amps_IRecv}
 * must have an associated \Ref{amps_Wait} to finalize the receive
 * (even if there has been a positive \Ref{amps_Test} done).
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * amps_Handle  handle;
 * int me, i;
 * double d;
 *
 * me = amps_Rank(amps_CommWorld);
 *
 * invoice = amps_NewInvoice("%i%d", &i, &d);
 *
 * handle = amps_IRecv(amps_CommWorld, me+1, invoice);
 *
 * while(amps_Test(handle))
 * {
 *      //
 * }
 * amps_Wait(handle);
 *
 * amps_FreeInvoice(invoice);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * If a non-blocking receive is not available then \Ref{amps_IRecv} does
 * nothing and the receive operation is done in the \Ref{amps_Wait}
 * routine.
 *
 * @memo Non-blocking receive
 * @param comm Communication context [IN]
 * @param source Rank of the sending node [IN]
 * @param invoice Data to receive [IN/OUT]
 * @return Handle for the receive
 */
amps_Handle amps_IRecv(amps_Comm comm, int source, amps_Invoice invoice)
{
  return amps_NewHandle(comm, source, invoice, NULL);
}
