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
 * \Ref{amps_Wait} is used to block until the communication initiated by
 * them has completed.  {\bf handle} is the communications handle that
 * was returned by the \Ref{amps_ISend}, \Ref{amps_IRecv}, or
 * \Ref{amps_IExchangePackage} commands.  You must always do an
 * \Ref{amps_Wait} on to finalize an initiated non-blocking
 * communication.
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
 * handle = amps_ISend(amps_CommWorld, me+1, invoice);
 *
 * // do some work
 *
 * amps_Wait(handle);
 *
 * handle = amps_IRecv(amps_CommWorld, me+1, invoice);
 *
 * while(amps_Test(handle))
 * {
 *      // do more work
 * }
 * amps_Wait(handle);
 *
 * amps_FreeInvoice(invoice);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * The requirement for to always finish an initiated communication with
 * \Ref{amps_Wait} is under consideration.
 *
 * @memo Wait for initialized non-blocking communication to finish
 * @param handle communication handle
 * @return Error code
 */
#ifndef AMPS_WAIT_SPECIALIZED
int amps_Wait(amps_Handle handle)
{
  if (handle)
  {
    if (handle->type)
      amps_Recv(handle->comm, handle->id, handle->invoice);
    else
      _amps_wait_exchange(handle);

    amps_FreeHandle(handle);
    handle = NULL;
  }

  return 0;
}
#endif
