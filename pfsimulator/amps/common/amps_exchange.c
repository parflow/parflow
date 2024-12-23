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

#ifndef AMPS_EXCHANGE_SPECIALIZED

void _amps_wait_exchange(amps_Handle handle)
{
  int notdone;
  int i;


  notdone = handle->package->recv_remaining;
  while (notdone > 1)
    for (i = 0; i < handle->package->num_recv; i++)
      if (handle->package->recv_handles[i])
        if (amps_Test((amps_Handle)handle->package->recv_handles[i]))
        {
          handle->package->recv_handles[i] = NULL;
          notdone--;
        }

  for (i = 0; i < handle->package->num_recv; i++)
    if (handle->package->recv_handles[i])
    {
      amps_Wait((amps_Handle)handle->package->recv_handles[i]);
      handle->package->recv_handles[i] = NULL;
    }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  package->recv_remaining = 0;

  for (i = 0; i < package->num_recv; i++)
    if ((package->recv_handles[i] =
           (struct amps_HandleObject *)amps_IRecv(amps_CommWorld,
                                                  package->src[i],
                                                  package->recv_invoices[i])))
      package->recv_remaining++;

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    amps_Send(amps_CommWorld,
              package->dest[i],
              package->send_invoices[i]);
  }

  return(amps_NewHandle(NULL, 0, NULL, package));
}

#endif
