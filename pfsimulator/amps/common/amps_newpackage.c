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

#ifndef AMPS_NEWPACKAGE_SPECIALIZED

amps_Package amps_NewPackage(amps_Comm     comm,
                             int           num_send,
                             int *         dest,
                             amps_Invoice *send_invoices,
                             int           num_recv,
                             int *         src,
                             amps_Invoice *recv_invoices)
{
  amps_Package package;


  package = (amps_Package)malloc(sizeof(amps_PackageStruct));

  package->num_send = num_send;
  package->dest = dest;
  package->send_invoices = send_invoices;

  package->num_recv = num_recv;
  package->src = src;
  package->recv_invoices = recv_invoices;

  if (num_recv)
    package->recv_handles = (struct amps_HandleObject **)malloc(sizeof(amps_Handle) * num_recv);

  return package;
}

void amps_FreePackage(amps_Package package)
{
  if (package->num_recv)
    free(package->recv_handles);
  free(package);
}

#endif
