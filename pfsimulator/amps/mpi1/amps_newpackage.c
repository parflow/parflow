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

#ifdef AMPS_MPI_NOT_USE_PERSISTENT

amps_Package amps_NewPackage(amps_Comm     comm,
                             int           num_send,
                             int *         dest,
                             amps_Invoice *send_invoices,
                             int           num_recv,
                             int *         src,
                             amps_Invoice *recv_invoices)
{
  amps_Package package;


  package = (amps_Package)calloc(1, sizeof(amps_PackageStruct));

  if (num_recv + num_send)
  {
    package->recv_requests = 
      (MPI_Request*)calloc((num_recv + num_send), sizeof(MPI_Request));

    package->send_requests = package->recv_requests + num_recv;

    package->status = 
      (MPI_Status*)calloc((num_recv + num_send), sizeof(MPI_Status));
  }

  package->num_send = num_send;
  package->dest = dest;
  package->send_invoices = send_invoices;

  package->num_recv = num_recv;
  package->src = src;
  package->recv_invoices = recv_invoices;

  return package;
}

void amps_FreePackage(amps_Package package)
{
  if (package->num_recv + package->num_send)
  {
    free(package->recv_requests);
    free(package->status);
  }

  free(package);
}

#else

amps_Package amps_NewPackage(amps_Comm     comm,
                             int           num_send,
                             int *         dest,
                             amps_Invoice *send_invoices,
                             int           num_recv,
                             int *         src,
                             amps_Invoice *recv_invoices)
{
  amps_Package package;

  (void)comm;

  package = (amps_Package)calloc(1, sizeof(amps_PackageStruct));

  package->num_recv = num_recv;
  package->src = src;
  package->recv_invoices = recv_invoices;

  package->num_send = num_send;
  package->dest = dest;
  package->send_invoices = send_invoices;

  return package;
}

void amps_FreePackage(amps_Package package)
{
  int i;
  MPI_Datatype type;

  if (package)
  {
    if (package->commited)
    {
      for (i = 0; i < package->num_recv; i++)
      {
        type = package->recv_invoices[i]->mpi_type;
        if (type != MPI_DATATYPE_NULL && type != MPI_BYTE)
        {
          MPI_Type_free(&(package->recv_invoices[i]->mpi_type));
        }


        MPI_Request_free(&package->recv_requests[i]);
      }

      for (i = 0; i < package->num_send; i++)
      {
        type = package->send_invoices[i]->mpi_type;
        if (type != MPI_DATATYPE_NULL && type != MPI_BYTE)
        {
          MPI_Type_free(&package->send_invoices[i]->mpi_type);
        }

        MPI_Request_free(&package->send_requests[i]);
      }

      if (package->num_send + package->num_recv)
      {
        free(package->recv_requests);
        free(package->status);
      }

      package->commited = FALSE;
    }

    free(package);
  }
}

#endif


