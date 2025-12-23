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

amps_Package amps_NewPackage(amps_Comm     comm,
                             int           num_send,
                             int *         dest,
                             amps_Invoice *send_invoices,
                             int           num_recv,
                             int *         src,
                             amps_Invoice *recv_invoices)
{
  amps_Package package;

  amps_PackageItem *items;
  amps_SrcInfo *src_info;
  amps_InvoiceEntry *ptr;
  int num_package_items;
  int i, j;
  int dim;
  int item;

  package = (amps_Package)malloc(sizeof(amps_PackageStruct));

  package->num_send = num_send;
  package->dest = dest;
  package->send_invoices = send_invoices;

  package->num_recv = num_recv;
  package->src = src;
  package->recv_invoices = recv_invoices;

  /* For each of the destinations pack up and send the package information
   * (ie locations, lengths and strides ) */

  package->snd_info = NULL;
  if (num_send)
  {
    package->snd_info = malloc(sizeof(amps_SrcInfo * *) * num_send);
    for (i = 0; i < num_send; i++)
    {
      /* allocate an array to send */
      num_package_items = amps_num_package_items(send_invoices[i]);

      item = 0;
      items = amps_TAlloc(amps_PackageItem, num_package_items);

      /* Pack up invoice */
      ptr = send_invoices[i]->list;

      while (ptr != NULL)
      {
        if (ptr->type > AMPS_INVOICE_LAST_CTYPE)
        {
          items[item].type = ptr->type;

          items[item].data = (ptr->data_type == AMPS_INVOICE_POINTER) ?
                             *((char**)(ptr->data)) : ptr->data;

          /* Store the dim of the vector */
          dim = items[item].dim =
            (ptr->dim_type == AMPS_INVOICE_POINTER) ?
            *(ptr->ptr_dim) : ptr->dim;

          /* Pack the vector len and strides into following package
           * items */
          for (j = 0; j < dim; j++)
          {
            items[item].len = ptr->ptr_len[j];
            items[item].stride = ptr->ptr_stride[j];

            item++;
          }
        }
        else
        {
          items[item].type = ptr->type;

          items[item].data = (ptr->data_type == AMPS_INVOICE_POINTER) ?
                             *((char**)(ptr->data)) : ptr->data;

          items[item].len = (ptr->len_type == AMPS_INVOICE_POINTER) ?
                            *(ptr->ptr_len) : ptr->len;

          items[item].stride =
            (ptr->stride_type == AMPS_INVOICE_POINTER) ?
            *(ptr->ptr_stride) :  ptr->stride;

          item++;
        }

        ptr = ptr->next;
      }

      src_info = (amps_SrcInfo*)amps_new(amps_CommWorld,
                                         sizeof(amps_SrcInfo));

      if ((src_info->send_sema = CreateSemaphore(0, 0, AMPS_MAX_MESGS, 0)) == NULL)
        printf("error allocating sema send\n");

      if ((src_info->recv_sema = CreateSemaphore(0, 0, AMPS_MAX_MESGS, 0)) == NULL)
        printf("error allocating sema send\n");

      src_info->items = items;

      package->snd_info[i] = src_info;

      /* send the package items to the dest */
      amps_xsend((char*)src_info, dest[i]);
    }
  }

  if (num_recv)
  {
    package->rcv_info = malloc(sizeof(amps_SrcInfo * *) * num_recv);

    /* For each of the src recv the package information */
    for (i = 0; i < num_recv; i++)
    {
      package->rcv_info[i] = (amps_SrcInfo*)amps_recvb(src[i]);
    }
  }

  return package;
}


void amps_FreePackage(amps_Package package)
{
  int i;


  if (package)
  {
    if (i = package->num_recv)
    {
      while (i--)
      {
        CloseHandle(package->rcv_info[i]->send_sema);
        CloseHandle(package->rcv_info[i]->recv_sema);
        amps_TFree(package->rcv_info[i]->items);
        amps_free(amps_CommWorld, package->rcv_info[i]);
      }
      free(package->rcv_info);
    }
    if (package->snd_info)
      free(package->snd_info);

    free(package);
  }
}
