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
#include <string.h>
#include <stdarg.h>

#include "amps.h"

int amps_CreateInvoice(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  int size, len, stride;

  size = amps_sizeof_invoice(comm, inv);

  /* if allocated then we deallocate                                       */
  amps_ClearInvoice(inv);

  cur_pos = inv->combuf = amps_new(mlr, size);

  /* set flag indicateing we have allocated the space                      */
  inv->combuf_flags |= AMPS_INVOICE_ALLOCATED;
  inv->combuf_flags |= AMPS_INVOICE_OVERLAYED;

  inv->comm = comm;

  ptr = inv->list;
  while (ptr != NULL)
  {
    if (ptr->len_type == AMPS_INVOICE_POINTER)
      len = *(ptr->ptr_len);
    else
      len = ptr->len;

    if (ptr->stride_type == AMPS_INVOICE_POINTER)
      stride = *(ptr->ptr_stride);
    else
      stride = ptr->stride;

    switch (ptr->type)
    {
      case AMPS_INVOICE_CHAR_CTYPE:
        cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(char) * len * stride);
        cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(short) * len * stride);
        cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(int) * len * stride);
        cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(long) * len * stride);
        cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(float) * len * stride);
        cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, stride);
        if (ptr->data_type == AMPS_INVOICE_POINTER)
          if (stride == 1)
            *((void**)(ptr->data)) = cur_pos;
          else
            *((void**)(ptr->data)) =
              malloc(sizeof(double) * len * stride);
        cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;
    }
    ptr = ptr->next;
  }
  return size;
}




