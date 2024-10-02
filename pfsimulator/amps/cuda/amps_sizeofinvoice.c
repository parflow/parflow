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

long amps_sizeof_invoice(
                         amps_Comm    comm,
                         amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;
  char *cur_pos = 0;
  char *temp_pos = 0;
  int len, stride;
  char *data;

  PF_UNUSED(stride);

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

    if (ptr->data_type == AMPS_INVOICE_POINTER)
      data = *((char**)(ptr->data));
    else
      data = (char*)ptr->data;

    switch (ptr->type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
        cur_pos += AMPS_CALL_BYTE_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_BYTE_SIZEOF(comm, data, cur_pos,
                                         len, stride);
        break;

      case AMPS_INVOICE_CHAR_CTYPE:
        cur_pos += AMPS_CALL_CHAR_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, data, cur_pos,
                                         len, stride);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        cur_pos += AMPS_CALL_SHORT_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, data, cur_pos,
                                          len, stride);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        cur_pos += AMPS_CALL_INT_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_INT_SIZEOF(comm, data, cur_pos,
                                        len, stride);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        cur_pos += AMPS_CALL_LONG_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_LONG_SIZEOF(comm, data, cur_pos,
                                         len, stride);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, data, cur_pos,
                                          len, stride);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, data, cur_pos,
                                           len, stride);
        break;

      default:
        temp_pos = cur_pos;
        cur_pos += amps_vector_align(comm,
                                     ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                     &data, &temp_pos, (int)ptr->dim,
                                     ptr->ptr_len, ptr->ptr_stride);
        temp_pos = cur_pos;
        cur_pos += amps_vector_sizeof_buffer(comm,
                                             ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                             &data, &temp_pos, (int)ptr->dim,
                                             ptr->ptr_len, ptr->ptr_stride);
    }

    ptr = ptr->next;
  }
  return (long)cur_pos;
}
