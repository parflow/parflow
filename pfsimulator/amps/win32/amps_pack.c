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

int amps_pack(comm, inv, buffer)
amps_Comm comm;
amps_Invoice inv;
char **buffer;
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  char *temp_pos;
  char *data;
  int size, stride;

  int len;
  int dim;

  size = amps_sizeof_invoice(comm, inv);

  inv->flags |= AMPS_PACKED;

  /* check to see if this was already allocated                            */
  if ((inv->combuf_flags & AMPS_INVOICE_ALLOCATED))
  {
    *buffer = inv->combuf;
  }
  else
  {
    if ((*buffer = amps_new(comm, size)) == NULL)
      amps_Error("amps_pack", OUT_OF_MEMORY, "malloc of letter", HALT);
    else
      inv->combuf_flags |= AMPS_INVOICE_ALLOCATED;
  }

  /* for each entry in the invoice pack that entry into the letter         */
  ptr = inv->list;
  cur_pos = *buffer;
  while (ptr != NULL)
  {
    /* invoke the packing convert out for the entry */
    /* if user then call user ones */
    /* else switch on builtin type */
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
      data = ptr->data;

    switch (ptr->type)
    {
      case AMPS_INVOICE_CHAR_CTYPE:

        cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_CHAR_OUT(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_SHORT_OUT(comm, data, cur_pos, len, stride);
        cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_INT_OUT(comm, data, cur_pos, len,
                            stride);
        cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_LONG_OUT(comm, data, cur_pos, len,
                             stride);
        cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_FLOAT_OUT(comm, data, cur_pos, len,
                              stride);
        cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
          AMPS_CALL_DOUBLE_OUT(comm, data, cur_pos, len,
                               stride);
        cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      default:
        dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ?
              *(ptr->ptr_dim) : ptr->dim;
        temp_pos = cur_pos;
        cur_pos += amps_vector_align(comm,
                                     ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                     &data, &cur_pos, dim,
                                     ptr->ptr_len, ptr->ptr_stride);
        temp_pos = cur_pos;
        amps_vector_out(comm, ptr->type - AMPS_INVOICE_LAST_CTYPE,
                        &data, &temp_pos, dim - 1, ptr->ptr_len,
                        ptr->ptr_stride);
        temp_pos = cur_pos;
        cur_pos += amps_vector_sizeof_buffer(comm,
                                             ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                             &data, &temp_pos, dim,
                                             ptr->ptr_len, ptr->ptr_stride);
    }
    ptr = ptr->next;
  }

  return size;
}


