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


int amps_unpack(comm, inv, buffer)
amps_Comm comm;
amps_Invoice inv;
char *buffer;
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  int len, stride;
  int malloced = FALSE;
  int size;
  char *data;
  char *temp_pos;
  int dim;

  /* we are unpacking so signal this operation */

  inv->flags &= ~AMPS_PACKED;

  /* for each entry in the invoice pack that entry into the letter         */
  ptr = inv->list;
  cur_pos = buffer;
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

    switch (ptr->type)
    {
      case AMPS_INVOICE_CHAR_CTYPE:
        cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_CHAR_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(char)
                                              * len * stride);
            malloced = TRUE;
            AMPS_CALL_CHAR_IN(comm, cur_pos, *((void**)(ptr->data)),
                              len, stride);
          }
          else
            AMPS_CALL_CHAR_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_SHORT_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(short) *
                                              len * stride);
            malloced = TRUE;
            AMPS_CALL_SHORT_IN(comm, cur_pos, *((void**)(ptr->data)),
                               len, stride);
          }
          else
            AMPS_CALL_SHORT_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_INT_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(int) *
                                              len * stride);
            malloced = TRUE;
            AMPS_CALL_INT_IN(comm, cur_pos, *((void**)(ptr->data)),
                             len, stride);
          }
          else
            AMPS_CALL_INT_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_LONG_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(long) *
                                              len * stride);
            malloced = TRUE;
            AMPS_CALL_LONG_IN(comm, cur_pos, *((void**)(ptr->data)),
                              len, stride);
          }
          else
            AMPS_CALL_LONG_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_FLOAT_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(float) *
                                              len * stride);
            malloced = TRUE;

            AMPS_CALL_FLOAT_IN(comm, cur_pos, *((void**)(ptr->data)),
                               len, stride);
          }
          else
            AMPS_CALL_FLOAT_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, stride);
        if (!ptr->ignore)
        {
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            if (stride == 1 && AMPS_DOUBLE_OVERLAY(comm))
              *((void**)(ptr->data)) = cur_pos;
            else
              *((void**)(ptr->data)) = malloc(sizeof(double) *
                                              len * stride);
            malloced = TRUE;

            AMPS_CALL_DOUBLE_IN(comm, cur_pos, *((void**)(ptr->data)),
                                len, stride);
          }
          else
            AMPS_CALL_DOUBLE_IN(comm, cur_pos, ptr->data, len, stride);
        }
        cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, stride);
        break;

      default:
        dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ?
              *(ptr->ptr_dim) : ptr->dim;

        cur_pos += amps_vector_align(comm,
                                     ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                     NULL, &cur_pos, dim,
                                     ptr->ptr_len, ptr->ptr_stride);
        size = amps_vector_sizeof_local(comm,
                                        ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                        NULL, &temp_pos, dim,
                                        ptr->ptr_len, ptr->ptr_stride);

        if (ptr->data_type == AMPS_INVOICE_POINTER)
          data = *(char**)(ptr->data) = (char*)malloc(size);
        else
          data = ptr->data;

        temp_pos = cur_pos;
        amps_vector_in(comm, ptr->type - AMPS_INVOICE_LAST_CTYPE,
                       &data, &temp_pos, dim - 1, ptr->ptr_len,
                       ptr->ptr_stride);

        cur_pos += amps_vector_sizeof_buffer(comm,
                                             ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                             &data, &temp_pos, dim,
                                             ptr->ptr_len, ptr->ptr_stride);
    }
    ptr = ptr->next;
  }

  if (malloced)
  {
    inv->combuf_flags |= AMPS_INVOICE_OVERLAYED;
    inv->combuf = buffer;
    inv->comm = comm;
  }
  return 0;
}

