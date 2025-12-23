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

#include <string.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

int amps_ReduceOperation(comm, invoice, buf_dest, buf_src, operation)
amps_Comm comm;
amps_Invoice invoice;
char *buf_dest, *buf_src;
int operation;
{
  amps_InvoiceEntry *ptr;
  char *pos_dest, *pos_src;
  char *end_dest;
  int len;

  if (operation)
  {
    ptr = invoice->list;
    pos_dest = buf_dest;
    pos_src = buf_src;

    while (ptr != NULL)
    {
      if (ptr->len_type == AMPS_INVOICE_POINTER)
        len = *(ptr->ptr_len);
      else
        len = ptr->len;

      switch (operation)
      {
        case amps_Max:
          switch (ptr->type)
          {
            case AMPS_INVOICE_BYTE_CTYPE:
              pos_dest += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest =
                  max(*(char*)pos_dest, *(char*)pos_src);
              break;

            case AMPS_INVOICE_CHAR_CTYPE:
              pos_dest += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest =
                  max(*(char*)pos_dest, *(char*)pos_src);
              break;

            case AMPS_INVOICE_SHORT_CTYPE:
              pos_dest += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(short);
                   pos_dest < end_dest;
                   pos_dest += sizeof(short), pos_src += sizeof(short))
                *(short*)pos_dest =
                  max(*(short*)pos_dest, *(short*)pos_src);
              break;

            case AMPS_INVOICE_INT_CTYPE:
              pos_dest += AMPS_CALL_INT_ALIGN(comm, NULL,
                                              pos_dest, len, 1);
              pos_src += AMPS_CALL_INT_ALIGN(comm, NULL,
                                             pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(int);
                   pos_dest < end_dest;
                   pos_dest += sizeof(int), pos_src += sizeof(int))
                *(int*)pos_dest =
                  max(*(int*)pos_dest, *(int*)pos_src);
              break;

            case AMPS_INVOICE_LONG_CTYPE:
              pos_dest += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(long);
                   pos_dest < end_dest;
                   pos_dest += sizeof(long), pos_src += sizeof(long))
                *(long*)pos_dest =
                  max(*(long*)pos_dest, *(long*)pos_src);
              break;

            case AMPS_INVOICE_FLOAT_CTYPE:
              pos_dest += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(float);
                   pos_dest < end_dest;
                   pos_dest += sizeof(float), pos_src += sizeof(float))
                *(float*)pos_dest =
                  max(*(float*)pos_dest, *(float*)pos_src);
              break;

            case AMPS_INVOICE_DOUBLE_CTYPE:
              pos_dest += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                 pos_dest, len, 1);
              pos_src += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(double);
                   pos_dest < end_dest;
                   pos_dest += sizeof(double), pos_src += sizeof(double))
                *(double*)pos_dest =
                  max(*(double*)pos_dest, *(double*)pos_src);
              break;
          }
          break;

        case amps_Min:
          switch (ptr->type)
          {
            case AMPS_INVOICE_BYTE_CTYPE:
              pos_dest += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest =
                  min(*(char*)pos_dest, *(char*)pos_src);
              break;

            case AMPS_INVOICE_CHAR_CTYPE:
              pos_dest += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest =
                  min(*(char*)pos_dest, *(char*)pos_src);
              break;

            case AMPS_INVOICE_SHORT_CTYPE:
              pos_dest += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(short);
                   pos_dest < end_dest;
                   pos_dest += sizeof(short), pos_src += sizeof(short))
                *(short*)pos_dest =
                  min(*(short*)pos_dest, *(short*)pos_src);
              break;

            case AMPS_INVOICE_INT_CTYPE:
              pos_dest += AMPS_CALL_INT_ALIGN(comm, NULL,
                                              pos_dest, len, 1);
              pos_src += AMPS_CALL_INT_ALIGN(comm, NULL,
                                             pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(int);
                   pos_dest < end_dest;
                   pos_dest += sizeof(int), pos_src += sizeof(int))
                *(int*)pos_dest =
                  min(*(int*)pos_dest, *(int*)pos_src);
              break;

            case AMPS_INVOICE_LONG_CTYPE:
              pos_dest += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(long);
                   pos_dest < end_dest;
                   pos_dest += sizeof(long), pos_src += sizeof(long))
                *(long*)pos_dest =
                  min(*(long*)pos_dest, *(long*)pos_src);
              break;

            case AMPS_INVOICE_FLOAT_CTYPE:
              pos_dest += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(float);
                   pos_dest < end_dest;
                   pos_dest += sizeof(float), pos_src += sizeof(float))
                *(float*)pos_dest =
                  min(*(float*)pos_dest, *(float*)pos_src);
              break;

            case AMPS_INVOICE_DOUBLE_CTYPE:
              pos_dest += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                 pos_dest, len, 1);
              pos_src += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(double);
                   pos_dest < end_dest;
                   pos_dest += sizeof(double), pos_src += sizeof(double))
                *(double*)pos_dest =
                  min(*(double*)pos_dest, *(double*)pos_src);
              break;
          }
          break;

        case amps_Add:
          switch (ptr->type)
          {
            case AMPS_INVOICE_BYTE_CTYPE:
              pos_dest += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_BYTE_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest += *(char*)pos_src;
              break;

            case AMPS_INVOICE_CHAR_CTYPE:
              pos_dest += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_CHAR_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(char);
                   pos_dest < end_dest;
                   pos_dest += sizeof(char), pos_src += sizeof(char))
                *(char*)pos_dest += *(char*)pos_src;
              break;

            case AMPS_INVOICE_SHORT_CTYPE:
              pos_dest += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_SHORT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(short);
                   pos_dest < end_dest;
                   pos_dest += sizeof(short), pos_src += sizeof(short))
                *(short*)pos_dest += *(short*)pos_src;
              break;

            case AMPS_INVOICE_INT_CTYPE:
              pos_dest += AMPS_CALL_INT_ALIGN(comm, NULL,
                                              pos_dest, len, 1);
              pos_src += AMPS_CALL_INT_ALIGN(comm, NULL,
                                             pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(int);
                   pos_dest < end_dest;
                   pos_dest += sizeof(int), pos_src += sizeof(int))
                *(int*)pos_dest += *(int*)pos_src;
              break;

            case AMPS_INVOICE_LONG_CTYPE:
              pos_dest += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                               pos_dest, len, 1);
              pos_src += AMPS_CALL_LONG_ALIGN(comm, NULL,
                                              pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(long);
                   pos_dest < end_dest;
                   pos_dest += sizeof(long), pos_src += sizeof(long))
                *(long*)pos_dest += *(long*)pos_src;
              break;

            case AMPS_INVOICE_FLOAT_CTYPE:
              pos_dest += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                                pos_dest, len, 1);
              pos_src += AMPS_CALL_FLOAT_ALIGN(comm, NULL,
                                               pos_src, len, 1);

              for (end_dest = pos_dest + len * sizeof(float);
                   pos_dest < end_dest;
                   pos_dest += sizeof(float), pos_src += sizeof(float))
                *(float*)pos_dest += *(float*)pos_src;
              break;

            case AMPS_INVOICE_DOUBLE_CTYPE:
              pos_dest += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                 pos_dest, len, 1);
              pos_src += AMPS_CALL_DOUBLE_ALIGN(comm, NULL,
                                                pos_src, len, 1);


              for (end_dest = pos_dest + len * sizeof(double);
                   pos_dest < end_dest;
                   pos_dest += sizeof(double), pos_src += sizeof(double))
                *(double*)pos_dest += *(double*)pos_src;
              break;

            default:
            {
              amps_Error("amps_pack", INVALID_INVOICE, "Invalid invoice type", HALT);
              pos_dest = 0;
              pos_src = 0;
            }
            break;
          }
          break;
      }

      ptr = ptr->next;
    }
    return 0;
  }
  else
    return 0;
}

int amps_AllReduce(comm, invoice, operation)
amps_Comm comm;
amps_Invoice invoice;
int operation;
{
  int n;
  int N;
  int d;
  int poft, log, npoft;
  int node;

  char *l_buffer;
  char *r_buffer;

  int size;

  N = amps_size;
  n = amps_rank;

  amps_FindPowers(N, &log, &npoft, &poft);

  /* nothing to do if only one node */
  if (N < 2)
    return 0;

  if (n < poft)
  {
    size = amps_pack(comm, invoice, &l_buffer);

    if (n < N - poft)
    {
      node = poft + n;

      r_buffer = amps_recvb(node, &size);

      amps_ReduceOperation(comm, invoice, l_buffer, r_buffer, operation);
    }
    else
      r_buffer = amps_new(comm, size);

    for (d = 1; d < poft; d <<= 1)
    {
      node = (n ^ d);

      memcpy(r_buffer, l_buffer, size);

      amps_xsend(comm, node, r_buffer, size);

      amps_free(comm, r_buffer);

      r_buffer = amps_recvb(node, &size);

      amps_ReduceOperation(comm, invoice, l_buffer, r_buffer, operation);
    }

    if (n < N - poft)
    {
      node = poft + n;

      memcpy(r_buffer, l_buffer, size);

      amps_xsend(comm, node, r_buffer, size);

      amps_free(comm, r_buffer);
    }
    else
      amps_free(comm, r_buffer);

    amps_unpack(comm, invoice, l_buffer);

    AMPS_PACK_FREE_LETTER(comm, invoice, l_buffer)
  }
  else
  {
    amps_Send(comm, n - poft, invoice);
    amps_Recv(comm, n - poft, invoice);
  }

  return 0;
}

