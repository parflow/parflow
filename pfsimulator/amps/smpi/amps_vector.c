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

#include <limits.h>

void amps_vector_in(comm, type, data, buf_ptr, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **buf_ptr;
char **data;
int *len;
int *stride;
{
  int i;

  if (dim == 0)
  {
    switch (type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
        *buf_ptr += AMPS_CALL_BYTE_ALIGN(comm, NULL, *buf_ptr, len[dim],
                                         stride[dim]);
        AMPS_CALL_BYTE_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_BYTE_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                          stride[dim]);
        *(char**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_CHAR_CTYPE:
        *buf_ptr += AMPS_CALL_CHAR_ALIGN(comm, NULL, *buf_ptr, len[dim],
                                         stride[dim]);
        AMPS_CALL_CHAR_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_CHAR_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                          stride[dim]);
        *(char**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        *buf_ptr += AMPS_CALL_SHORT_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_SHORT_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_SHORT_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(short**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_INT_CTYPE:
        *buf_ptr += AMPS_CALL_INT_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_INT_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_INT_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(int**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        *buf_ptr += AMPS_CALL_LONG_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_LONG_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_LONG_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(long**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        *buf_ptr += AMPS_CALL_FLOAT_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_FLOAT_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_FLOAT_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(float**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        *buf_ptr += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_DOUBLE_IN(comm, *buf_ptr, *data, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_DOUBLE_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(double**)data += (len[dim] - 1) * stride[dim];
        break;
    }
  }
  else
  {
    for (i = 0; i < len[dim] - 1; i++)
    {
      amps_vector_in(comm, type, data, buf_ptr, dim - 1, len, stride);

      switch (type)
      {
	case AMPS_INVOICE_BYTE_CTYPE:
          *(char**)data += stride[dim];
          break;

        case AMPS_INVOICE_CHAR_CTYPE:
          *(char**)data += stride[dim];
          break;

        case AMPS_INVOICE_SHORT_CTYPE:
          *(short**)data += stride[dim];
          break;

        case AMPS_INVOICE_INT_CTYPE:
          *(int**)data += stride[dim];
          break;

        case AMPS_INVOICE_LONG_CTYPE:
          *(long**)data += stride[dim];
          break;

        case AMPS_INVOICE_FLOAT_CTYPE:
          *(float**)data += stride[dim];
          break;

        case AMPS_INVOICE_DOUBLE_CTYPE:
          *(double**)data += stride[dim];
          break;
      }
    }

    /* Do one last time without increment of data */
    amps_vector_in(comm, type, data, buf_ptr, dim - 1, len, stride);
  }
}

void amps_vector_out(comm, type, data, buf_ptr, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **buf_ptr;
char **data;
int *len;
int *stride;
{
  int i;

  if (dim == 0)
  {
    switch (type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
        *buf_ptr += AMPS_CALL_BYTE_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_BYTE_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_BYTE_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                          stride[dim]);
        *(char**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_CHAR_CTYPE:
        *buf_ptr += AMPS_CALL_CHAR_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_CHAR_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_CHAR_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                          stride[dim]);
        *(char**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        *buf_ptr += AMPS_CALL_SHORT_ALIGN(comm, NULL, *buf_ptr, len[dim],
                                          stride[dim]);
        AMPS_CALL_SHORT_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_SHORT_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                           stride[dim]);
        *(short**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_INT_CTYPE:
        *buf_ptr += AMPS_CALL_INT_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_INT_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_INT_SIZEOF(comm, *buf_ptr, NULL, len[dim], stride[dim]);
        *(int**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        *buf_ptr += AMPS_CALL_LONG_ALIGN(comm, NULL, *buf_ptr, len[dim], stride[dim]);
        AMPS_CALL_LONG_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_LONG_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                          stride[dim]);
        *(long**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        *buf_ptr += AMPS_CALL_FLOAT_ALIGN(comm, NULL, *buf_ptr, len[dim],
                                          stride[dim]);
        AMPS_CALL_FLOAT_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_FLOAT_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                           stride[dim]);
        *(float**)data += (len[dim] - 1) * stride[dim];
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        *buf_ptr += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, *buf_ptr, len[dim],
                                           stride[dim]);
        AMPS_CALL_DOUBLE_OUT(comm, *data, *buf_ptr, len[dim], stride[dim]);
        *buf_ptr += AMPS_CALL_DOUBLE_SIZEOF(comm, *buf_ptr, NULL, len[dim],
                                            stride[dim]);
        *(double**)data += (len[dim] - 1) * stride[dim];
        break;
    }
  }
  else
  {
    for (i = 0; i < len[dim] - 1; i++)
    {
      amps_vector_out(comm, type, data, buf_ptr, dim - 1, len, stride);

      switch (type)
      {
	case AMPS_INVOICE_BYTE_CTYPE:
          *(char**)data += stride[dim];
          break;

        case AMPS_INVOICE_CHAR_CTYPE:
          *(char**)data += stride[dim];
          break;

        case AMPS_INVOICE_SHORT_CTYPE:
          *(short**)data += stride[dim];
          break;

        case AMPS_INVOICE_INT_CTYPE:
          *(int**)data += stride[dim];
          break;

        case AMPS_INVOICE_LONG_CTYPE:
          *(long**)data += stride[dim];
          break;

        case AMPS_INVOICE_FLOAT_CTYPE:
          *(float**)data += stride[dim];
          break;

        case AMPS_INVOICE_DOUBLE_CTYPE:
          *(double**)data += stride[dim];
          break;
      }
    }

    /* do the last time with no increment of data */
    amps_vector_out(comm, type, data, buf_ptr, dim - 1, len, stride);
  }
}

int amps_vector_align(comm, type, data, buf_ptr, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **buf_ptr;
char **data;
int *len;
int *stride;
{
  int align;
  switch (type)
  {
    case AMPS_INVOICE_BYTE_CTYPE:
      align = AMPS_CALL_BYTE_ALIGN(comm, NULL, *buf_ptr, len[0],
                                   stride[0]);
      break;

    case AMPS_INVOICE_CHAR_CTYPE:
      align = AMPS_CALL_CHAR_ALIGN(comm, NULL, *buf_ptr, len[0],
                                   stride[0]);
      break;

    case AMPS_INVOICE_SHORT_CTYPE:
      align = AMPS_CALL_SHORT_ALIGN(comm, NULL, *buf_ptr, len[0],
                                    stride[0]);
      break;

    case AMPS_INVOICE_INT_CTYPE:
      align = AMPS_CALL_INT_ALIGN(comm, NULL, *buf_ptr, len[0], stride[0]);
      break;

    case AMPS_INVOICE_LONG_CTYPE:
      align = AMPS_CALL_LONG_ALIGN(comm, NULL, *buf_ptr, len[0],
                                   stride[0]);
      break;

    case AMPS_INVOICE_FLOAT_CTYPE:
      align = AMPS_CALL_FLOAT_ALIGN(comm, NULL, *buf_ptr, len[0],
                                    stride[0]);
      break;

    case AMPS_INVOICE_DOUBLE_CTYPE:
      align = AMPS_CALL_DOUBLE_ALIGN(comm, NULL, *buf_ptr, len[0],
                                     stride[0]);
      break;
    default:
      amps_Error("amps_pack", INVALID_INVOICE, "Invalid invoice type", HALT);
      align = INT_MIN;
      break;
  }

  return align;
}

int amps_vector_sizeof_buffer(comm, type, data, buf_ptr, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **buf_ptr;
char **data;
int *len;
int *stride;
{
  int size;
  int i;

  switch (type)
  {
    case AMPS_INVOICE_BYTE_CTYPE:
      size = AMPS_CALL_BYTE_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_CHAR_CTYPE:
      size = AMPS_CALL_CHAR_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_SHORT_CTYPE:
      size = AMPS_CALL_SHORT_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_INT_CTYPE:
      size = AMPS_CALL_INT_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_LONG_CTYPE:
      size = AMPS_CALL_LONG_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_FLOAT_CTYPE:
      size = AMPS_CALL_FLOAT_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;

    case AMPS_INVOICE_DOUBLE_CTYPE:
      size = AMPS_CALL_DOUBLE_SIZEOF(comm, *buf_ptr, NULL, len[0], 1);
      break;
    default:
      amps_Error("amps_pack", INVALID_INVOICE, "Invalid invoice type", HALT);
      size = INT_MIN;
  }

  for (i = 1; i < dim; i++)
  {
    size = size * len[i];
  }

  return size;
}

int amps_vector_sizeof_local(comm, type, data, buf_ptr, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **buf_ptr;
char **data;
int *len;
int *stride;
{
  int size = 0;
  int el_size = 0;
  int i;

  switch (type)
  {
    case AMPS_INVOICE_BYTE_CTYPE:
      size = AMPS_CALL_BYTE_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                   stride[0]);
      el_size = sizeof(char);
      break;

    case AMPS_INVOICE_CHAR_CTYPE:
      size = AMPS_CALL_CHAR_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                   stride[0]);
      el_size = sizeof(char);
      break;

    case AMPS_INVOICE_SHORT_CTYPE:
      size = AMPS_CALL_SHORT_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                    stride[0]);
      el_size = sizeof(short);
      break;

    case AMPS_INVOICE_INT_CTYPE:
      size = AMPS_CALL_INT_SIZEOF(comm, *buf_ptr, NULL, len[0], stride[0]);
      el_size = sizeof(int);
      break;

    case AMPS_INVOICE_LONG_CTYPE:
      size = AMPS_CALL_LONG_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                   stride[0]);
      el_size = sizeof(long);
      break;

    case AMPS_INVOICE_FLOAT_CTYPE:
      size = AMPS_CALL_FLOAT_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                    stride[0]);
      el_size = sizeof(float);
      break;

    case AMPS_INVOICE_DOUBLE_CTYPE:
      size = AMPS_CALL_DOUBLE_SIZEOF(comm, *buf_ptr, NULL, len[0],
                                     stride[0]);
      el_size = sizeof(double);
      break;
  }

  for (i = 1; i < dim; i++)
  {
    size = size * len[i] + stride[i] * (len[i] - 1) * el_size;
  }

  return size;
}
