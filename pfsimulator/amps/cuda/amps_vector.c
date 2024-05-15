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
#include <strings.h>

int amps_vector_align(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride)
{
  int align = 0;

  (void)comm;
  (void)data;
  (void)dim;
  (void)len;
  (void)stride;

  switch (type)
  {
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
  }

  return align;
}

int amps_vector_sizeof_buffer(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride)
{
  unsigned int size = 0;
  int i;

  (void)comm;
  (void)data;
  (void)buf_ptr;
  (void)stride;

  switch (type)
  {
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
  }

  for (i = 1; i < dim; i++)
  {
    size = size * (size_t)len[i];
  }

  return (int)size;
}

int amps_vector_sizeof_local(amps_Comm comm, int type, char **data, char **buf_ptr, int dim, int *len, int *stride)
{
  unsigned int size = 0;
  unsigned int el_size = 0;
  int i;

  (void)comm;
  (void)data;
  (void)buf_ptr;

  switch (type)
  {
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
    size = size * (size_t)(len[i]) + (size_t)(stride[i] * (len[i] - 1)) * el_size;
  }

  return (int)size;
}
