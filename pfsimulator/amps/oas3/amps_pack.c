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

#if MPI_VERSION < 2
#define MPI_Get_address(location, address) MPI_Address((location), (address))
#define MPI_Type_create_hvector(count, blocklength, stride, oldtype, newtype) MPI_Type_hvector((count), (blocklength), (stride), (oldtype), (newtype))
#define MPI_Type_create_struct(count, array_of_blocklengths, array_of_displacements, array_of_types, newtype) MPI_Type_struct((count), (array_of_blocklengths), (array_of_displacements), (array_of_types), (newtype))
#endif

int amps_create_mpi_cont_send_type(
                                   amps_Comm    comm,
                                   amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;
  char *cur_pos;
  char *temp_pos;
  char *data;
  int size = 0;

  int len;
  int dim;

  int i;

  int element;

  MPI_Datatype *mpi_types;
  MPI_Aint     *mpi_displacements;
  int          *mpi_block_len;

  MPI_Datatype *base_type;
  MPI_Datatype *temp_type;
  MPI_Datatype *new_type = NULL;

  mpi_types = (MPI_Datatype*)calloc((size_t)(inv->num), sizeof(MPI_Datatype));
  mpi_block_len = (int*)malloc(sizeof(int) * (size_t)(inv->num));

  mpi_displacements = (MPI_Aint*)calloc((size_t)(inv->num), sizeof(MPI_Aint));

  /* for each entry in the invoice pack that entry into the letter         */
  ptr = inv->list;
  cur_pos = 0;
  element = 0;
  while (ptr != NULL)
  {
    /* invoke the packing convert out for the entry */
    /* if user then call user ones */
    /* else switch on builtin type */
    if (ptr->len_type == AMPS_INVOICE_POINTER)
      len = *(ptr->ptr_len);
    else
      len = ptr->len;

    switch (ptr->type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
        cur_pos += AMPS_CALL_BYTE_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_BYTE, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_BYTE_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;
	
      case AMPS_INVOICE_CHAR_CTYPE:
        cur_pos += AMPS_CALL_CHAR_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_BYTE, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_CHAR_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        cur_pos += AMPS_CALL_SHORT_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_SHORT, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_SHORT_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      case AMPS_INVOICE_INT_CTYPE:
        cur_pos += AMPS_CALL_INT_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_INT, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_INT_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        cur_pos += AMPS_CALL_LONG_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_LONG, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_LONG_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        cur_pos += AMPS_CALL_FLOAT_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_FLOAT, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_FLOAT_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        cur_pos += AMPS_CALL_DOUBLE_ALIGN(comm, NULL, cur_pos, len, 1);
        MPI_Type_vector(len, 1, 1, MPI_DOUBLE, &mpi_types[element]);
        MPI_Get_address(cur_pos, &mpi_displacements[element]);
        cur_pos += AMPS_CALL_DOUBLE_SIZEOF(comm, cur_pos, NULL, len, 1);
        break;

      default:
        dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ?
              *(ptr->ptr_dim) : ptr->dim;

        /* Temporary datatypes for constructing vectors */
        if (dim > 1)
        {
          base_type = (MPI_Datatype*)calloc(1, sizeof(MPI_Datatype));
          new_type = (MPI_Datatype*)calloc(1, sizeof(MPI_Datatype));
        }
        else
          base_type = &mpi_types[element];

        len = ptr->ptr_len[dim - 1];

        switch (ptr->type - AMPS_INVOICE_LAST_CTYPE)
        {
	  case AMPS_INVOICE_BYTE_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_BYTE, base_type);
            }
            break;
	    
          case AMPS_INVOICE_CHAR_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_CHAR, base_type);
            }
            break;

          case AMPS_INVOICE_SHORT_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_SHORT, base_type);
            }
            break;

          case AMPS_INVOICE_INT_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_INT, base_type);
            }
            break;

          case AMPS_INVOICE_LONG_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_LONG, base_type);
            }
            break;

          case AMPS_INVOICE_FLOAT_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_FLOAT, base_type);
            }
            break;

          case AMPS_INVOICE_DOUBLE_CTYPE:
            if (!ptr->ignore)
            {
              MPI_Type_vector(len, 1, 1, MPI_DOUBLE, base_type);
            }
            break;
        }

        temp_pos = cur_pos;
        cur_pos += amps_vector_align(comm,
                                     ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                     &data, &cur_pos, dim,
                                     ptr->ptr_len, ptr->ptr_stride);
        for (i = dim - 2; i >= 0; i--)
        {
          if (i == 0)
          {
            MPI_Type_create_hvector(ptr->ptr_len[i], 1, 0,
                                    *base_type, &mpi_types[element]);
            MPI_Type_free(base_type);
          }
          else
          {
            MPI_Type_create_hvector(ptr->ptr_len[i], 1, 0,
                                    *base_type, new_type);
            MPI_Type_free(base_type);
            temp_type = base_type;
            base_type = new_type;
            new_type = temp_type;
          }
        }

        if (dim > 1)
        {
          free(base_type);
        }

        free(new_type);

        temp_pos = cur_pos;
        cur_pos += amps_vector_sizeof_local(comm,
                                            ptr->type - AMPS_INVOICE_LAST_CTYPE,
                                            NULL, &temp_pos, dim,
                                            ptr->ptr_len, ptr->ptr_stride);


        break;
    }

    mpi_block_len[element] = 1;
    element++;
    ptr = ptr->next;
  }

  return size;
}

void amps_create_mpi_type(
                          amps_Comm    comm,
                          amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;
  char *data;
  int element_size = 0;
  int stride;
  int len;
  int dim;

  int i;

  int element;

  int base_size;

  MPI_Datatype *mpi_types;
  MPI_Aint     *mpi_displacements;
  int          *mpi_block_len;

  MPI_Datatype *base_type;
  MPI_Datatype *temp_type;
  MPI_Datatype *new_type;

  (void)comm;

  mpi_types = (MPI_Datatype*)calloc((size_t)(inv->num), sizeof(MPI_Datatype));
  mpi_block_len = (int*)malloc(sizeof(int) * (size_t)(inv->num));
  mpi_displacements = (MPI_Aint*)calloc((size_t)(inv->num), sizeof(MPI_Aint));

  /* for each entry in the invoice pack that entry into the letter         */

  element = 0;
  ptr = inv->list;
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
      data = (char*)ptr->data;

    switch (ptr->type)
    {
      case AMPS_INVOICE_BYTE_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_BYTE,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_CHAR_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_CHAR,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_SHORT_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_SHORT,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_INT_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_INT,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_LONG_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_LONG,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_FLOAT_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_FLOAT,
                          &mpi_types[element]);
        }
        break;

      case AMPS_INVOICE_DOUBLE_CTYPE:
        if (!ptr->ignore)
        {
          MPI_Type_vector(len, 1, stride, MPI_DOUBLE,
                          &mpi_types[element]);
        }
        break;

      default:
        dim = (ptr->dim_type == AMPS_INVOICE_POINTER) ?
              *(ptr->ptr_dim) : ptr->dim;

        /* Temporary datatypes for constructing vectors */
        if (dim > 1)
          base_type = (MPI_Datatype*)calloc(1, sizeof(MPI_Datatype));
        else
          base_type = &mpi_types[element];

        new_type = (MPI_Datatype*)calloc(1, sizeof(MPI_Datatype));

        len = ptr->ptr_len[0];
        stride = ptr->ptr_stride[0];

        switch (ptr->type - AMPS_INVOICE_LAST_CTYPE)
        {
	  case AMPS_INVOICE_BYTE_CTYPE:
	    MPI_Type_vector(len, 1, stride, MPI_BYTE, base_type);
            element_size = sizeof(char);
            break;

          case AMPS_INVOICE_CHAR_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_CHAR, base_type);
            element_size = sizeof(char);
            break;

          case AMPS_INVOICE_SHORT_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_SHORT, base_type);
            element_size = sizeof(short);
            break;

          case AMPS_INVOICE_INT_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_INT, base_type);
            element_size = sizeof(int);
            break;

          case AMPS_INVOICE_LONG_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_LONG, base_type);
            element_size = sizeof(long);
            break;

          case AMPS_INVOICE_FLOAT_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_FLOAT, base_type);
            element_size = sizeof(float);
            break;

          case AMPS_INVOICE_DOUBLE_CTYPE:
            MPI_Type_vector(len, 1, stride, MPI_DOUBLE, base_type);
            element_size = sizeof(double);
            break;
        }

        base_size = element_size * (len + (len - 1) * (stride - 1));

        for (i = 1; i < dim; i++)
        {
          if (i == dim - 1)
          {
            MPI_Type_create_hvector(ptr->ptr_len[i], 1,
                                    base_size +
                                    (ptr->ptr_stride[i] - 1) * element_size,
                                    *base_type, &mpi_types[element]);
            base_size = base_size * ptr->ptr_len[i]
                        + (ptr->ptr_stride[i] - 1) * (ptr->ptr_len[i] - 1)
                        * element_size;
            MPI_Type_free(base_type);
          }
          else
          {
            MPI_Type_create_hvector(ptr->ptr_len[i], 1,
                                    base_size +
                                    (ptr->ptr_stride[i] - 1) * element_size,
                                    *base_type, new_type);
            base_size = base_size * ptr->ptr_len[i]
                        + (ptr->ptr_stride[i] - 1) * (ptr->ptr_len[i] - 1)
                        * element_size;
            MPI_Type_free(base_type);
            temp_type = base_type;
            base_type = new_type;
            new_type = temp_type;
          }
        }

        if (dim > 1)
        {
          free(base_type);
        }

        free(new_type);

        break;
    }

    MPI_Get_address(data, &mpi_displacements[element]);

    mpi_block_len[element] = 1;
    element++;
    ptr = ptr->next;
  }

  MPI_Type_create_struct(inv->num,
                         mpi_block_len,
                         mpi_displacements,
                         mpi_types,
                         &inv->mpi_type);

  for (element = 0; element < inv->num; element++)
  {
    MPI_Type_free(&mpi_types[element]);
  }

  free(mpi_block_len);
  free(mpi_displacements);
  free(mpi_types);
}

