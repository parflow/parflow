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

/**
 *
 * \Ref{amps_SFBCast} is used to read data from a shared file.  Note
 * that the input is described by an \Ref{amps_Invoice} rather than the
 * standard {\bf printf} syntax.  This is to allow a closer mapping to
 * the communication routines.  Due to this change be careful; items in
 * the input file must match what is in the invoice description.  As it's
 * name implies this function reads from a file and broadcasts the data
 * in the file to all the nodes who are in the {\bf comm} context.  Think
 * of it as doing an \Ref{amps_BCAST} with a file replacing the node as
 * the source.  The data is stored in ASCII format and read in using
 * the Standard C library function {\bf scanf} so it's formatting rules
 * apply.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_File file;
 * amps_Invoice invoice;
 *
 * file = amps_SFopen(filename, "r");
 *
 * amps_SFBCast(amps_CommWorld, file, invoice);
 *
 * amps_SFclose(file);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Broadcast from a shared file
 * @param comm Communication context [IN]
 * @param file Shared file handle [IN]
 * @param invoice Descriptions of data to read from file and distribute [IN/OUT]
 * @return Error code
 */
int amps_SFBCast(amps_Comm comm, amps_File file, amps_Invoice invoice)
{
  amps_InvoiceEntry *ptr;
  int stride, len;

  if (!amps_Rank(comm))
  {
    amps_ClearInvoice(invoice);

    invoice->combuf_flags |= AMPS_INVOICE_NON_OVERLAYED;

    invoice->comm = comm;

    /* for each entry in the invoice read the value from the input file */
    ptr = invoice->list;
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
        case AMPS_INVOICE_BYTE_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(char) * (size_t)(len * stride));
            amps_ScanByte(file, *( char**)(ptr->data), len, stride);
          }
          else
            amps_ScanByte(file, (char*)ptr->data, len, stride);
          break;

        case AMPS_INVOICE_CHAR_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(char) * (size_t)(len * stride));
            amps_ScanChar(file, *( char**)(ptr->data), len, stride);
          }
          else
            amps_ScanChar(file, (char*)ptr->data, len, stride);
          break;

        case AMPS_INVOICE_SHORT_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(short) * (size_t)(len * stride));
            amps_ScanShort(file, *( short**)(ptr->data), len, stride);
          }
          else
            amps_ScanShort(file, (short*)ptr->data, len, stride);

          break;

        case AMPS_INVOICE_INT_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(int) * (size_t)(len * stride));
            amps_ScanInt(file, *( int**)(ptr->data), len, stride);
          }
          else
            amps_ScanInt(file, (int*)ptr->data, len, stride);

          break;

        case AMPS_INVOICE_LONG_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(long) * (size_t)(len * stride));
            amps_ScanLong(file, *( long**)(ptr->data), len, stride);
          }
          else
            amps_ScanLong(file, (long*)ptr->data, len, stride);

          break;

        case AMPS_INVOICE_FLOAT_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(float) * (size_t)(len * stride));
            amps_ScanFloat(file, *( float**)(ptr->data), len, stride);
          }
          else
            amps_ScanFloat(file, (float*)ptr->data, len, stride);

          break;

        case AMPS_INVOICE_DOUBLE_CTYPE:
          if (ptr->data_type == AMPS_INVOICE_POINTER)
          {
            *((void**)(ptr->data)) = (void*)malloc(sizeof(double) * (size_t)(len * stride));
            amps_ScanDouble(file, *( double**)(ptr->data), len, stride);
          }
          else
            amps_ScanDouble(file, (double*)ptr->data, len, stride);

          break;
      }
      ptr = ptr->next;
    }
  }

  return amps_BCast(comm, 0, invoice);
}


