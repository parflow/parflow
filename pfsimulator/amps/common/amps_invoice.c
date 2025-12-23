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
#include <ctype.h>
#include <stdarg.h>

#include "amps.h"

#define to_digit(c)     ((c) - '0')

void amps_AppendInvoice(
                        amps_Invoice *invoice,
                        amps_Invoice  append_invoice)
{
  if (*invoice)
  {
    (*invoice)->num += append_invoice->num;
    ((*invoice)->end_list)->next = append_invoice->list;
    (*invoice)->end_list = append_invoice->end_list;

    free(append_invoice);
  }
  else
    *invoice = append_invoice;
}

amps_Invoice amps_new_empty_invoice()
{
  amps_Invoice temp;

  if ((temp = (amps_Invoice)calloc(1, sizeof(amps_InvoiceStruct))) == NULL)
    amps_Error("zip_new_empty_invoice", OUT_OF_MEMORY, "", HALT);

  return temp;
}


int amps_FreeInvoice(
                     amps_Invoice inv)
{
  amps_InvoiceEntry *ptr, *next;

  if (inv == NULL)
    return 0;

  /* Delete any storage associated with this invoice */
  amps_ClearInvoice(inv);

  if ((ptr = inv->list) == NULL)
    return 0;

  next = ptr->next;

  while (next)
  {
    free(ptr);
    ptr = next;
    next = ptr->next;
  }

  free(ptr);

  free(inv);

  return 0;
}

int amps_add_invoice(amps_Invoice *inv, int ignore, int type, int len_type, int len, int *ptr_len, int stride_type, int stride, int *ptr_stride, int dim_type, int dim, int *ptr_dim, int data_type, void *data)
{
  amps_InvoiceEntry *ptr, *new_entry;

  if ((new_entry = (amps_InvoiceEntry*)
                   calloc(1, sizeof(amps_InvoiceEntry))) == NULL)
    amps_Error("amps_new_empty_invoice", OUT_OF_MEMORY, "", HALT);

  new_entry->next = NULL;

  new_entry->type = type;

  new_entry->len_type = len_type;
  new_entry->len = len;
  new_entry->ptr_len = ptr_len;

  new_entry->stride_type = stride_type;
  new_entry->stride = stride;
  new_entry->ptr_stride = ptr_stride;


  new_entry->dim_type = dim_type;
  new_entry->ptr_dim = ptr_dim;
  new_entry->dim = dim;


  new_entry->ignore = ignore;

  new_entry->data_type = data_type;
  new_entry->data = data;

  /* check if the invoice is null                                          */
  if (*inv == NULL)
    *inv = amps_new_empty_invoice();

  /* check if list is empty                                                */
  if ((ptr = (*inv)->end_list) == NULL)
    (*inv)->list = (*inv)->end_list = new_entry;
  else
  {
    (*inv)->end_list->next = new_entry;
    (*inv)->end_list = new_entry;
  }

  return TRUE;
}

amps_Invoice amps_NewInvoice(const char *fmt0, ...)
{
  va_list ap;

  short ignore;
  char *fmt;
  int ch;
  int n;
  char *cp;
  int len;
  int *ptr_len = NULL;
  int len_type;
  int stride;
  int *ptr_stride = NULL;
  int stride_type;
  int dim_type = 0;
  int *ptr_dim = NULL;
  int dim = 0;
  void *ptr_data;
  int ptr_data_type;
  int type;
  int num = 0;
  amps_Invoice inv;


  va_start(ap, fmt0);

  inv = NULL;

  fmt = (char*)fmt0;

  for (;;)
  {
    for (cp = fmt; (ch = *fmt) != '\0' && ch != '%'; fmt++)
    {
    }
    ;

#if 0
    for (cp = fmt; (ch = *fmt) != '\0' && ch != '%'; fmt +)
    {
      return 0;                               /*error */
    }
#endif
    if ((n = fmt - cp) != 0)
      /* error condition */
      return 0;

    if (ch == '\0')
      goto done;
    fmt++;              /* skip over '%' */


    ptr_data = NULL;
    stride_type = AMPS_INVOICE_CONSTANT;
    len_type = AMPS_INVOICE_CONSTANT;
    len = 1;
    stride = 1;
    ignore = FALSE;
    ptr_data_type = AMPS_INVOICE_CONSTANT;

rflag:
    ch = *fmt++;
reswitch:
    switch (ch)
    {
      case ' ':
        /*
         * ignore spaces between % and start of format
         */
        goto rflag;

      case '-':
        /* user wants to skip the space */
        ignore = TRUE;
        goto rflag;

      case '*':
        len = va_arg(ap, int);
        goto rflag;

      case '&':
        ptr_len = va_arg(ap, int *);
        len_type = AMPS_INVOICE_POINTER;
        goto rflag;

      case '@':
        /* we are getting a pointer to pointer to data */
        ptr_data_type = AMPS_INVOICE_POINTER;
        goto rflag;

      case '.':
        if ((ch = *fmt++) == '&')
        {
          ptr_stride = va_arg(ap, int *);
          stride_type = AMPS_INVOICE_POINTER;
          goto rflag;
        }
        else if (ch == '*')
        {
          stride = va_arg(ap, int);
          goto rflag;
        }
        stride = 0;
        while (isdigit(ch))
        {
          stride = 10 * stride + to_digit(ch);
          ch = *fmt++;
        }
        goto reswitch;

      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        n = 0;
        do
        {
          n = 10 * n + to_digit(ch);
          ch = *fmt++;
        }
        while (isdigit(ch));
        len = n;
        goto reswitch;

      case 'b':
        type = AMPS_INVOICE_BYTE_CTYPE;
        break;

      case 'c':
        type = AMPS_INVOICE_CHAR_CTYPE;
        break;

      case 's':
        type = AMPS_INVOICE_SHORT_CTYPE;
        break;

      case 'i':
        type = AMPS_INVOICE_INT_CTYPE;
        break;

      case 'l':
        type = AMPS_INVOICE_LONG_CTYPE;
        break;

      case 'd':
        type = AMPS_INVOICE_DOUBLE_CTYPE;
        break;

      case 'f':
        type = AMPS_INVOICE_FLOAT_CTYPE;
        break;

      case 'D':
        type = AMPS_INVOICE_DOUBLE_CTYPE + AMPS_INVOICE_LAST_CTYPE;
        /* skip over "(" */
        fmt++;
        if ((ch = *fmt++) == '&')
        {
          dim_type = AMPS_INVOICE_POINTER;
          ptr_dim = va_arg(ap, int *);
          fmt++;
        }
        else
        {
          dim_type = AMPS_INVOICE_CONSTANT;
          if (ch == '*')
          {
            dim = va_arg(ap, int);
            fmt++;
          }
          else
          {
            dim = 0;
            while (isdigit(ch))
            {
              dim = 10 * dim + to_digit(ch);
              ch = *fmt++;
            }
          }
        }
        break;

      default:
        printf("AMPS Error: invalid invoice specification\n");
        printf("character %c", ch);
        exit(1);
        break;
    }

    /* if user had an extra we already have grabbed the data pointer */
    if (!ptr_data && !ignore)
    {
      if (ptr_data_type == AMPS_INVOICE_POINTER)
      {
        ptr_data = va_arg(ap, void  **);
      }
      else
      {
        ptr_data = va_arg(ap, void *);
      }
    }

    amps_add_invoice(&inv, ignore, type,
                     len_type, len, ptr_len,
                     stride_type, stride, ptr_stride,
                     dim_type, dim, ptr_dim,
                     ptr_data_type, ptr_data);
    num++;
  }

done:

  inv->num = num;

  return inv;
}


/* Returns the number of amps_package items that are in an invoice */
/* This is the number of elements  1 for each dim in > 1           */
/* Used to send/recv information about an invoice during a the init of */
/* an package */
int amps_num_package_items(amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;
  int num = 0;

  ptr = inv->list;

  while (ptr != NULL)
  {
    if (ptr->type > AMPS_INVOICE_LAST_CTYPE)
      num += (ptr->dim_type == AMPS_INVOICE_POINTER) ?
             *(ptr->ptr_dim) : ptr->dim;
    else
      num++;

    ptr = ptr->next;
  }

  return num;
}



