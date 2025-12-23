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

/*===========================================================================*/
/**
 *
 * When an invoice is no longer needed it should be freed with
 * \Ref{amps_FreeInvoice}.  Access to overlayed variables in an invoice is
 * not allowed after the invoice has been freed.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int me, i;
 * double d;
 *
 * me = amps_Rank(amps_CommWorld);
 *
 * invoice = amps_NewInvoice("%i%d", &i, &d);
 *
 * amps_Send(amps_CommWorld, me+1, invoice);
 *
 * amps_Recv(amps_CommWorld, me-1, invoice);
 *
 * amps_FreeInvoice(invoice);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Release storage related to an \Ref{amps_Invoice}
 * @param inv The invoice to free [IN/OUT]
 * @return Error code
 */
int amps_FreeInvoice(amps_Invoice inv)
{
  amps_InvoiceEntry *ptr, *next;

  if (inv == NULL)
    return 0;

  /* Delete any storage associated with this invoice */
  amps_ClearInvoice(inv);

  if (inv->mpi_type != MPI_DATATYPE_NULL && inv->mpi_type != MPI_BYTE)
  {
    MPI_Type_free(&inv->mpi_type);
  }

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
  {
    (*inv)->list = (*inv)->end_list = new_entry;
  }
  else
  {
    (*inv)->end_list->next = new_entry;
    (*inv)->end_list = new_entry;
  }

  return TRUE;
}


/*===========================================================================*/
/**
 *
 * The \Ref{amps_AppendInvoice} appends more data references to an
 * existing invoice.  It can be used to build up and invoice over several
 * function calls or in a loop.  The first argument is a pointer to the
 * invoice which is appended to.  The second argument is the invoice to
 * attach to the first.  The two invoices must have already been created
 * using the \Ref{amps_NewInvoice} routine.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice a;
 * amps_Invoice b;
 *
 * a = amps_NewInvoice("%d", &d1);
 * b = amps_NewInvoice("%d", &d2);
 *
 *
 * amps_AppendInvoice(&a, b);
 * // The "a" invoice will now communicate both d1 and d2
 *
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * @memo Append an invoice to another one
 * @param invoice The invoice to be appended to [IN/OUT]
 * @param append_invoice the invoice to be appended [IN]
 * @return */
void amps_AppendInvoice(amps_Invoice *invoice, amps_Invoice append_invoice)
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
  amps_Invoice temp = NULL;

  if ((temp = (amps_Invoice)calloc(1, sizeof(amps_InvoiceStruct))) == NULL)
    amps_Error("zip_new_empty_invoice", OUT_OF_MEMORY, "", HALT);

  temp->mpi_type = MPI_DATATYPE_NULL;

  return temp;
}


/*===========================================================================*/
/**
 *
 * Invoice creation is the most difficult part of {\em AMPS}.  Part of the
 * difficulty of invoice construction arises from the many options that
 * were included.  While a simpler interface is possible we wanted to
 * develop a system which was flexible enough to enable efficient
 * implementations on existing and future platforms.
 *
 * \Ref{amps_NewInvoice} constructs an \Ref{amps_Invoice}.  It takes a
 * variable number of arguments, starting with a format string
 * ({\bf fmt}) similar to the commonly used {\bf printf} strings.  The
 * format string contains one or more conversion specifications.  A
 * conversion specification is introduced by a percent sign {\bf \%} and
 * is followed by
 *
 * \begin{itemize}
 *
 * \item
 *      A positive integer indicating the length (number of items to
 *      convert), or a `{\bf \*}' or `{\bf \&}' indicating
 *      argument-list specification of an integer expression or
 *      address (see below).  If no integer is specified the default
 *      is one item.
 *
 * \item
 *      An optional stride factor indicated by a `.' followed by a
 *      positive integer indicating the stride; optionally a
 *      `{\bf \*}' or `{\bf \&}' may be specified, signifying
 *      argument-list specification of an integer expression or
 *      address (see below).  If no stride is specified the default is
 *      one.
 *
 * \item
 *      An optional `{\bf @}' character indicating that the variable
 *      is overlaid (see below).
 *
 * \item
 *      A character specifying an internal type.
 *
 * \end{itemize}
 *
 * There is a special class of conversion specifications called vectors
 * which have a slightly different syntax (see below).
 *
 * For both the length or stride, `{\bf \*}' or `{\bf \&}' can replace
 * the hard-coded integer in the format string.  If `{\bf \*}' is used,
 * then the next argument on the argument list is used as an integer
 * expression specifying the length (or stride).  `{\bf \*}' allows the
 * size of an invoice item (or stride) to be specified at invoice
 * construction time.  Both the length or stride factor can be indirected
 * by using `{\bf \&}' instead of an integer.  The `{\bf \&}' indicates
 * that a pointer to an integer follows on the argument list, this
 * integer will be dereferenced for the length (or stride) when the
 * invoice is used.  When `{\bf \&}' is used, the length (or stride) is
 * not evaluated immediately, but is deferred until the actual packing of
 * the data occurs.  `{\bf \&}'-indirection consequently allows
 * variable-size invoices to be constructed at runtime; we call this
 * feature deferred sizing.  Note that one must be cautious of the scope
 * of C variables when using `{\bf \&}'.  For example, it is improper to
 * create an invoice in a subroutine that has a local variable as a
 * stride factor and then attempt to pass this invoice out and use it
 * elsewhere, since the stride factor points at a variable that no longer
 * is in scope.
 *
 * The built-in types are:
 *
 * \begin{tabular}{ll}
 * c & character \\
 * s & short \\
 * i & int \\
 * l & long \\
 * f & float \\
 * d & double
 * \end{tabular}
 *
 * For each conversion specification in the format string, a pointer to an
 * array of that type must be passed as an argument.  This array is where
 * {\em AMPS} will pack/unpack variables.  If the `{\bf @}' overlay
 * operator option is used then a pointer to a pointer to an array must be
 * passed.  The overlay operator is used to prevent an extra copy from
 * being made on systems with communications buffers.  The overlay operator
 * indicates that {\em AMPS} is responsible for allocating and
 * deallocating storage for that array.  On systems which use
 * communications buffers, {\em AMPS} ``overlays'' the user variable on
 * top of the communications buffer.  This enables the ``trick'' of
 * manipulating values inside of a communications buffer.  Without the
 * overlay operator, a copy is implied in all cases which would introduce
 * greater overhead.  On systems that allow gather/scatter communication
 * this notation is not needed since data can be moved from the network
 * directly into scattered user-space.  This feature should be used only
 * for temporary variables that will be discarded.  The communications
 * buffer space can not be freed until the user releases this space with
 * the \Ref{amps_Clear} function.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int i[10];
 * double d[10];
 *
 * invoice = amps_NewInvoice("%10i %5.2d", i, d);
 * \end{verbatim}
 *
 * This specifies that the user wishes to send all the elements of the
 * integer array {\bf i} and the even elements of the double array {\bf d}.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int i[10];
 * double d[10];
 * int length;
 *
 * length = 10;
 * invoice = amps_NewInvoice(&invoice, "%*i %5.2d", length, i, d);
 * \end{verbatim}
 *
 * This invoice contains the same information as the previous example but uses
 * the `{\bf *}' operator to control the length of the integer array.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int i[10];
 * double d[10];
 * int length;
 *
 * invoice = amps_NewInvoice("%&i %5.2d", &length, i, d);
 * length = 10;
 * amps_Send(amps_CommWorld, 0, invoice);
 * length = 5;
 * amps_Send(amps_CommWorld, 0, invoice);
 * \end{verbatim}
 *
 * In this example, the deferred sizing feature is demonstrated.  In the
 * first use of the invoice the entire array {\bf i} would be packed; in
 * the second use only the first five elements of {\bf i} would be
 * packed.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int *i;
 * double d[10];
 *
 * invoice = amps_NewInvoice("%10\\i %5.2d", &i, d);
 *
 * \end{verbatim}
 *
 * \begin{verbatim}
 * % \begin{figure}
 * % \epsfxsize=3.2in
 * % \epsfbox[0 0 634 431]{overlay.eps}
 * % \caption{Overlay example for an invoice ``10\i\ \%5.2d''}
 * % \label{overlay-fig}
 * % \end{figure}
 * \end{verbatim}
 *
 * The example above demonstrates the use of the overlaid feature.  In this
 * case the array {\bf i} is overlaid.  Figure 2 (???) indicates what would
 * happen if this invoice were used in a receive operation on a system that
 * used contiguous communications buffers.  The values for array {\bf d}
 * would be copied from the communications buffer.  The pointer for array
 * {\bf i} would be directed towards the communications buffer to avoid a
 * copy.
 *
 * As was indicated earlier there is a separate class of conversion
 * specifiers that allow easier communications of larger blocks of data.
 * These are called vectors in {\em AMPS}.  A vector is an n-dimensional
 * object which can have a different length and stride in each dimension.
 * A vector specifier is introduced by the `{\bf \%}' sign and is followed
 * by `{\bf \&.\&}' to indicate two arrays on the argument list.  These two
 * arrays contain the lengths and striding for each of the dimensions of
 * the vector.  Next the type of the vector is specified by one of the
 * following:
 *
 * \begin{tabular}{ll}
 * C & character vector \\
 * S & short vector \\
 * I & int vector \\
 * L & long vector \\
 * F & float vector \\
 * D & double vector
 * \end{tabular}
 *
 * The type is followed by `{\bf (dim)}' where {\bf dim} can be replaced
 * by a positive integer indicating the dimension or a `{\bf \*}' or
 * `{\bf \&}' indicating argument-list specification of an integer
 * expression or address of an integer.   The dimension indicates the
 * number of items that should be striding over for that dimension.
 *
 * For example the following invoice specification picks out the element
 * of the array marked by an `x'.
 **
 **
 **{\large Example:}
 **\begin{verbatim}
 **0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28
 **x     x     x           x     x     x           x   x    x           x     x     x
 **
 **amps_Invoice invoice;
 **int array[29];
 **int len = { 3, 4 };
 **int stride = { 2, 3 };
 **
 **invoice = amps_NewInvoice("&.&I(2)", len, stride, array);
 **
 **\end{verbatim}
 **
 **The vector type (as it's name implies) is often used to communicate
 **"parts" of an n-dimensional matrix or vector.  The vector type is useful
 **for communication of boundary data in matrices and vectors.
 **
 **@memo Create a new invoice
 **@param fmt0 The invoice format string [IN]
 **@param .... The arguments for the format string [IN]
 **@return The newly created invoice
 */
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
      return 0;                               /*error */
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
