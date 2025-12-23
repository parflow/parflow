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

/*===========================================================================*/
/**
 *
 * This function is used to free overlayed variables in the
 * {\bf invoice}.  \Ref{amps_Clear} is used after a receive operation
 * and when you have finished manipulating the overlayed variables that
 * were received.  After the {\bf invoice} has been cleared it is illegal
 * to access the overlayed variables.  Overlayed variables are generally
 * used for temporary values that need to be received but don't need to
 * be kept for around for an extended period of time.
 *
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int me, i;
 * double *d;
 *
 * me = amps_Rank(amps_CommWorld);
 *
 * invoice = amps_NewInvoice("%*\\d", 10, &d);
 *
 * amps_Recv(amps_CommWorld, me-1, invoice);
 *
 * for(i=0; i<10; i++)
 * {
 *      do_work(d[i]);
 * }
 *
 * amps_Clear(invoice);
 *
 * // can't access d array after clear
 *
 * amps_FreeInvoice(invoice);
 *
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * In message passing systems that use buffers, overlayed variables are
 * located in the buffer.  This eliminates a copy.
 *
 * @memo Free overlayed variables associated with an invoice.
 * @param inv Invoice to clear [IN/OUT]
 * @return none
 */
void amps_ClearInvoice(amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;

  /* if allocated then we deallocate                                       */
  if (inv->combuf_flags & AMPS_INVOICE_ALLOCATED)
    amps_free(comm, inv->combuf);

  /* set flag to unpack so free will occur if strange things happen */
  inv->flags &= ~AMPS_PACKED;

  if ((inv->combuf_flags & AMPS_INVOICE_OVERLAYED)
      || (inv->combuf_flags & AMPS_INVOICE_NON_OVERLAYED))
  {
    /* for each entry in the invoice pack null out the pointer
     * if needed  and free up space if we malloced any          */
    ptr = inv->list;
    while (ptr != NULL)
    {
      /* check if we actually created any space */
      if (ptr->data_type == AMPS_INVOICE_POINTER)
      {
        free(*((void**)(ptr->data)));
        *((void**)(ptr->data)) = NULL;
      }

      ptr = ptr->next;
    }

    inv->combuf_flags &= ~AMPS_INVOICE_OVERLAYED;
  }

  /* No longer have any malloced space associated with this invoice */
  inv->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
  inv->combuf = NULL;

  inv->combuf_flags &= ~AMPS_INVOICE_NON_OVERLAYED;

  return;
}
