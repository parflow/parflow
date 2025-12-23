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

void amps_ClearInvoice(amps_Invoice inv)
{
  amps_InvoiceEntry *ptr;
  int stride;


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
      if (ptr->stride_type == AMPS_INVOICE_POINTER)
        stride = *(ptr->ptr_stride);
      else
        stride = ptr->stride;

      /* check if we actually created any space */
      if (ptr->data_type == AMPS_INVOICE_POINTER)
      {
        if (inv->combuf_flags & AMPS_INVOICE_NON_OVERLAYED)
          free(*((void**)(ptr->data)));
        else
        if (stride != 1)
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
