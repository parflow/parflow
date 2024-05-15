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

int
amps_BCast(comm, source, invoice)
amps_Comm comm;
int source;
amps_Invoice invoice;
{
  int not_found = 1;
  int *src_ptr;
  char *ptr;
  int dest;
  int size;

  AMPS_ShMemBuffer *buf;

  /* if running single node we don't need to do anything */
  if (amps_size > 1)
    if (source == amps_rank)
    {
      size = amps_pack(comm, invoice, &ptr) + sizeof(double);

      src_ptr = (int*)(ptr - sizeof(double));
      *src_ptr = amps_rank;

      /* Only allow one bcast at a time */
      WaitForSingleObject(bcast_lock, INFINITE);

      /* place the buffer on the bcast queues for all procs */
      for (dest = 0; dest < amps_size; dest++)
        if (dest != source)
        {
          /* Set up the new buffer */
          if ((buf = (AMPS_ShMemBuffer*)malloc(sizeof(AMPS_ShMemBuffer)))
              == NULL)
          {
            printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
            exit(1);
          }

          if ((buf->data = (char*)malloc(size)) == NULL)
          {
            printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
            exit(1);
          }
          memcpy(buf->data, src_ptr, size);

          buf->next = NULL;
          buf->count = 0;

          WaitForSingleObject(bcast_locks[dest], INFINITE);

          buf->prev = bcast_buffers_end[dest];

          if (bcast_buffers_end[dest])
            bcast_buffers_end[dest]->next = buf;

          bcast_buffers_end[dest] = buf;

          if (!bcast_buffers[dest])
            bcast_buffers[dest] = buf;

          /* increment semaphore so dest knows there is another message
           *  on the queue */
          ReleaseSemaphore(bcast_sema[dest], 1, NULL);

          ReleaseMutex(bcast_locks[dest]);
        }

      ReleaseMutex(bcast_lock);


      (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
      free(src_ptr);
    }
    else
    {
      /* not the source so receive */

      amps_ClearInvoice(invoice);

      /* first look on the local queue */
      buf = bcast_buffers_local[amps_rank];
      while (buf)
      {
        if (*(int*)(buf->data) == source)
        {
          /* remove the found buffer from the list */
          if (buf->prev)
            buf->prev->next = buf->next;
          else
            /* start of list */
            bcast_buffers_local[amps_rank]
              = buf->next;

          if (buf->next)
            buf->next->prev = buf->prev;
          else
            bcast_buffers_local_end[amps_rank] =
              buf->prev;

          /* return the buffer that was found */
          not_found = 0;
          break;
        }
        buf = buf->next;
      }

      while (not_found)
      {
        /* wait until someone puts another buffer on queue */
        WaitForSingleObject(bcast_sema[amps_rank], INFINITE);

        WaitForSingleObject(bcast_locks[amps_rank], INFINITE);

        buf = bcast_buffers[amps_rank];

        /* start of list */
        bcast_buffers[amps_rank] = buf->next;

        if (buf->next)
          buf->next->prev = NULL;
        else
          bcast_buffers_end[amps_rank] = NULL;

        ReleaseMutex(bcast_locks[amps_rank]);

        /* if this is the one we are looking for then we are done */
        if (*(int*)(buf->data) == source)
          not_found = 0;
        else
        {
          /* not the buffer we are looking for add to the local buffer
           * list */
          buf->next = NULL;

          buf->prev = bcast_buffers_local_end[amps_rank];

          if (bcast_buffers_local_end[amps_rank])
            bcast_buffers_end[amps_rank]->next = buf;

          bcast_buffers_local_end[amps_rank] = buf;

          if (!bcast_buffers_local[amps_rank])
            bcast_buffers_local[amps_rank] = buf;
        }
      }

      amps_unpack(comm, invoice, (buf->data) + sizeof(double));

      if ((invoice)->combuf_flags & AMPS_INVOICE_OVERLAYED)
        (invoice)->combuf_flags |= AMPS_INVOICE_ALLOCATED;
      else
      {
        (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;
#if 0
        printf("%d: freeing data\n", amps_rank);
#endif
        free(buf->data);
      }

#if 0
      printf("%d: freeing buffer\n", amps_rank);
#endif
      free(buf);
    }

  return 0;
}

