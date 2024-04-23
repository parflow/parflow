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

char *amps_recvb(src)
int src;
{
  int not_found;
  char *ret;
  AMPS_ShMemBuffer *buf;

  not_found = 1;

#if 0
  printf("start of amps_recvb() %d\n", _heapchk());
#endif

  /* first look on the local queue */
  buf = buffers_local[amps_rank];
  while (buf)
  {
    if (*(int*)(buf->data) == src)
    {
      /* remove the found buffer from the list */
      if (buf->prev)
        buf->prev->next = buf->next;
      else
        /* start of list */
        buffers_local[amps_rank] = buf->next;

      if (buf->next)
        buf->next->prev = buf->prev;
      else
        buffers_local_end[amps_rank] = buf->prev;

      /* return the buffer that was found */
      not_found = 0;
      break;
    }
    buf = buf->next;
  }


  while (not_found)
  {
#if 0
    printf("%d: waiting on semaphore\n", amps_rank);
#endif
    /* try and get semaphore */
    WaitForSingleObject(sema[amps_rank], INFINITE);

#if 0
    printf("%d: got semaphore\n", amps_rank);


    printf("%d: waiting on lock\n", amps_rank);
#endif
    /* there was a waiting message */
    WaitForSingleObject(locks[amps_rank], INFINITE);

#if 0
    printf("%d: got lock\n", amps_rank);
#endif

    buf = buffers[amps_rank];

    /* remove the first buffer from the list */
    if (buf->prev)
      buf->prev->next = buf->next;
    else
      /* start of list */
      buffers[amps_rank] = buf->next;

    if (buf->next)
      buf->next->prev = buf->prev;
    else
      buffers_end[amps_rank] = buf->prev;

    ReleaseMutex(locks[amps_rank]);

    /* if this is the one we are looking for then we are done */
    if (*(int*)(buf->data) == src)
    {
      not_found = 0;
    }
    else
    {
      /* not the buffer we are looking for add to the local buffer
       * list */
      buf->prev = buffers_local_end[amps_rank];
      buf->next = NULL;

      if (buffers_local_end[amps_rank])
        buffers_local_end[amps_rank]->next = buf;

      buffers_local_end[amps_rank] = buf;

      if (!buffers_local[amps_rank])
        buffers_local[amps_rank] = buf;
    }
  }

#if 0
  printf("%d\n", _heapchk());
#endif

  ret = (buf->data) + sizeof(double);

#if 0
  printf("%d\n", _heapchk());
#endif

  free(buf);

  return ret;
}

int amps_Recv(comm, source, invoice)
amps_Comm comm;
int source;
amps_Invoice invoice;
{
  char *buffer;

  amps_ClearInvoice(invoice);

  buffer = amps_recvb(source);

  amps_unpack(comm, invoice, buffer);

  AMPS_PACK_FREE_LETTER(comm, invoice, buffer);

  return 0;
}
