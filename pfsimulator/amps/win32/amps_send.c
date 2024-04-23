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

int amps_xsend(buffer, dest)
char *buffer;
int dest;
{
  int *src_ptr;
  AMPS_ShMemBuffer *buf;

  src_ptr = (int*)(buffer - sizeof(double));
  *src_ptr = amps_rank;

#if 0
  printf("%d AMPS sending to %d\n", amps_rank, dest);
#endif

  /* Set up the new buffer */
  if ((buf = (AMPS_ShMemBuffer*)malloc(sizeof(AMPS_ShMemBuffer)))
      == NULL)
  {
    printf("AMPS Error: AMPS_ShMemBuffer allocation failed\n");
    exit(1);
  }

  buf->data = (char*)src_ptr;
  buf->next = NULL;

  /* place the buffer on the message queue */

  WaitForSingleObject(locks[dest], INFINITE);

  buf->prev = buffers_end[dest];

  if (buffers_end[dest])
    buffers_end[dest]->next = buf;

  buffers_end[dest] = buf;

  if (!buffers[dest])
    buffers[dest] = buf;


  /* increment semaphore so dest knows there is another message
   * on the queue */

#if 0
  printf("%d AMPS semaphoring %d\n", amps_rank, dest);
#endif
  ReleaseSemaphore(sema[dest], 1, NULL);

  ReleaseMutex(locks[dest]);


  return 0;
}

int amps_Send(comm, dest, invoice)
amps_Comm comm;
int dest;
amps_Invoice invoice;
{
  char *buffer;


  amps_pack(comm, invoice, &buffer);

  amps_xsend(buffer, dest);

  AMPS_CLEAR_INVOICE(invoice);

  return 0;
}
