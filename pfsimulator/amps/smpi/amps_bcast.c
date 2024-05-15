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

#define vtor(node) \
  (node > source ? node : (node) ? node - 1 : source)

#define rtov(node) \
  (node > source ? node : (node < source) ? node + 1 : 0)

/*===========================================================================*/
/**
 *
 * The collective operation \Ref{amps_BCast} is used to communicate
 * information from one node to all the other nodes who are members
 * of a context.  {\bf rank} is the rank of the node that contains
 * the information to be communicated.  {\bf comm} is the communication
 * context (group of nodes) that should receive the information.
 *
 * In order to allow more symmetry in the code on the source and destination
 * nodes, overlay-ed variables are valid after an \Ref{amps_BCast} until a
 * \Ref{amps_Clear} is invoked on both the source and destination
 * nodes.  This differs from some systems were the buffer is considered
 * destroyed on the source node once the broadcast is invoked.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_Invoice invoice;
 * int length, source_rank;
 * char *string;
 *
 * me = amps_Rank(amps_CommWorld);
 *
 * if(me == source_rank)
 * {
 *      // source knows the length
 *      length = strlen(string)+1;
 *      invoice = amps_NewInvoice("%i%*c", &length, length, string);
 * }
 * else
 * {
 *      // receiving nodes do not know length so user overlayed variable
 *      invoice = amps_NewInvoice("%i%&\%", &length, &length, &recvd_string);
 * }
 *
 * // broadcast a character array and it's length
 * amps_BCast(amps_CommWorld, source_rank, invoice);
 *
 * amps_FreeInvoice(invoice);
 * \end{verbatim}
 *
 * @memo Send to all nodes
 * @param comm communication context for the broadcast [IN]
 * @param source rank of the node that is sending [IN]
 * @param invoice the data to send [IN/OUT]
 * @return Error code
 */
int amps_BCast(amps_Comm comm, int source, amps_Invoice invoice)
{
  int n;
  int N;
  int d;
  int poft, log, npoft;
  int start;
  int startd;
  int node;

  int size;
  int packed;
  char *buffer;

  int recvd_flag;

  N = amps_size;
  n = rtov(amps_rank);

  amps_FindPowers(N, &log, &npoft, &poft);

  packed = 0;
  start = 0;        /* start of sub "power of two" block we are working on */
  recvd_flag = (n == start);

  if (n >= poft)
  {
    node = vtor(n - poft);
    amps_Recv(comm, node, invoice);
  }
  else
  {
    for (d = poft >> 1; d >= 1; d >>= 1)
    {
      startd = start ^ d;

      if (!recvd_flag && (startd != n))
      {
        if (n >= startd)
          start = startd;
        continue;
      }

      if (recvd_flag)
      {
        node = vtor((n ^ d));
        size = amps_pack(comm, invoice, &buffer);
        amps_xsend(comm, node, buffer, size);
        packed = 1;
      }
      else
      {
        node = vtor(start);
        recvd_flag = 1;
        amps_Recv(comm, node, invoice);
      }
    }

    if (n < N - poft)
    {
      if (!packed)
        size = amps_pack(comm, invoice, &buffer);

      node = vtor(poft + n);
      amps_xsend(comm, node, buffer, size);
    }

    /*amps_unpack (comm, invoice, buffer);
     * AMPS_PACK_FREE_LETTER(comm, invoice, buffer);
     */
  }
  return 0;
}
