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
/*
 * Test for ghost exchange.
 */

#include "amps.h"
#include "amps_test.h"

#include <stdio.h>

#define size 10

int main(int argc, char *argv[])
{
  amps_Package package;
  amps_Handle handle;

  int num;
  int me;

  int i;

  int loop;
  int t;

  int result = 0;

  int *recvl;
  int *recvr;
  int *send;

  double *d;

  amps_Invoice send_invoice[2];
  amps_Invoice recv_invoice[2];

  int src[2];
  int dest[2];

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  loop = atoi(argv[1]);

  num = amps_Size(amps_CommWorld);

  if (num < 2)
  {
    amps_Printf("Error: need > 1 node\n");
    exit(1);
  }

  me = amps_Rank(amps_CommWorld);

  d = amps_TAlloc(double, 8);

  recvl = amps_TAlloc(int, (me + 1) * 5);
  recvr = amps_TAlloc(int, (me + 1) * 5);
  send = amps_TAlloc(int, (me + 1) * 5);

  i = (me + 1) * 5;
  while (i--)
  {
    recvl[i] = recvr[i] = -1;
    send[i] = (i % (me + 1) ? -2 : me);
  }

  i = 8;
  while (i--)
    d[i] = (double)me;

  if (me == 0)
  {
    send_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[4],
                                      me + 1, send,
                                      &d[5]);

    recv_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[6],
                                      me + 1, recvr,
                                      &d[7]);


    src[0] = me + 1;
    dest[0] = me + 1;

    package = amps_NewPackage(amps_CommWorld,
                              1, dest, send_invoice, 1, src, recv_invoice);
  }
  else if (me == num - 1)
  {
    send_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[0],
                                      me + 1, send,
                                      &d[1]);

    recv_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[2],
                                      me + 1, recvl,
                                      &d[3]);

    src[0] = me - 1;
    dest[0] = me - 1;

    package = amps_NewPackage(amps_CommWorld,
                              1, dest, send_invoice, 1, src, recv_invoice);
  }
  else
  {
    send_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[0],
                                      me + 1, send,
                                      &d[1]);

    recv_invoice[0] = amps_NewInvoice("%d%5.*i%d",
                                      &d[2],
                                      me + 1, recvl,
                                      &d[3]);


    send_invoice[1] = amps_NewInvoice("%d%5.*i%d",
                                      &d[4],
                                      me + 1, send,
                                      &d[5]);

    recv_invoice[1] = amps_NewInvoice("%d%5.*i%d",
                                      &d[6],
                                      me + 1, recvr,
                                      &d[7]);



    src[0] = me - 1;
    dest[0] = me - 1;

    src[1] = me + 1;
    dest[1] = me + 1;

    package = amps_NewPackage(amps_CommWorld,
                              2, dest, send_invoice, 2, src, recv_invoice);
  }

  for (t = loop; t; t--)
  {
    /* Do exchange  */
    handle = amps_IExchangePackage(package);
    amps_Wait(handle);
  }


  if (me == 0)
  {
    result |= (d[4] != me);
    result |= (d[5] != me);
    result |= (d[6] != me + 1);
    result |= (d[7] != me + 1);

    i = (me + 1) * 5 - 1;
    while (i--)
    {
      result |= (send[i] != ((i % (me + 1)) ? -2 : me));
      result |= (recvr[i] != ((i % (me + 1)) ? -1 : me + 1));
    }
  }
  else if (me == num - 1)
  {
    result |= (d[0] != me);
    result |= (d[1] != me);
    result |= (d[2] != me - 1);
    result |= (d[3] != me - 1);

    i = (me + 1) * 5 - 1;
    while (i--)
    {
      result |= (send[i] != ((i % (me + 1)) ? -2 : me));
      result |= (recvl[i] != ((i % (me + 1)) ? -1 : me - 1));
    }
  }
  else
  {
    result |= (d[0] != me);
    result |= (d[1] != me);

    result |= (d[2] != me - 1);
    result |= (d[3] != me - 1);

    result |= (d[4] != me);
    result |= (d[5] != me);

    result |= (d[6] != me + 1);
    result |= (d[7] != me + 1);

    i = (me + 1) * 5 - 1;
    while (i--)
    {
      result |= (send[i] != ((i % (me + 1)) ? -2 : me));

      result |= (recvl[i] != ((i % (me + 1)) ? -1 : me - 1));

      result |= (recvr[i] != ((i % (me + 1)) ? -1 : me + 1));
    }
  }

  amps_TFree(d);
  amps_TFree(recvl);
  amps_TFree(recvr);
  amps_TFree(send);

  amps_FreePackage(package);

  if (me == 0)
  {
    amps_FreeInvoice(send_invoice[0]);
    amps_FreeInvoice(recv_invoice[0]);
  }
  else if (me == num - 1)
  {
    amps_FreeInvoice(send_invoice[0]);
    amps_FreeInvoice(recv_invoice[0]);
  }
  else
  {
    amps_FreeInvoice(send_invoice[0]);
    amps_FreeInvoice(recv_invoice[0]);
    amps_FreeInvoice(send_invoice[1]);
    amps_FreeInvoice(recv_invoice[1]);
  }

  amps_Finalize();

  return amps_check_result(result);
}


