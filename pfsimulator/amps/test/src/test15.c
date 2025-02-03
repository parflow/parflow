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
 * Tests for layer ghost exchange.
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

  int i, j, k;

  int loop;
  int t;

  int result = 0;

  double *a;

  amps_Invoice send_invoice[2];
  amps_Invoice recv_invoice[2];

  int length[2];
  int stride[2];

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

  length[0] = size + 2;
  length[1] = size + 2;
  stride[0] = size + 2;
  stride[1] = (size + 2);

  a = amps_CTAlloc(double, (size + 2) * (size + 2) * (size + 2));

  if (me == 0)
  {
    send_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + size);
    recv_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + size + 1);

    src[0] = me + 1;
    dest[0] = me + 1;

    package = amps_NewPackage(amps_CommWorld,
                              1, dest, send_invoice, 1, src, recv_invoice);
  }
  else if (me == num - 1)
  {
    send_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + 1);
    recv_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + 0);

    src[0] = me - 1;
    dest[0] = me - 1;

    package = amps_NewPackage(amps_CommWorld,
                              1, dest, send_invoice, 1, src, recv_invoice);
  }
  else
  {
    send_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + 1);
    recv_invoice[0] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + 0);


    send_invoice[1] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + size);
    recv_invoice[1] = amps_NewInvoice("%&.&D(2)", length, stride,
                                      a + 0 + 0 + size + 1);

    src[0] = me - 1;
    dest[0] = me - 1;

    src[1] = me + 1;
    dest[1] = me + 1;

    package = amps_NewPackage(amps_CommWorld,
                              2, dest, send_invoice, 2, src, recv_invoice);
  }

  /* Initialize all points */
  for (k = 0; k <= size + 1; k++)
    for (j = 0; j <= size + 1; j++)
      for (i = 0; i <= size + 1; i++)
        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i) = -1;

  /* Set the "interior points" */
  for (k = 0; k <= size + 1; k++)
    for (j = 0; j <= size + 1; j++)
      for (i = 1; i <= size; i++)
        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i)
          = 1000000 * k + 1000 * j + i + me * size;

  for (t = loop; t; t--)
  {
    /* Compute on the "interior points" that need to be sent */
    for (k = 0; k <= size + 1; k++)
      for (j = 0; j <= size + 1; j++)
      {
        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + 1) += 1;
        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + size) += 1;
      }

    /* Initialize exchange of boundary points */
    handle = amps_IExchangePackage(package);

    /* Compute on the "interior points" */
    for (k = 0; k <= size + 1; k++)
      for (j = 0; j <= size + 1; j++)
        for (i = 2; i <= size - 1; i++)
          *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i) += 1;



    amps_Wait(handle);
  }

  if (me == 0)
  {
    /* Node 0 does not have correct i=0 plane */
    for (k = 0; k <= size + 1; k++)
      for (j = 0; j <= size + 1; j++)
        for (i = 1; i <= size + 1; i++)
        {
          if ((int)(*(a + k * (size + 2) * (size + 2) + j * (size + 2) + i))
              != 1000000 * k + 1000 * j + i + me * size + loop)
          {
#if 0
            amps_Printf("%d: (%d, %d, %d) = %f != %d\n",
                        me, k, j, i,
                        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i),
                        1000000 * k + 1000 * j + i + me * size + loop);
#endif


            result = 1;
          }
        }
  }
  else if (me == num - 1)
  {
    /* Last Node does not have correct i=size+1 plane */
    for (k = 0; k <= size + 1; k++)
      for (j = 0; j <= size + 1; j++)
        for (i = 0; i <= size; i++)
        {
          if ((int)(*(a + k * (size + 2) * (size + 2) + j * (size + 2) + i))
              != 1000000 * k + 1000 * j + i + me * size + loop)
          {
#if 0
            amps_Printf("%d: (%d, %d, %d) = %f != %d\n",
                        me, k, j, i,
                        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i),
                        1000000 * k + 1000 * j + i + me * size + loop);
#endif
            result = 1;
          }
        }
  }
  else
  {
    for (k = 0; k <= size + 1; k++)
      for (j = 0; j <= size + 1; j++)
        for (i = 0; i <= size + 1; i++)
        {
          if ((int)(*(a + k * (size + 2) * (size + 2) + j * (size + 2) + i))
              != 1000000 * k + 1000 * j + i + me * size + loop)
          {
#if 0
            amps_Printf("%d: (%d, %d, %d) = %f != %d\n",
                        me, k, j, i,
                        *(a + k * (size + 2) * (size + 2) + j * (size + 2) + i),
                        1000000 * k + 1000 * j + i + me * size + loop);
#endif

            result = 1;
          }
        }
  }

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

  amps_TFree(a);

  amps_Finalize();

  return amps_check_result(result);
}


