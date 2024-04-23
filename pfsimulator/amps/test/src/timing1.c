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
#include <amps.h>

#define method "Shared Memory Put"

#define min_msglen 1
#define max_msglen 1024 * 1024
#define word_size 8

int main(int argc, char **argv)
{
  amps_Invoice invoice, sum_invoice;
  amps_Package package;
  amps_Handle handle;

  double *msgbuf;

  int nodes;
  int mypid;

  int reps, count, start_msglen, stride_msglen;
  int otherpid;

  double r_avg;

  amps_Clock_t t_start, t_stop;
  amps_Clock_t t_calibrate;
  amps_Clock_t t_elapsed;
  double t_avg;
  double t_total;

  int c;
  int msglen;
  int stride;
  int r;

  /* AMPS STUFF */
  amps_Init(&argc, &argv);

  nodes = amps_Size(amps_CommWorld);

  mypid = amps_Rank(amps_CommWorld);

  reps = atoi(argv[1]);
  count = atoi(argv[2]);
  start_msglen = atoi(argv[3]) / 8;
  stride_msglen = atoi(argv[4]) / 8;

  amps_Sync(amps_CommWorld);

  msgbuf = amps_CTAlloc(double, max_msglen);

  stride = 1;
  invoice = amps_NewInvoice("%&.&D(1)", &msglen, &stride, msgbuf);

  if ((mypid % 2) == 0)
  {
    otherpid = mypid + 1 % nodes;
    package = amps_NewPackage(amps_CommWorld,
                              1, &otherpid, &invoice,
                              0, 0, NULL);
  }
  else
  {
    otherpid = (nodes + mypid - 1) % nodes;

    package = amps_NewPackage(amps_CommWorld,
                              0, 0, NULL,
                              1, &otherpid, &invoice);

    t_start = amps_Clock();
    t_stop = amps_Clock();
    t_calibrate = t_stop - t_start;
  }

  sum_invoice = amps_NewInvoice("%d", &t_total);

  for (c = 1; c <= count + 1; c++)
  {
    if (c == 1)
      msglen = min_msglen;
    else if (c == 2)
      msglen = start_msglen;
    else
      msglen = start_msglen + stride_msglen * (c - 2);

    if (msglen > max_msglen)
    {
      amps_Printf("Message too big\n");
      exit(1);
    }

    t_total = 0.0;
    if (mypid % 2 == 0)
    {
      amps_Sync(amps_CommWorld);
      for (r = 0; r < reps; r++)
      {
        handle = amps_IExchangePackage(package);
        amps_Wait(handle);
      }

      amps_Sync(amps_CommWorld);

      amps_AllReduce(amps_CommWorld, sum_invoice, amps_Add);

      if (!mypid)
      {
        t_avg = (((t_total * 2.0) / (double)(nodes * reps)) / AMPS_TICKS_PER_SEC);

        if (c == 1)
        {
          amps_Printf("T3D COMMUNICATION TIMING\n");
          amps_Printf("------------------------\n");
          amps_Printf("        Method: %s\n", method);
          amps_Printf("          PE's: %d\n", nodes);
          amps_Printf("   Repetitions: %d\n", reps);
          amps_Printf("       Latency: %lg us (transmit time for %d-double msg)\n", t_avg * 1.0E6, min_msglen);
          amps_Printf("=====================  ==============  ==============\n");
          amps_Printf("    MESSAGE LENGTH      TRANSMIT TIME     COMM RATE\n");
          amps_Printf("  (bytes)    (words)         (us)           (MB/s)\n");
          amps_Printf("========== ==========  ==============  ==============\n");
        }
        else
        {
          r_avg = ((msglen * 8) / t_avg) / (1024 * 1024);
          amps_Printf(" %7d      %6d        %lg      %7.2lf\n",
                      8 * msglen, 8 * msglen / word_size, t_avg * 1.0E6, r_avg);
        }
      }
    }
    else
    {
      amps_Sync(amps_CommWorld);
      for (r = 0; r < reps; r++)
      {
        t_start = amps_Clock();

        handle = amps_IExchangePackage(package);
        amps_Wait(handle);

        t_stop = amps_Clock();

        t_total = t_total + t_stop - t_start - t_calibrate;
      }
      amps_Sync(amps_CommWorld);
      amps_AllReduce(amps_CommWorld, sum_invoice, amps_Add);
    }

    amps_Sync(amps_CommWorld);
  }

  amps_Finalize();

  return 0;
}


