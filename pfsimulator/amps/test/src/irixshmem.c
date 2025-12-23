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

#define method "Shared Memory Memcpy"

#define min_msglen 1
#define max_msglen 1024 * 1024
#define word_size 8

char msgbuf[max_msglen][4];

int main(int argc, char **argv)
{
  amps_Invoice invoice, sum_invoice;
  amps_Package package;
  amps_Handle handle;

  int nodes;
  int mypid;

  int x;

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

  int check;

  /* AMPS STUFF */
  amps_Init(&argc, &argv);

  nodes = amps_Size(amps_CommWorld);

  mypid = amps_Rank(amps_CommWorld);

  reps = atoi(argv[1]);
  count = atoi(argv[2]);
  start_msglen = atoi(argv[3]);
  stride_msglen = atoi(argv[4]);

  amps_Sync(amps_CommWorld);

  if ((mypid % 2) == 0)
  {
    otherpid = mypid + 1 % nodes;
    for (x = 0; x < max_msglen; x++)
      msgbuf[x][mypid] = x;
  }
  else
  {
    otherpid = (nodes + mypid - 1) % nodes;

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

    check = msglen + 1;

    t_total = 0.0;
    if (mypid % 2 == 0)
    {
      for (r = 0; r < reps; r++)
      {
        amps_Sync(amps_CommWorld);
        memcpy(&(msgbuf[0][otherpid]), &(msgbuf[0][mypid]), check);
        amps_Sync(amps_CommWorld);
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
          r_avg = ((msglen) / t_avg) / (1024 * 1024);
          amps_Printf(" %7d      %6d        %lg      %7.2lf\n",
                      msglen, msglen / word_size, t_avg * 1.0E6, r_avg);
        }
      }
    }
    else
    {
      t_start = amps_Clock();
      for (r = 0; r < reps; r++)
      {
        amps_Sync(amps_CommWorld);
        amps_Sync(amps_CommWorld);
      }
      t_stop = amps_Clock();
      t_total = t_total + t_stop - t_start - t_calibrate;
      amps_Sync(amps_CommWorld);
      amps_AllReduce(amps_CommWorld, sum_invoice, amps_Add);
    }

    amps_Sync(amps_CommWorld);
  }

  amps_Finalize();

  return 0;
}


