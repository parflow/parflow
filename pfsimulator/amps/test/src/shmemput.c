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
#include <mpp/shmem.h>
#include <math.h>

#define method "Shared Memory Put"

#define min_msglen 8
#define max_msglen 3554432
#define word_size 8
#define max_words (max_msglen / word_size)

#define cycle_rate 150
#define rate_factor (1000000.0 / pow(2.0, 20.0))

long msgbuf[max_msglen];

long psync[_SHMEM_BCAST_SYNC_SIZE];

double t_total;

int main(int argc, char **argv)
{
  int nodes;
  int mypid;

  int reps, mult, count, start_msglen, stride;
  int otherpid;
  int x;

  double r_avg;

  double t_start, t_stop;
  double t_calibrate;
  double t_elapsed;
  double t_avg;

  int c;
  int msglen;
  int check;
  int r;

  double g_total;


  int i;

  nodes = _num_pes();

  mypid = _MY_PE();

  shmem_set_cache_inv();

  reps = atoi(argv[1]);
  mult = atoi(argv[2]);
  count = atoi(argv[3]);
  start_msglen = atoi(argv[4]);
  stride = atoi(argv[5]);

  for (i = 0; i < _SHMEM_BCAST_SYNC_SIZE; i++)
  {
    psync[i] = _SHMEM_SYNC_VALUE;
  }

  barrier();

  if ((mypid % 2) == 0)
  {
    otherpid = mypid + 1 % nodes;
    for (x = 0; x < max_words; x++)
      msgbuf[x] = x;
  }
  else
  {
    otherpid = (nodes + mypid - 1) % nodes;
    t_start = IRTC();
    t_stop = IRTC();
    t_calibrate = t_stop - t_start;
  }

  for (c = 1; c <= count + 1; c++)
  {
    if (c == 1)
      msglen = min_msglen;
    else if (c == 2)
      msglen = start_msglen;
    else if (mult)
      msglen = start_msglen * (int)pow((double)stride, (double)(c - 2));
    else
      msglen = start_msglen + stride * (c - 2);

    if (msglen % word_size != 0)
      msglen = (msglen / word_size) * word_size;

    if (msglen > max_msglen)
    {
      printf("Message too big\n");
      exit(1);
    }

    check = msglen / word_size + 1;

    if (mypid % 2 == 0)
    {
      for (r = 0; r < reps; r++)
      {
        barrier();
        msgbuf[check] = 1;
        shmem_put(msgbuf, msgbuf, check + 1, otherpid);
      }

      barrier();

      if (!mypid)
      {
        g_total = 0.0;
        for (x = 1; x < nodes; x += 2)
        {
          shmem_get((long*)&t_total, (long*)&t_total, 1, x);
          g_total = g_total + t_total;
        }

        t_avg = g_total / (nodes / 2.0 * reps * cycle_rate);

        if (c == 1)
        {
          printf("T3D COMMUNICATION TIMING\n");
          printf("------------------------\n");
          printf("        Method: %s\n", method);
          printf("          PE's: %d\n", nodes);
          printf("   Repetitions: %d\n", reps);
          printf("       Latency: %lf ", t_avg);
          printf("us (transmit time for %d-byte msg)\n", min_msglen);
          printf("=====================  ==============  ==============\n");
          printf("    MESSAGE LENGTH      TRANSMIT TIME     COMM RATE\n");
          printf("  (bytes)    (words)         (us)           (MB/s)\n");
          printf("========== ==========  ==============  ==============\n");
        }
        else
        {
          r_avg = (rate_factor * msglen) / t_avg;
          printf(" %7d      %6d        %7.2lf      %7.2lf\n",
                 msglen, msglen / word_size, t_avg, r_avg);
        }
      }
    }
    else
    {
      t_total = 0;
      for (r = 0; r < reps; r++)
      {
        msgbuf[check] = 0;
        barrier();

        t_start = IRTC();
        do
        {
          shmem_udcflush();
        }
        while (msgbuf[check] == 0);
        t_stop = IRTC();

        t_elapsed = t_stop - t_start - t_calibrate;
        t_total = t_total + t_elapsed;
      }
      barrier();
    }
    barrier();
  }

  barrier();
}


