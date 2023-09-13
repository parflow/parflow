/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
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
 * This is a simple "ring" test.  It send a message from the host
 * to all the nodes.
 *
 */


#include "amps.h"
#include "amps_test.h"

#include <stdio.h>

#define V1_len 5
#define V2_len 11
#define V1_stride 7
#define V2_stride 3

int main(int argc, char *argv[])
{
  amps_Invoice invoice;

  int v1_len = V1_len;
  int v2_len = V2_len;
  int v1_stride = V1_stride;
  int v2_stride = V2_stride;

  int dim = 2;
  int len_array[] = { V1_len, V2_len };
  int stride_array[] = { V1_stride, V2_stride };

  int total_length;

  double *vector;

  int num;
  int me;

  int i, j;
  int c;

  int loop;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    exit(1);
  }

  loop = atoi(argv[1]);

  total_length = ((v1_len - 1) * v1_stride + 1) * v2_len + (v2_stride - 1) * (v2_len - 1);
  /* Init Vector */
  if ((vector = (double*)calloc(total_length, sizeof(double))) == NULL)
    amps_Printf("Error mallocing vector\n");

  for (; loop; loop--)
  {
    /* SGS order of args */
    invoice = amps_NewInvoice("%&.&D(*)", &len_array, &stride_array,
                              dim, vector);

    num = amps_Size(amps_CommWorld);

    me = amps_Rank(amps_CommWorld);

    if (me)
    {
      amps_Recv(amps_CommWorld, me - 1, invoice);
      amps_Send(amps_CommWorld, (me + 1) % num, invoice);
    }
    else
    {
      /* Set up the Vector */
      for (c = 0; c < total_length; c++)
        vector[c] = c;

      amps_Send(amps_CommWorld, 1, invoice);

      /* clear the array */
      /* Set up the Vector */
      for (c = 0; c < total_length; c++)
        vector[c] = 0;

      amps_Recv(amps_CommWorld, num - 1, invoice);
    }

    /* check the result */
    for (j = 0; j < v2_len; j++)
      for (i = 0; i < v1_len; i++)
        if (vector[j * ((v1_len - 1) * v1_stride + v2_stride) + i * (v1_stride)]
            != j * ((v1_len - 1) * v1_stride + v2_stride) + i * (v1_stride))
          result = 1;
        else
          vector[j * ((v1_len - 1) * v1_stride + v2_stride) + i * (v1_stride)] = 0.0;

    for (c = 0; c < total_length; c++)
      if (vector[c])
        result = 1;

    amps_FreeInvoice(invoice);
  }

  free(vector);

  amps_Finalize();

  return amps_check_result(result);
}

