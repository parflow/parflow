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
 * to all the nodes.  The message contains every supported datatype.
 */

#include "amps.h"
#include "amps_test.h"

#include <stdio.h>
#include <string.h>

char *string = "ATestString";
amps_ThreadLocalDcl(char *, recvd_string);
amps_ThreadLocalDcl(int, string_length);
amps_ThreadLocalDcl(int, string_recvd_length);

short shorts[] = { 4, 10, 234, 5, 6 };
amps_ThreadLocalDcl(short *, recvd_shorts);
int shorts_length = 5;
amps_ThreadLocalDcl(int, shorts_recvd_length);

int ints[] = { 65555, 200, 234, 678, 890, 6789, 2789 };
amps_ThreadLocalDcl(int *, recvd_ints);
int ints_length = 7;
amps_ThreadLocalDcl(int, ints_recvd_length);

long longs[] = { 100000, 2789, 78, 8, 1, 98, 987, 98765 };
amps_ThreadLocalDcl(long *, recvd_longs);
int longs_length = 8;
amps_ThreadLocalDcl(int, longs_recvd_length);

double doubles[] = { 12.5, 10, 12.0005, 0.078, 17.4, 13.5, 679.8, 189.7 };
amps_ThreadLocalDcl(double *, recvd_doubles);
int doubles_length = 4;
amps_ThreadLocalDcl(int, doubles_recvd_length);

float floats[] = { 12.5, 10, 12.0005, 0.078, 17.4, 13.5, 679.8, 189.7, 0.01,
                   0.5 };
amps_ThreadLocalDcl(float *, recvd_floats);
int floats_length = 4;
int floats_stride = 3;
amps_ThreadLocalDcl(int, floats_recvd_length);

int test_results()
{
  int i;
  int result = 0;

  if (strcmp(amps_ThreadLocal(recvd_string), string))
  {
    amps_Printf("ERROR!!!!! chars do not match expected (%s) recvd (%s)\n", string, amps_ThreadLocal(recvd_string));
    result |= 1;
  }
  else
  {
    result |= 0;
  }

  for (i = 0; i < shorts_length; i++)
    if (shorts[i] != amps_ThreadLocal(recvd_shorts)[i])
    {
      amps_Printf("ERROR!!!!! shorts do not match expected (%hd) recvd (%hd)\n", shorts[i], amps_ThreadLocal(recvd_shorts)[i]);
      result |= 1;
    }
    else
    {
      result |= 0;
    }

  for (i = 0; i < ints_length; i++)
    if (ints[i] != amps_ThreadLocal(recvd_ints)[i])
    {
      amps_Printf("ERROR!!!!! ints do not match expected (%d) recvd (%d)\n", ints[i], amps_ThreadLocal(recvd_ints)[i]);
      result |= 1;
    }
    else
    {
      result |= 0;
    }

  for (i = 0; i < longs_length; i++)
    if (longs[i] != amps_ThreadLocal(recvd_longs)[i])
    {
      amps_Printf("ERROR!!!!! longs do not match expected (%ld) recvd (%ld)\n", longs[i], amps_ThreadLocal(recvd_longs)[i]);
      result |= 1;
    }
    else
    {
      result |= 0;
    }

  for (i = 0; i < doubles_length * 2; i += 2)
    if (doubles[i] != amps_ThreadLocal(recvd_doubles)[i])
    {
      amps_Printf("ERROR!!!!! doubles do not match expected (%lf) recvd (%lf)\n", doubles[i], amps_ThreadLocal(recvd_doubles)[i]);
      result |= 1;
    }
    else
    {
      result |= 0;
    }

  for (i = 0; i < floats_length * 3; i += 3)
    if (floats[i] != amps_ThreadLocal(recvd_floats)[i])
    {
      amps_Printf("ERROR!!!!! floats do not match expected (%f) recvd (%f)\n", floats[i], amps_ThreadLocal(recvd_floats)[i]);
      result |= 1;
    }
    else
    {
      result |= 0;
    }
  return result;
}

int main(int argc, char *argv[])
{
  amps_Invoice invoice;

  amps_Invoice send_invoice;

  int num;
  int me;

  int loop;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  loop = atoi(argv[1]);

  num = amps_Size(amps_CommWorld);

  me = amps_Rank(amps_CommWorld);


  for (; loop; loop--)
  {
    invoice = amps_NewInvoice("%i%&@c%i%&@s%i%&@i%i%&@l%i%&.*@d%i%&.*@f",
                              &amps_ThreadLocal(string_recvd_length),
                              &amps_ThreadLocal(string_recvd_length),
                              &amps_ThreadLocal(recvd_string),
                              &amps_ThreadLocal(shorts_recvd_length),
                              &amps_ThreadLocal(shorts_recvd_length),
                              &amps_ThreadLocal(recvd_shorts),
                              &amps_ThreadLocal(ints_recvd_length),
                              &amps_ThreadLocal(ints_recvd_length),
                              &amps_ThreadLocal(recvd_ints),
                              &amps_ThreadLocal(longs_recvd_length),
                              &amps_ThreadLocal(longs_recvd_length),
                              &amps_ThreadLocal(recvd_longs),
                              &amps_ThreadLocal(doubles_recvd_length),
                              &amps_ThreadLocal(doubles_recvd_length),
                              2,
                              &amps_ThreadLocal(recvd_doubles),
                              &amps_ThreadLocal(floats_recvd_length),
                              &amps_ThreadLocal(floats_recvd_length),
                              floats_stride,
                              &amps_ThreadLocal(recvd_floats));

    if (me == 0)
    {
      /* Put the string in the invoice */

      send_invoice = amps_NewInvoice("%i%&c%i%&s%i%*i%i%*l%i%*.2d%i%*.&f",
                                     &amps_ThreadLocal(string_length),
                                     &amps_ThreadLocal(string_length),
                                     string,
                                     &shorts_length,
                                     &shorts_length,
                                     shorts,
                                     &ints_length,
                                     ints_length,
                                     ints,
                                     &longs_length,
                                     longs_length,
                                     longs,
                                     &doubles_length,
                                     doubles_length,
                                     doubles,
                                     &floats_length,
                                     floats_length,
                                     &floats_stride,
                                     floats);


      amps_ThreadLocal(string_length) = strlen(string) + 1;

      amps_Send(amps_CommWorld, 1, send_invoice);
      amps_FreeInvoice(send_invoice);

      amps_Recv(amps_CommWorld, num - 1, invoice);

      result |= test_results();
    }
    else
    {
      amps_Recv(amps_CommWorld, me - 1, invoice);
      amps_Send(amps_CommWorld, (me + 1) % num, invoice);
    }

    amps_FreeInvoice(invoice);
  }

  amps_Finalize();

  return amps_check_result(result);
}

