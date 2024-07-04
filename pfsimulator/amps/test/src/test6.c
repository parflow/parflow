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
 * This is a simple "ring" test.  It send a message from the host
 * to all the nodes.
 *
 * Using IRecv and ISend
 *
 */


#include "amps.h"
#include "amps_test.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *string = "ATestString";

#define TEST_COUNT 3

int main(int argc, char *argv[])
{
  amps_Invoice invoice;
  amps_Invoice send_invoice;
  amps_Handle handle;

  int num;
  int me;

  int loop;

  char *recvd_string = NULL;
  int length;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  loop = atoi(argv[1]);

  invoice = amps_NewInvoice("%i%&@c", &length, &length, &recvd_string);

  num = amps_Size(amps_CommWorld);

  me = amps_Rank(amps_CommWorld);

  for (; loop; loop--)
  {
    /* First test case without using amps_Test */
    if (me == 0)
    {
      send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
      length = strlen(string) + 1;
      handle = amps_ISend(amps_CommWorld, 1, send_invoice);
      sleep(1);
      amps_Wait(handle);
      amps_FreeInvoice(send_invoice);
    }
    else
    {
      handle = amps_IRecv(amps_CommWorld, me - 1, invoice);
      sleep(1);
      amps_Wait(handle);
    }

    if (me == num - 1)
    {
      handle = amps_ISend(amps_CommWorld, 0, invoice);
      sleep(1);
      amps_Wait(handle);
    }
    else
    {
      if (me == 0)
      {
        handle = amps_IRecv(amps_CommWorld, num - 1, invoice);
        sleep(1);
        amps_Wait(handle);
        if (strcmp(recvd_string, string))
        {
          amps_Printf("ERROR!!!!! strings do not match\n");
          result = 1;
        }
      }
      else
      {
        handle = amps_ISend(amps_CommWorld, me + 1, invoice);
        amps_Wait(handle);
      }
    }

    amps_Sync(amps_CommWorld);

    /* Test with a call to amps_Test */
    if (me == 0)
    {
      send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
      length = strlen(string) + 1;
      handle = amps_ISend(amps_CommWorld, 1, send_invoice);
      while (!amps_Test(handle))
        sleep(1);
      amps_FreeInvoice(send_invoice);
    }
    else
    {
      handle = amps_IRecv(amps_CommWorld, me - 1, invoice);
      while (!amps_Test(handle))
        sleep(1);
    }

    if (me == num - 1)
    {
      handle = amps_ISend(amps_CommWorld, 0, invoice);
      while (!amps_Test(handle))
        sleep(1);
    }
    else
    {
      if (me == 0)
      {
        handle = amps_IRecv(amps_CommWorld, num - 1, invoice);
        while (!amps_Test(handle))
          sleep(1);
        if (strcmp(recvd_string, string))
        {
          amps_Printf("ERROR!!!!! strings do not match\n");
          result = 1;
        }
      }
      else
      {
        handle = amps_ISend(amps_CommWorld, me + 1, invoice);
        while (!amps_Test(handle))
          sleep(1);
      }
    }
  }

  amps_FreeInvoice(invoice);

  amps_Finalize();

  return amps_check_result(result);
}

