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
 * to all the nodes
 *
 */

#include "amps.h"
#include "amps_test.h"

#include <stdio.h>
#include <string.h>

char *string = "ATestString";

int main(int argc, char *argv[])
{
  amps_Invoice invoice;
  amps_Invoice send_invoice;

  int num;
  int me;

  char *recvd_string = NULL;
  int length;

  int loop;

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
    if (me)
    {
      amps_Recv(amps_CommWorld, me - 1, invoice);
      amps_Send(amps_CommWorld, (me + 1) % num, invoice);
    }
    else
    {
      /* Put the string in the invoice */

      send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
      length = strlen(string) + 1;

      amps_Send(amps_CommWorld, 1, send_invoice);

      amps_FreeInvoice(send_invoice);

      amps_Recv(amps_CommWorld, num - 1, invoice);

      /* check the result */
      if (strcmp(recvd_string, string))
      {
        amps_Printf("ERROR!!!!! strings do not match\n");
        amps_Printf("recvd %s != %s\n", recvd_string, string);
        result = 1;
      }
    }
  }
  amps_FreeInvoice(invoice);

  amps_Finalize();

  return amps_check_result(result);
}

