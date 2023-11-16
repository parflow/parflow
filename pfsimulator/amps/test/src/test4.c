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


#include "amps.h"
#include "amps_test.h"

#include <stdio.h>
#include <string.h>

static char *string = "ATestString";

int main(int argc, char *argv[])
{
  amps_Invoice invoice;

  int me;

  int loop;
  int temp;

  int source;

  char *recvd_string = NULL;
  int length;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  loop = atoi(argv[1]);
  source = 0;

  me = amps_Rank(amps_CommWorld);

  if (me == source)
  {
    length = strlen(string) + 1;
    invoice = amps_NewInvoice("%i%i%*c", &loop, &length, length, string);
  }
  else
  {
    invoice = amps_NewInvoice("%i%i%&@c", &temp, &length, &length, &recvd_string);
  }

  for (; loop; loop--)
  {
    amps_BCast(amps_CommWorld, source, invoice);

    if (me != source)
    {
      result = strcmp(recvd_string, string);
      if (result)
      {
	result |= 1;
        amps_Printf("############## ERROR - strings don't match\n");
      }
	

      if (loop != temp)
      {
        result |= 1;
        amps_Printf("############## ERROR - ints don't match\n");
      }
    }

    amps_ClearInvoice(invoice);

    amps_Sync(amps_CommWorld);
  }

  amps_FreeInvoice(invoice);

  amps_Finalize();

  return amps_check_result(result);
}

