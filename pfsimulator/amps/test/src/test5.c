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
#include "amps.h"
#include "amps_test.h"

#include <stdio.h>
#include <stdlib.h>

int sum(int x)
{
  int i, result = 0;

  for (i = 1; i <= x; i++)
    result += i;

  return result;
}

int main(int argc, char *argv[])
{
  amps_Invoice invoice;

  int test;
  double d_result;
  int i_result;

  int num;
  int me;

  int loop, i;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  loop = atoi(argv[1]);


  num = amps_Size(amps_CommWorld);

  me = amps_Rank(amps_CommWorld);

  invoice = amps_NewInvoice("%d", &d_result);

  for (i = loop; i; i--)
  {
    /* Test the Max function */

    d_result = me + 1;


    amps_AllReduce(amps_CommWorld, invoice, amps_Max);


    if ((d_result != (double)num))
    {
      amps_Printf("ERROR!!!!! MAX result is incorrect: %f  %d\n",
                  d_result, num);
      result = 1;
    }

    /* Test the Min function */

    d_result = me + 1;

    amps_AllReduce(amps_CommWorld, invoice, amps_Min);


    if ((d_result != (double)1))
    {
      amps_Printf("ERROR!!!!! MIN result is incorrect: %f  %d\n",
                  d_result, 1.0);
      result = 1;
    }

    /* Test the Add function */

    d_result = me + 1;


    amps_AllReduce(amps_CommWorld, invoice, amps_Add);


    test = sum(num);
    if ((d_result != (double)test))
    {
      amps_Printf("ERROR!!!!! Add result is incorrect: %f  %d\n",
                  d_result, test);
      result = 1;
    }
  }


  amps_FreeInvoice(invoice);

  invoice = amps_NewInvoice("%i%d", &i_result, &d_result);

  for (i = loop; i; i--)
  {
    /* Test the Max function */

    d_result = i_result = me + 1;


    amps_AllReduce(amps_CommWorld, invoice, amps_Max);


    if ((d_result != (double)num) || (i_result != num))
    {
      amps_Printf("ERROR!!!!! MAX result is incorrect: %f  %d\n",
                  d_result, i_result);
      result = 1;
    }

    /* Test the Min function */

    d_result = i_result = me + 1;

    amps_AllReduce(amps_CommWorld, invoice, amps_Min);


    if ((d_result != (double)1) || (i_result != 1))
    {
      amps_Printf("ERROR!!!!! MIN result is incorrect: %f  %d\n",
                  d_result, i_result);
      result = 1;
    }

    /* Test the Add function */

    d_result = i_result = me + 1;


    amps_AllReduce(amps_CommWorld, invoice, amps_Add);


    test = sum(num);
    if ((d_result != (double)test) || (i_result != test))
    {
      amps_Printf("ERROR!!!!! Add result is incorrect: %f  %d want %d\n",
                  d_result, i_result, test);
      result = 1;
    }
  }

  amps_FreeInvoice(invoice);

  amps_Finalize();

  return amps_check_result(result);
}

