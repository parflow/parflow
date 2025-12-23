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

#include <stdio.h>
#include "amps.h"

#define SOURCE 0

int main(argc, argv)
int argc;
char *argv[];
{
  amps_Clock_t time;

  double time_ticks;

  amps_Invoice invoice;
  amps_Invoice max_invoice;

  int num;
  int me;

  int loop = 100;

  int length = 100;
  int array[100];

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  num = amps_Size(amps_CommWorld);

  me = amps_Rank(amps_CommWorld);

  invoice = amps_NewInvoice("%*i", length, array);


  time = amps_Clock();

  for (; loop; loop--)
    amps_BCast(amps_CommWorld, SOURCE, invoice);

  time -= amps_Clock();

  amps_FreeInvoice(invoice);


  max_invoice = amps_NewInvoice("%d", &time_ticks);

  time_ticks = -(double)(time);
  amps_AllReduce(amps_CommWorld, max_invoice, amps_Max);

  if (!me)
    amps_Printf("  wall clock time   = %lf seconds\n",
                time_ticks / AMPS_TICKS_PER_SEC);

  amps_Finalize();

  return 0;
}

