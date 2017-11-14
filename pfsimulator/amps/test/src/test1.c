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
 * This test prints out information size and rank information on all
 * the nodes and then amps_Exits
 */

#include <stdio.h>
#include "amps.h"

int main(argc, argv)
int argc;
char *argv[];
{
  int num;
  int me;
  int i;

  int result = 0;

  char *ptr;

  /* To make sure that malloc checking is on */

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("Error amps_Init\n");
    amps_Exit(1);
  }

  ptr = malloc(20);

  num = amps_Size(amps_CommWorld);

  me = amps_Rank(amps_CommWorld);

  amps_Printf("Node %d: number procs = %d\n", me, num);

  for (i = 1; i < argc; i++)
  {
    amps_Printf("arg[%d] = %s\n", i, argv[i]);
  }

  amps_Finalize();

  return result;
}

