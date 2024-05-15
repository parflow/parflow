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

char *filename = "test17.input";

int main(int argc, char *argv[])
{
  amps_File file;
  amps_Invoice recv_invoice;

  /* Number of times to execute SFBcast; default test is 1 */  
  int loop = 1;

  int me;

  unsigned char buffer[100];
  int length = 100;

  int result = 0;

  if (amps_Init(&argc, &argv))
  {
    amps_Printf("ERROR: Error amps_Init\n");
    amps_Exit(1);
  }

  if (argc > 2)
  {
    amps_Printf("ERROR: Invalid number of arguments\n");
    amps_Exit(1);
  }
  else if (argc == 2)
  {
    loop = atoi(argv[1]);
  }

  recv_invoice = amps_NewInvoice("%*b", length, buffer);

  me = amps_Rank(amps_CommWorld);

  if(me == 0)
  {
    FILE* test_file;

    test_file = fopen(filename, "wb");

    for(unsigned char i = 0; i < length; i++)
    {
      fwrite(&i, 1, 1, test_file);
    }

    fclose(test_file);
  }

  for (; loop; loop--)
  {
    if (!(file = amps_SFopen(filename, "rb")))
    {
      amps_Printf("Error on open\n");
      amps_Exit(1);
    }

    amps_SFBCast(amps_CommWorld, file, recv_invoice);

    for(unsigned char i = 0; i < length; i++)
    {
      if (buffer[i] != i)
      {
	amps_Printf("ERROR - byte buffers do not match\n");
	result = 1;
	break;
      }
    }
    
    amps_SFclose(file);
  }

  amps_FreeInvoice(recv_invoice);

  amps_Finalize();

  return amps_check_result(result);
}

