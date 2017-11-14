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
/* The following needs to be in the input file test9.input */
/*
 * 11
 * ATestString
 * 4 10 234 5 6
 * 65555 200 234 678 890 6789 2789
 * 100000 2789 78 8 1 98 987 98765
 * 12.500000 12.000500 17.400000 679.800000
 * 12.500000 0.078000 679.799988 0.500000
 */

#include <stdio.h>
#include "amps.h"

char *filename = "test9.input";

int main(argc, argv)
int argc;
char *argv[];
{
  amps_File file;
  amps_Invoice recv_invoice;

  int num;
  int me;
  int i;

  char recvd_string[20];
  int string_length;

  short recvd_shorts[5];
  int shorts_length = 5;

  int recvd_ints[7];
  int ints_length = 7;

  long recvd_longs[8];
  int longs_length = 8;

  double recvd_doubles[8];
  int doubles_length = 4;

  float recvd_floats[12];
  int floats_length = 4;

  int result = 0;

  int loop;

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
    recv_invoice = amps_NewInvoice("%i%&c%*s%*i%*l%*d%*f",
                                   &string_length,
                                   &string_length, recvd_string,
                                   shorts_length, recvd_shorts,
                                   ints_length, recvd_ints,
                                   longs_length, recvd_longs,
                                   doubles_length, recvd_doubles,
                                   floats_length, recvd_floats);



    if (!(file = amps_SFopen(filename, "r")))
    {
      amps_Printf("Error on open\n");
      amps_Exit(1);
    }

    amps_SFBCast(amps_CommWorld, file, recv_invoice);

    amps_SFclose(file);

    file = amps_Fopen(filename, "w");

    amps_Fprintf(file, "%d\n", string_length);
    for (i = 0; i < string_length; i++)
      amps_Fprintf(file, "%c", recvd_string[i]);
    amps_Fprintf(file, "\n");

    for (i = 0; i < shorts_length; i++)
      amps_Fprintf(file, "%hd ", recvd_shorts[i]);
    amps_Fprintf(file, "\n");

    for (i = 0; i < ints_length; i++)
      amps_Fprintf(file, "%d ", recvd_ints[i]);
    amps_Fprintf(file, "\n");

    for (i = 0; i < longs_length; i++)
      amps_Fprintf(file, "%ld ", recvd_longs[i]);
    amps_Fprintf(file, "\n");

    for (i = 0; i < doubles_length; i++)
      amps_Fprintf(file, "%lf ", recvd_doubles[i]);
    amps_Fprintf(file, "\n");

    for (i = 0; i < floats_length; i++)
      amps_Fprintf(file, "%f ", recvd_floats[i]);
    amps_Fprintf(file, "\n");

    amps_Fclose(file);

    amps_FreeInvoice(recv_invoice);
  }


  amps_Finalize();

  return result;
}

