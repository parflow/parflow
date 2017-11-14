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
/* The following needs to be in the input file test9.input.[0-n] */
/* Normally run after test10 which will create these files */
/*
 * ATestString
 * 4 10 234 5 6
 * 65555 200 234 678 890 6789 2789
 * 100000 2789 78 8 1 98 987 98765
 * 12.5 12.0005 17.4 679.8
 * 12.5 0.078 679.8 0.5
 */

#include <stdio.h>
#include "amps.h"

char *filename = "test9.input";

int main(argc, argv)
int argc;
char *argv[];
{
  amps_File file;

  int num;
  int me;
  int i;
  char *string = "ATestString";
  char recvd_string[20];
  int string_length;

  short shorts[] = { 4, 10, 234, 5, 6 };
  short recvd_shorts[5];
  int shorts_length = 5;

  int ints[] = { 65555, 200, 234, 678, 890, 6789, 2789 };
  int recvd_ints[7];
  int ints_length = 7;

  long longs[] = { 100000, 2789, 78, 8, 1, 98, 987, 98765 };
  long recvd_longs[8];
  int longs_length = 8;

  double doubles[] = { 12.5, 12.0005, 17.4, 679.8 };
  double recvd_doubles[4];
  int doubles_length = 4;

  float floats[] = { 12.5, 0.078, 679.8, 0.5 };
  float recvd_floats[4];
  int floats_length = 4;

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
    if (!(file = amps_Fopen(filename, "r")))
    {
      amps_Printf("Error opening file\n");
      amps_Exit(1);
    }

    amps_Fscanf(file, "%d ", &string_length);
    amps_Fscanf(file, "%s", recvd_string);

    for (i = 0; i < shorts_length; i++)
      amps_Fscanf(file, "%hd ", &recvd_shorts[i]);

    for (i = 0; i < ints_length; i++)
      amps_Fscanf(file, "%d ", &recvd_ints[i]);

    for (i = 0; i < longs_length; i++)
      amps_Fscanf(file, "%ld ", &recvd_longs[i]);

    for (i = 0; i < doubles_length; i++)
      amps_Fscanf(file, "%lf ", &recvd_doubles[i]);

    for (i = 0; i < floats_length; i++)
      amps_Fscanf(file, "%f ", &recvd_floats[i]);

    if (strcmp(recvd_string, string))
    {
      amps_Printf("ERROR: chars do not match expected (%s) recvd (%s)\n",
                  string, recvd_string);
      result |= 1;
    }
    else
    {
      result |= 0;
    }

    for (i = 0; i < shorts_length; i++)
      if (shorts[i] != recvd_shorts[i])
      {
        amps_Printf("ERROR: shorts do not match expected (%hd) recvd (%hd)\n",
                    shorts[i], recvd_shorts[i]);
        result |= 1;
      }
      else
      {
        result |= 0;
      }

    for (i = 0; i < ints_length; i++)
      if (ints[i] != recvd_ints[i])
      {
        amps_Printf("ERROR: ints do not match expected (%i) recvd (%i)\n",
                    ints[i], recvd_ints[i]);
        result |= 1;
      }
      else
      {
        result |= 0;
      }

    for (i = 0; i < longs_length; i++)
      if (longs[i] != recvd_longs[i])
      {
        amps_Printf("ERROR: longs do not match expected (%ld) recvd (%ld)\n",
                    longs[i], recvd_longs[i]);
        result |= 1;
      }
      else
      {
        result |= 0;
      }

    for (i = 0; i < doubles_length; i++)
      if (doubles[i] != recvd_doubles[i])
      {
        amps_Printf("ERROR: doubles do not match (%lf) recvd (%lf)\n",
                    doubles[i], recvd_doubles[i]);
        result |= 1;
      }
      else
      {
        result |= 0;
      }

    for (i = 0; i < floats_length; i++)
      if (floats[i] != recvd_floats[i])
      {
        amps_Printf("ERROR: floats do not match expected (%f) recvd (%f)\n",
                    floats[i], recvd_floats[i]);
        result |= 1;
      }
      else
      {
        result |= 0;
      }

    amps_Fclose(file);

    if (result == 0)
      amps_Printf("Success\n");
    else
      amps_Printf("ERROR\n");
  }

  amps_Finalize();

  return result;
}


