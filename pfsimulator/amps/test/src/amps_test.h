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
#ifndef AMPS_TEST_H
#define AMPS_TEST_H

int amps_check_result(int result)
{
  if (result)
  {
    printf("FAILED\n");
  }
  else
  {
    int me = amps_Rank(amps_CommWorld);

    if (me == 0)
    {
      printf("PASSED\n");
    }
  }

  return result;
}

int amps_compare_files(char *filename1, char *filename2)
{
  FILE *file1 = fopen(filename1, "r");
  FILE *file2 = fopen(filename2, "r");

  char ch1;
  char ch2;

  do
  {
    ch1 = fgetc(file1);
    ch2 = fgetc(file2);

    if (ch1 != ch2)
    {
      return 1;
    }
  }
  while (ch1 != EOF && ch2 != EOF);

  fclose(file1);
  fclose(file2);

  if (ch1 == EOF && ch2 == EOF)
  {
    return 0;
  }

  return 1;
}

#endif /* amps_test */

