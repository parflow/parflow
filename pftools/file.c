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
#include <string.h>
#include <stdlib.h>

#include "databox.h"
#include "readdatabox.h"
#include "file.h"


int FileType(
             char *filename)
{
  char *ptr;

  if ((ptr = strrchr(filename, '.')) == NULL)
    return -1;

  ptr++;

  if (strcmp(ptr, "pfb") == 0)
    return ParflowB;
  else if (strcmp(ptr, "sa") == 0)
    return SimpleA;
  else if (strcmp(ptr, "sb") == 0)
    return SimpleB;
  else
    return -1;
}

/*-----------------------------------------------------------------------
 * Read the input file
 *-----------------------------------------------------------------------*/

Databox *Read(
              int   type,
              char *filename)
{
  Databox *indatabox;
  double default_value = 0.0;

  switch (type)
  {
    case ParflowB:
      indatabox = ReadParflowB(filename, default_value);
      break;

    case SimpleA:
      indatabox = ReadSimpleA(filename, default_value);
      break;

    case SimpleB:
      indatabox = ReadSimpleB(filename, default_value);
      break;

    default:
      printf("Cannot read from that file type\n");
      exit(1);
  }

  return indatabox;
}

