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
#include <stdio.h>

#include "amps.h"

#ifndef SEEK_SET
#define SEEK_SET 0
#endif

amps_File amps_FFopen(comm, filename, type, size)
amps_Comm comm;
char *filename;
char *type;
long size;
{
  FILE *file, *dfile;
  char dist_filename[MAXPATHLEN];
  int p;
  long start;
  long total;
  amps_Invoice invoice;

  invoice = amps_NewInvoice("%l", &start);

  if (!strchr(type, 'r'))
    /* open file for writing */
    if (amps_Rank(comm))
    {
      start = size;
      amps_Send(comm, 0, invoice);
      amps_Recv(comm, 0, invoice);

      if (strchr(type, 'b'))
        file = fopen(filename, "r+b");
      else
        file = fopen(filename, "r+");
      fseek(file, start, SEEK_SET);
    }
    else
    {
      /* Create the dist file while gathering the size information
       * from each node */
      strcpy(dist_filename, filename);
      strcat(dist_filename, ".dist");

      unlink(filename);
      /* Node 0 always starts at byte 0 */
      file = fopen(filename, type);
      fseek(file, 0L, SEEK_SET);

      if ((dfile = fopen(dist_filename, "w")) == NULL)
      {
        printf("AMPS Error: Can't open the distribution file %s\n",
               dist_filename);
        exit(1);
      }

      total = start = size;
      fprintf(dfile, "0\n");

      for (p = 1; p < amps_Size(comm); p++)
      {
        amps_Recv(comm, p, invoice);
        size = start;
        start = total;
        fprintf(dfile, "%ld\n", start);
        amps_Send(comm, p, invoice);
        total += size;
      }
      fclose(dfile);
    }
  else
  if (amps_Rank(comm))
  {
    amps_Recv(comm, 0, invoice);
    file = fopen(filename, type);
    fseek(file, start, SEEK_SET);
  }
  else
  {
    /* Open the  dist file and send the size information to each node */
    strcpy(dist_filename, filename);
    strcat(dist_filename, ".dist");

    if ((file = fopen(dist_filename, "r")) == NULL)
    {
      printf("AMPS Error: Can't open the distribution file %s for reading\n",
             dist_filename);
      exit(1);
    }

    fscanf(file, "%ld", &start);
    for (p = 1; p < amps_Size(comm); p++)
    {
      fscanf(file, "%ld", &start);
      amps_Send(comm, p, invoice);
    }
    fclose(file);

    file = fopen(filename, type);
    fseek(file, 0, SEEK_SET);
  }

  amps_FreeInvoice(invoice);

  return file;
}

