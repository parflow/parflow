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

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/param.h>

#include "amps.h"

#ifndef SEEK_SET
#define SEEK_SET 0
#endif

/*===========================================================================*/
/**
 *
 * The \Ref{amps_FFopen} command is used to open a fixed file.  This
 * file is written to and read from using the \Ref{amps_Fprintf},
 * \Ref{amps_Fscanf}, \Ref{amps_ReadType}, and
 * \Ref{amps_WriteType} commands just as you would for a normal
 * UNIX file.  You must ensure that the distributed file exists on all
 * nodes before attempting to open it for reading.  Two arguments to
 * \Ref{amps_FFopen} are the same as for {\bf fopen}: {\bf filename} is
 * the name of the file to and {\bf type} is the mode to open.
 * {\bf comm} is the communicator that specifies the nodes which are
 * opening the file, {\bf size} specifies the size (in bytes) of the
 * local contribution to the total size of the file.  {\bf size} is not
 * used when doing an open for a read (the information is in the
 * associated {\bf .dist} file).
 *
 * The macros \Ref{amps_Sizeof} are used to
 * determine the size (in bytes) of the standard C types.  You should not
 * use the Standard C {\bf sizeof} routine since {\em AMPS} writes
 * binary output using {\em XDR} format (\cite{xdr.87}).
 *
 * \Ref{amps_FFopen} returns NULL if the the file open fails.
 *
 * {\large Example:}
 * \begin{verbatim}
 * amps_File file;
 *
 * char *filename;
 * double d;
 *
 * file = amps_FFopen(amps_CommWorld, filename, "wb", amps_SizeofDouble);
 *
 * amps_WriteDouble(file, &d, 1);
 *
 * amps_FFclose(file);
 *
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * There should be some commands to take a normal file and distribute it to
 * the nodes.  This presents a problem since {\em AMPS} does not know how
 * you want to split up the file to the nodes.  This is being worked on;
 * basically the {\em AMPS} I/O calls are a hack until some standard can
 * be arrived at.  The functionality is sufficient for {\em ParFlow} at
 * the current time.
 *
 * When you are reading/writing to a file with binary data be sure and
 * use the ``b'' to signal a binary file.  This is used for portability to
 * the Win32 where binary and text files are treated differently.  Most
 * UNIX programmers do not use this feature of the ANSI C standard.
 *
 * @memo Open fixed size file
 * @param comm Communication context [IN]
 * @param filename name of the file [IN]
 * @param type fopen like type string [IN]
 * @param size size of this nodes contribution to the file [IN]
 * @return File handle or NULL
 */

amps_File amps_FFopen(amps_Comm comm, char *filename, char *type, long size)
{
  FILE *file;
  char dist_filename[MAXPATHLEN];
  long start;

#ifndef AMPS_SPLIT_FILE
  int p;
  long total;
  FILE *dfile;
#endif

  amps_Invoice invoice;

  char temp_filename[MAXPATHLEN];

  (void)comm;
  (void)size;

  sprintf(temp_filename, "%s.%05d", filename, amps_Rank(amps_CommWorld));

  invoice = amps_NewInvoice("%l", &start);

  if (!strchr(type, 'r'))
    /* open file for writing */
    if (amps_Rank(comm))
    {
#ifdef AMPS_SPLIT_FILE
      file = fopen(temp_filename, type);
#else
      start = size;
      amps_Send(comm, 0, invoice);
      amps_Recv(comm, 0, invoice);

      if (strchr(type, 'b'))
        file = fopen(filename, "r+b");
      else
        file = fopen(filename, "r+");

      fseek(file, start, SEEK_SET);
#endif
    }
    else
    {
      /* Create the dist file while gathering the size information
       * from each node */
      strcpy(dist_filename, filename);
      strcat(dist_filename, ".dist");

      unlink(filename);
      /* Node 0 always starts at byte 0 */

#ifdef AMPS_SPLIT_FILE
      file = fopen(temp_filename, type);
#else
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
#endif
    }
  else
  if (amps_Rank(comm))
  {
#ifdef AMPS_SPLIT_FILE
    file = fopen(temp_filename, type);
#else
    amps_Recv(comm, 0, invoice);
    file = fopen(filename, type);
    fseek(file, start, SEEK_SET);
#endif
  }
  else
  {
#ifdef AMPS_SPLIT_FILE
    file = fopen(temp_filename, type);
#else
    /* Open the  dist file and send the size information to each node */
    strcpy(dist_filename, filename);
    strcat(dist_filename, ".dist");

    if ((file = fopen(dist_filename, "r")) == NULL)
    {
      printf("AMPS Error: Can't open the distribution file %s for reading\n",
             dist_filename);
      AMPS_ABORT("AMPS Error");
    }

    if(fscanf(file, "%ld", &start) != 1)
    {
      printf("AMPS Error: Can't read start in file %s\n", dist_filename);
      AMPS_ABORT("AMPS Error");
    }
    
    for (p = 1; p < amps_Size(comm); p++)
    {
      if(fscanf(file, "%ld", &start) != 1)
      {
	printf("AMPS Error: Can't read start in file %s\n", dist_filename);
	AMPS_ABORT("AMPS Error");
      }
      
      amps_Send(comm, p, invoice);
    }
    fclose(file);

    file = fopen(filename, type);
    fseek(file, 0, SEEK_SET);
#endif
  }

  amps_FreeInvoice(invoice);

  return file;
}

