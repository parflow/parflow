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

void amps_ScanByte(
                   amps_File file,
                   char *    data,
                   int       len,
                   int       stride)
{
  char *ptr;
  char *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fread(ptr, 1, 1, file) != 1)
    {
      printf("AMPS Error: Can't read byte\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}


void amps_ScanChar(
                   amps_File file,
                   char *    data,
                   int       len,
                   int       stride)
{
  char *ptr;
  char *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%c", ptr) != 1)
    {
      printf("AMPS Error: Can't read char\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

void amps_ScanShort(
                    amps_File file,
                    short *   data,
                    int       len,
                    int       stride)
{
  short *ptr;
  short *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%hd ", ptr) != 1)
    {
      printf("AMPS Error: Can't read short\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

void amps_ScanInt(
                  amps_File file,
                  int *     data,
                  int       len,
                  int       stride)
{
  int *ptr;
  int *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%d ", ptr) != 1)
    {
      printf("AMPS Error: Can't read int\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

void amps_ScanLong(
                   amps_File file,
                   long *    data,
                   int       len,
                   int       stride)
{
  long *ptr;
  long *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%ld ", ptr) != 1)
    {
      printf("AMPS Error: Can't read long\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

void amps_ScanFloat(
                    amps_File file,
                    float *   data,
                    int       len,
                    int       stride)
{
  float *ptr;
  float *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%f ", ptr) != 1)
    {
      printf("AMPS Error: Can't read float\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

void amps_ScanDouble(
                     amps_File file,
                     double *  data,
                     int       len,
                     int       stride)
{
  double *ptr;
  double *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
  {
    if (fscanf(file, "%lf ", ptr) != 1)
    {
      printf("AMPS Error: Can't read double\n");
      AMPS_ABORT("AMPS Error");
    }
  }
}

#ifndef CASC_HAVE_BIGENDIAN

#include <stdio.h>

/*---------------------------------------------------------------------------*/
/* On the nCUBE2 nodes store numbers with wrong endian so we need to swap    */
/*---------------------------------------------------------------------------*/
void amps_WriteDouble(amps_File file, double *ptr, int len)
{
  int i;
  double *data;

  union {
    double number;
    char buf[8];
  } a, b;

  /* write out each double with bytes swaped                               */
  for (i = len, data = ptr; i--;)
  {
    a.number = *data++;
    b.buf[0] = a.buf[7];
    b.buf[1] = a.buf[6];
    b.buf[2] = a.buf[5];
    b.buf[3] = a.buf[4];
    b.buf[4] = a.buf[3];
    b.buf[5] = a.buf[2];
    b.buf[6] = a.buf[1];
    b.buf[7] = a.buf[0];

    fwrite(&b.number, sizeof(double), 1, (FILE*)file);
  }
}

void amps_WriteInt(amps_File file, int *ptr, int len)
{
  int i;
  int *data;

  union {
    long number;
    char buf[4];
  } a, b;


  /* write out int with bytes swaped                                       */
  for (i = len, data = ptr; i--;)
  {
    a.number = *data++;
    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];

    fwrite(&b.number, sizeof(int), 1, (FILE*)file);
  }
}

void amps_ReadDouble(amps_File file, double *ptr, int len)
{
  int i;
  double *data;

  union {
    double number;
    char buf[8];
  } a, b;

  /* read in each double with bytes swaped                               */
  for (i = len, data = ptr; i--;)
  {
    if (fread(&a.number, sizeof(double), 1, (FILE*)file) != 1)
    {
      printf("AMPS Error: Can't read double\n");
      AMPS_ABORT("AMPS Error");
    }

    b.buf[0] = a.buf[7];
    b.buf[1] = a.buf[6];
    b.buf[2] = a.buf[5];
    b.buf[3] = a.buf[4];
    b.buf[4] = a.buf[3];
    b.buf[5] = a.buf[2];
    b.buf[6] = a.buf[1];
    b.buf[7] = a.buf[0];


    *data++ = b.number;
  }
}

void amps_ReadInt(amps_File file, int *ptr, int len)
{
  int i;
  int *data;

  union {
    int number;
    char buf[4];
  } a, b;


  for (i = len, data = ptr; i--;)
  {
    if (fread(&a.number, sizeof(int), 1, (FILE*)file) != 1)
    {
      printf("AMPS Error: Can't read int\n");
      AMPS_ABORT("AMPS Error");
    }

    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];


    *data++ = b.number;
  }
}

#endif

#ifdef AMPS_INTS_ARE_64

void amps_WriteInt(file, ptr, len)
amps_File file;
int *ptr;
int len;
{
  int i;
  int *data;
  short number;

  /* write out int with bytes swaped                                       */
  for (i = len, data = ptr; i--;)
  {
    number = *data++;
    fwrite(&number, sizeof(short), 1, (FILE*)file);
  }
}

void amps_ReadInt(file, ptr, len)
amps_File file;
int *ptr;
int len;
{
  int i;
  int *data;
  short number;

  for (i = len, data = ptr; i--;)
  {
    if (fread(&number, sizeof(short), 1, (FILE*)file) != 1)
    {
      printf("AMPS Error: Can't read byte\n");
      AMPS_ABORT("AMPS Error");
    }

    *data++ = number;
  }
}

#endif

