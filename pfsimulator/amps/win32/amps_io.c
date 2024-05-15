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

void amps_ScanChar(file, data, len, stride)
amps_File file;
char *data;
int len;
int stride;
{
  char *ptr;
  char *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%c", ptr);
}

void amps_ScanShort(file, data, len, stride)
amps_File file;
short *data;
int len;
int stride;
{
  short *ptr;
  short *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%hd ", ptr);
}

void amps_ScanInt(file, data, len, stride)
amps_File file;
int *data;
int len;
int stride;
{
  int *ptr;
  int *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%d ", ptr);
}

void amps_ScanLong(file, data, len, stride)
amps_File file;
long *data;
int len;
int stride;
{
  long *ptr;
  long *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%ld ", ptr);
}

void amps_ScanFloat(file, data, len, stride)
amps_File file;
float *data;
int len;
int stride;
{
  float *ptr;
  float *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%f ", ptr);
}

void amps_ScanDouble(file, data, len, stride)
amps_File file;
double *data;
int len;
int stride;
{
  double *ptr;
  double *end_ptr;

  for (end_ptr = data + len * stride, ptr = data; ptr < end_ptr;
       ptr += stride)
    fscanf(file, "%lf ", ptr);
}

/*---------------------------------------------------------------------------*/
/* On the NT nodes store numbers with wrong endian so we need to swap    */
/*---------------------------------------------------------------------------*/
void amps_WriteDouble(file, ptr, len)
amps_File file;
double *ptr;
int len;
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

void amps_WriteInt(file, ptr, len)
amps_File file;
int *ptr;
int len;
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

void amps_ReadDouble(file, ptr, len)
amps_File file;
double *ptr;
int len;
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
    fread(&a.number, sizeof(double), 1, (FILE*)file);

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

void amps_ReadInt(file, ptr, len)
amps_File file;
int *ptr;
int len;
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
    fread(&a.number, sizeof(int), 1, (FILE*)file);
    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];

    *data++ = b.number;
  }
}


