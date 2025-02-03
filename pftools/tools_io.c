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
/* Code for reading/writing XDR format on non-standard systems */

/* create a dummy function call to prevent warnings when compiling
 * if non of the below apply */
void tools_io_dummy_func_call()
{
}


#ifdef TOOLS_CRAY

#include <stdio.h>

int tools_length;

int tools_Integer = 1;
int tools_Real = 2;
int tools_Double = 8;
int tools_Short = 7;

static int zero = 0;
static int one = 1;

//void tools_WriteDouble(
//   FILE * file,
//   double *ptr,
//   int len)
//{
//   int i;
//   double *data;
//   double number;
//
//   /* write out each double with bytes swapped                               */
//   for(i=len, data=ptr; i--;)
//   {
//      CRAY2IEG(&tools_Double, &one, &number, &zero, data);
//      fwrite( &number, 8, 1, file );
//      data++;
//   }
//}

void tools_WriteInt(
                    FILE * file,
                    int *  ptr,
                    int    len)
{
  int i;
  int *data;
  int number;

  /* write out each double with bytes swapped                               */
  for (i = len, data = ptr; i--;)
  {
    CRAY2IEG(&tools_Integer, &one, &number, &zero, data);
    fwrite(&number, 4, 1, file);
    data++;
  }
}


void tools_ReadDouble(
                      FILE *  file,
                      double *ptr,
                      int     len)
{
  int i;
  double *data;
  double number;

  for (i = len, data = ptr; i--;)
  {
    fread(&number, 8, 1, file);
    IEG2CRAY(&tools_Double, &one, &number, &zero, data);
    data++;
  }
}


void tools_ReadInt(
                   FILE * file,
                   int *  ptr,
                   int    len)
{
  int i;
  int *data;
  int number;

  for (i = len, data = ptr; i--;)
  {
    fread(&number, 4, 1, file);
    IEG2CRAY(&tools_Integer, &one, &number, &zero, data);
    data++;
  }
}

#endif

#ifndef CASC_HAVE_BIGENDIAN

#include <stdio.h>

void tools_WriteFloat(
                      FILE * file,
                      float *ptr,
                      int    len)
{
  int i;
  float *data;

  union {
//      double number;
//      char buf[8];
    float number;
    char buf[4];
  } a, b;

  /* write out each double with bytes swapped                               */
  for (i = len, data = ptr; i--;)
  {
    a.number = *data++;
    b.buf[0] = a.buf[3];
    b.buf[1] = a.buf[2];
    b.buf[2] = a.buf[1];
    b.buf[3] = a.buf[0];

    fwrite(&b.number, sizeof(float), 1, (FILE*)file);
  }
}

void tools_WriteDouble(
                       FILE *  file,
                       double *ptr,
                       int     len)
{
  int i;
  double *data;

  union {
    double number;
    char buf[8];
  } a, b;

  /* write out each double with bytes swapped                               */
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

//       b.buf[0] = a.buf[0];
//       b.buf[1] = a.buf[1];
//       b.buf[2] = a.buf[2];
//       b.buf[3] = a.buf[3];
//       b.buf[4] = a.buf[4];
//       b.buf[5] = a.buf[5];
//       b.buf[6] = a.buf[6];
//       b.buf[7] = a.buf[7];
    fwrite(&b.number, sizeof(double), 1, (FILE*)file);
  }
}

void tools_WriteInt(
                    FILE * file,
                    int *  ptr,
                    int    len)
{
  int i;
  int *data;

  union {
    long number;
    char buf[4];
  } a, b;


  /* write out int with bytes swapped                                       */
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

void tools_ReadDouble(
                      FILE *  file,
                      double *ptr,
                      int     len)
{
  int i;
  double *data;

  union {
    double number;
    char buf[8];
  } a, b;

  /* read in each double with bytes swapped                               */
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
void tools_ReadInt(
                   FILE * file,
                   int *  ptr,
                   int    len)
{
  int i;
  int *data;

  union {
    long number;
    char buf[4];
  } a, b;


  /* write out int with bytes swapped                                       */
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

#endif

#ifdef TOOLS_INTS_ARE_64

#include <stdio.h>

void tools_WriteInt(file, ptr, len)
FILE * file;
int *ptr;
int len;
{
  int i;
  int *data;
  short number;

  /* write out int with bytes swapped                                       */
  for (i = len, data = ptr; i--;)
  {
    number = *data++;
    fwrite(&number, sizeof(short), 1, (FILE*)file);
  }
}

void tools_ReadInt(file, ptr, len)
FILE * file;
int *ptr;
int len;
{
  int i;
  int *data;
  short number;

  for (i = len, data = ptr; i--;)
  {
    fread(&number, sizeof(short), 1, (FILE*)file);
    *data++ = number;
  }
}

#endif
