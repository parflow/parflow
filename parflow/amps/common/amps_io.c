/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

void amps_ScanChar(file, data, len, stride)
amps_File file;
char *data;
int len;
int stride;
{
   char *ptr;
   char *end_ptr;

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
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

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
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

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
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

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
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

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
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

   for(end_ptr = data + len*stride, ptr = data; ptr < end_ptr; 
       ptr += stride)
      fscanf(file, "%lf ", ptr);

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
   union
   {
      double number;
      char buf[8];
   } a, b;

   /* write out each double with bytes swaped                               */
   for(i=len, data=ptr; i--;) 
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

      fwrite( &b.number, sizeof(double), 1, (FILE *)file ); 
   } 
} 

void amps_WriteInt(amps_File file, int *ptr, int len)
{ 
   int i; 
   int *data; 
   union
   {
      long number;
      char buf[4];
   } a, b;


   /* write out int with bytes swaped                                       */
   for(i=len, data=ptr; i--;) 
   { 
      a.number = *data++;
      b.buf[0] = a.buf[3];
      b.buf[1] = a.buf[2];
      b.buf[2] = a.buf[1];
      b.buf[3] = a.buf[0];
      
      fwrite( &b.number, sizeof(int), 1, (FILE *)file ); 
   } 
} 

void amps_ReadDouble(amps_File file, double *ptr, int len)
{ 
   int i; 
   double *data;
   union
   {
      double number;
      char buf[8];
   } a, b;

   /* read in each double with bytes swaped                               */
   for(i=len, data=ptr; i--;) 
   { 
      fread( &a.number, sizeof(double), 1, (FILE *)file ); 


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
   union
   {
      int number;
      char buf[4];
   } a, b;


   for(i=len, data=ptr; i--;) 
   { 


      fread( &a.number, sizeof(int), 1, (FILE *)file ); 

      b.buf[0] = a.buf[3];
      b.buf[1] = a.buf[2];
      b.buf[2] = a.buf[1];
      b.buf[3] = a.buf[0];


      *data++=b.number;
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
   for(i=len, data=ptr; i--;) 
   { 
     number = *data++;
     fwrite( &number, sizeof(short), 1, (FILE *)file ); 
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

   for(i=len, data=ptr; i--;) 
   { 
      fread( &number, sizeof(short), 1, (FILE *)file ); 
      *data++=number;
   } 
} 

#endif

