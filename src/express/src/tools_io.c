/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "tools_io.h"

/* Code for reading/writing XDR format on non-standard systems */

/* create a dummy function call to prevent warnings when compiling 
 if non of the below apply */
void tools_io_dummy_func_call()
{   
}

#ifdef TOOLS_BYTE_SWAP	

#include <stdio.h>	

void tools_WriteDouble(file, ptr, len) 
FILE * file;
double *ptr;
int len;
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
 
void tools_WriteInt(file, ptr, len) 
FILE * file;
int *ptr;
int len;
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
 
void tools_ReadDouble(file, ptr, len) 
FILE * file;
double *ptr;
int len;
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
void tools_ReadInt(file, ptr, len) 
FILE * file;
int *ptr;
int len;
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
      fread( &a.number, sizeof(int), 1, (FILE *)file ); 

      b.buf[0] = a.buf[3];
      b.buf[1] = a.buf[2];
      b.buf[2] = a.buf[1];
      b.buf[3] = a.buf[0];
 
      *data++=b.number;
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
 
   /* write out int with bytes swaped                                       */
   for(i=len, data=ptr; i--;) 
   { 
     number = *data++;
     fwrite( &number, sizeof(short), 1, (FILE *)file ); 
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
 
   for(i=len, data=ptr; i--;) 
   { 
      fread( &number, sizeof(short), 1, (FILE *)file ); 
      *data++=number;
   } 
} 
 
#endif
