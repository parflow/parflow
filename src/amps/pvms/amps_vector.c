/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

void amps_vector_out(comm, type, data, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **data;
int *len;
int *stride;
{

   int i;

   if(dim == 0)
   {
      switch(type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 pvm_pkbyte(*data, len[dim], stride[dim]);
	 *(char **)data += (len[dim]-1)*stride[dim];
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 pvm_pkshort((short*)*data, len[dim], stride[dim]);
	 *(short **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 pvm_pkint((int*)*data, len[dim], stride[dim]);
	 *(int **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 pvm_pklong((long*)*data, len[dim], stride[dim]);
	 *(long **)data += (len[dim] - 1)*stride[dim];
	 break;
       case AMPS_INVOICE_FLOAT_CTYPE:
	 pvm_pkfloat((float*)*data, len[dim], stride[dim]);
	 *(float **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 pvm_pkdouble((double*)*data, len[dim], stride[dim]);
	 *(double **)data += (len[dim] - 1)*stride[dim];
	 break;
      }
   }
   else
   {
      for(i = 0; i < (len[dim] - 1); i++)
      {
	 amps_vector_out(comm, type, data, dim-1, len, stride);

	 switch(type)
	 {
	 case AMPS_INVOICE_CHAR_CTYPE:
	    *(char **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_SHORT_CTYPE:
	    *(short **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_INT_CTYPE:
	    *(int **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_LONG_CTYPE:
	    *(long **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_FLOAT_CTYPE:
	    *(float **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_DOUBLE_CTYPE:
	    *(double **)data += stride[dim];
	    break;
	 }
      }
      amps_vector_out(comm, type, data, dim-1, len, stride);
   }
   
}

void amps_vector_in(comm, type, data, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **data;
int *len;
int *stride;
{
   int i;

   if(dim == 0)
   {
      switch(type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 pvm_upkbyte(*data, len[dim], stride[dim]);
	 *(char **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 pvm_upkshort((short*)*data, len[dim], stride[dim]);
	 *(short **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 pvm_upkint((int*)*data, len[dim], stride[dim]);
	 *(int **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 pvm_upklong((long*)*data, len[dim], stride[dim]);
	 *(long **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 pvm_upkfloat((float*)*data, len[dim], stride[dim]);
	 *(float **)data += (len[dim] - 1)*stride[dim];
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 pvm_upkdouble((double*)*data, len[dim], stride[dim]);
	 *(double **)data += (len[dim] - 1)*stride[dim];
	 break;
      }
   }
   else
   {
      for(i = 0; i < (len[dim] - 1); i++)
      {
	 amps_vector_in(comm, type, data, dim-1, len, stride);

	 switch(type)
	 {
	 case AMPS_INVOICE_CHAR_CTYPE:
	    *(char **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_SHORT_CTYPE:
	    *(short **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_INT_CTYPE:
	    *(int **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_LONG_CTYPE:
	    *(long **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_FLOAT_CTYPE:
	    *(float **)data += stride[dim];
	    break;
	 case AMPS_INVOICE_DOUBLE_CTYPE:
	    *(double **)data += stride[dim];
	    break;
	 }
      }
      amps_vector_in(comm, type, data, dim-1, len, stride);
   }
}


int amps_vector_sizeof_local(comm, type, data, dim, len, stride)
amps_Comm comm;
int type;
int dim;
char **data;
int *len;
int *stride;
{
   int size;
   int el_size;
   int i;

   switch(type)
   {
   case AMPS_INVOICE_CHAR_CTYPE:
      el_size = sizeof(char);
      break;
   case AMPS_INVOICE_SHORT_CTYPE:
      el_size = sizeof(short);
      break;
   case AMPS_INVOICE_INT_CTYPE:
      el_size = sizeof(int);
      break;
   case AMPS_INVOICE_LONG_CTYPE:
      el_size = sizeof(long);
      break;
   case AMPS_INVOICE_FLOAT_CTYPE:
      el_size = sizeof(float);
      break;
   case AMPS_INVOICE_DOUBLE_CTYPE:
      el_size = sizeof(double);
      break;
   }

   size = el_size*(len[0]-1)*stride[0];

   for(i = 1; i < dim ; i++)
   {
      size = size*len[i] + stride[i]*(len[i] - 1)*el_size;
   }

   return size;
}



