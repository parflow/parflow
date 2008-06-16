/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

/* returns local size we don't know about remote stuff */
int amps_sizeof_invoice(comm, inv)
amps_Comm comm;
amps_Invoice inv;
{
   amps_InvoiceEntry *ptr;
   char *cur_pos = 0;
   int len, stride=1;

   ptr = inv -> list;

   while(ptr != NULL)
   {
      if(ptr -> len_type == AMPS_INVOICE_POINTER)
	 len = *(ptr -> ptr_len);
      else
	 len = ptr ->len;

      switch(ptr->type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 cur_pos += AMPS_ALIGN(char, cur_pos);
	 cur_pos += AMPS_SIZEOF(char, len, stride);
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 cur_pos += AMPS_ALIGN(short, cur_pos);
	 cur_pos += AMPS_SIZEOF(short, len, stride);
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 cur_pos += AMPS_ALIGN(int, cur_pos);
	 cur_pos += AMPS_SIZEOF(int, len, stride);
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 cur_pos += AMPS_ALIGN(long, cur_pos);
	 cur_pos += AMPS_SIZEOF(long, len, stride);
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 cur_pos += AMPS_ALIGN(float, cur_pos);
	 cur_pos += AMPS_SIZEOF(float, len, stride);
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 cur_pos += AMPS_ALIGN(double, cur_pos);
	 cur_pos += AMPS_SIZEOF(double, len, stride);
	 break;
      }
      ptr = ptr->next;
   }
   return (int) cur_pos;
}



