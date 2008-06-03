/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/* 
   This is a simple "ring" test.  It send a message from the host
   to all the nodes

*/

#include <stdio.h>
#include "amps.h"

char *string = "ATestString";

int main (argc, argv)
int argc;
char *argv[];
{
   amps_Invoice invoice;
   amps_Invoice send_invoice;
    
   int   num;
   int   me;

   int i;

   char *recvd_string = NULL;
   int length;

   int loop;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   loop=atoi(argv[1]);
   
   invoice = amps_NewInvoice("%i%&@c", &length, &length, &recvd_string);
   
   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   for(;loop;loop--)
   {
      if(me)
      {
	 amps_Recv(amps_CommWorld, me-1, invoice);
	 amps_Send(amps_CommWorld, (me+1) % num, invoice);
      }
      else
      {
	 /* Put the string in the invoice */
	 
	 send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
       length = strlen(string) + 1;
	 
	 amps_Send(amps_CommWorld, 1, send_invoice);
	 
	 amps_FreeInvoice(send_invoice);
	 
	 amps_Recv(amps_CommWorld, num-1, invoice);
	 
	 /* check the result */
	 if(strcmp(recvd_string, string))
	 {
	    amps_Printf("ERROR!!!!! strings do not match\n");
	    amps_Printf("recvd %s != %s\n", recvd_string, string);
	    result = 1;
	 }
	 else
	 {
	    amps_Printf("Success\n");
	    result = 0;
	 }
      }
   }
   amps_FreeInvoice(invoice);            
   
   amps_Finalize();
   
   return result;
}

