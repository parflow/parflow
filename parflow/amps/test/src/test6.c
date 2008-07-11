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
   to all the nodes.
   
   Using IRecv and ISend
   
   */

#include <stdio.h>
#include "amps.h"

char *string = "ATestString";

#define TEST_COUNT 3

int main (argc, argv)
int argc;
char *argv[];
{
 
   amps_Invoice invoice;
   amps_Invoice send_invoice;
   amps_Handle handle;
    
   int   num;
   int   me;

   int loop;

   int cnt;

   char *recvd_string = NULL;
   int length;

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

      /* First test case without using amps_Test */
      if(me == 0)
      {
	 send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
	 length = strlen(string) + 1;
	 handle = amps_ISend(amps_CommWorld, 1, send_invoice);
	 sleep(1);
	 amps_Wait(handle);
	 amps_FreeInvoice(send_invoice);
      }
      else
      {
	 handle = amps_IRecv(amps_CommWorld, me-1, invoice);
	 sleep(1);
	 amps_Wait(handle);
      }
       
      if(me == num-1)
      {
	 handle = amps_ISend(amps_CommWorld, 0, invoice);
	 sleep(1);
	 amps_Wait(handle);
      }
      else
      {
	 if ( me == 0 )
	 {

	    handle = amps_IRecv(amps_CommWorld, num-1, invoice);
	    sleep(1);
	    amps_Wait(handle);
	    if(strcmp(recvd_string, string))
	    {
	       amps_Printf("ERROR!!!!! strings do not match\n");
	       result = 1;
	    }
	 }
	 else
	 {
	    handle = amps_ISend(amps_CommWorld, me+1, invoice);
	    amps_Wait(handle);
	 }
      }

      amps_Sync(amps_CommWorld);

      /* Test with a call to amps_Test */
      if(me == 0)
      {
	 send_invoice = amps_NewInvoice("%i%&c", &length, &length, string);
	 length = strlen(string) + 1;
	 handle = amps_ISend(amps_CommWorld, 1, send_invoice);
	 while(!amps_Test(handle))
	    sleep(1);
	 amps_FreeInvoice(send_invoice);
      }
      else
      {
	 handle = amps_IRecv(amps_CommWorld, me-1, invoice);
	 while(!amps_Test(handle))
	    sleep(1);
      }

      if(me == num-1)
      {
	 handle = amps_ISend(amps_CommWorld, 0, invoice);
	 cnt = TEST_COUNT;
	 while(!amps_Test(handle))
	    sleep(1);
      }
      else
      {
	 if ( me == 0 )
	 {
	    handle = amps_IRecv(amps_CommWorld, num-1, invoice);
	    while(!amps_Test(handle))
	       sleep(1);
	    if(strcmp(recvd_string, string))
	    {
	       amps_Printf("ERROR!!!!! strings do not match\n");
	       result = 1;
	    }
	 }
	 else
	 {
	    handle = amps_ISend(amps_CommWorld, me+1, invoice);
	    while(!amps_Test(handle))
	       sleep(1);
	 }
      }




      if(!me )
	 if(!result)
	    amps_Printf("Success\n");
	 else
	    amps_Printf("ERROR!!!!!!!\n");
   }

   amps_FreeInvoice(invoice);

   amps_Finalize();

   return result;
}

