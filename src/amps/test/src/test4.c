/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <stdio.h>
#include "amps.h"

static char *string = "ATestString";

int main (argc, argv)
int argc;
char *argv[];
{
 
   amps_Invoice invoice;
    
   int   num;
   int   me;

   int loop;
   int temp;

   int source;

   char *recvd_string = NULL;
   int length;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   loop=atoi(argv[1]);
   source=0;
   
   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   if(me == source)
   {
      length = strlen(string)+1;
      invoice = amps_NewInvoice("%i%i%*c", &loop, &length, length, string);
   }
   else
   {
      invoice = amps_NewInvoice("%i%i%&@c", &temp, &length, &length, &recvd_string);
   }

   for(;loop;loop--)
   {

      amps_BCast(amps_CommWorld, source, invoice);
      
      if( me != source )
      {
	 result = strcmp(recvd_string, string);
	 if(result)
	    amps_Printf("############## ERROR - strings don't match\n");

	 if( loop != temp)
	 {
	    result |= 1;
	    amps_Printf("############## ERROR - ints don't match\n");
	 }
      }

      amps_ClearInvoice(invoice);

      amps_Sync(amps_CommWorld);
   }

   if( me != source )
   {
      if(result == 0)
	 amps_Printf("Success\n");
   }
   
   amps_FreeInvoice(invoice);
   
   amps_Finalize();

   return result;
}

