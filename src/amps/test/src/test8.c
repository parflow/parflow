/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/* This is a test string */

#include <stdio.h>
#include "amps.h"

static char *string = "ThisIsATestString";
char *filename = "test8.input";

int main (argc, argv)
int argc;
char *argv[];
{
 
   amps_File file;
   amps_Invoice recv_invoice;

   int loop;
    
   int   num;
   int   me;

   char *recvd_string = NULL;
   int length;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   loop=atoi(argv[1]);
   
   recv_invoice = amps_NewInvoice("%i%&@c", &length, &length, &recvd_string);
   
   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   for(;loop;loop--)
   {

      if(!(file = amps_SFopen(filename, "r")))
      {
	 amps_Printf("Error on open\n");
	 amps_Exit(1);
      }
      
      amps_SFBCast(amps_CommWorld, file, recv_invoice);
      
      amps_SFclose(file);
      
      if(strncmp(recvd_string, string, 17))
      {
	 amps_Printf("############## ERROR - strings do not match\n");
	 amps_Printf("correct=<%s> returned=<%s>", string, recvd_string);
	 result = 1;
      }
      else
      {
	 amps_Printf("Success\n");
	 result = 0;
      }
      
   }

   amps_FreeInvoice(recv_invoice);

   amps_Finalize();

   return result;
}

