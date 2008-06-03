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

#define SOURCE 0

int main (argc, argv)
int argc;
char *argv[];
{

   amps_Clock_t     time;

   double time_ticks;
 
   amps_Invoice invoice;
   amps_Invoice max_invoice;
    
   int   num;
   int   me;
   
   int loop = 100;

   int length = 100;
   int array[100];

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);

   invoice = amps_NewInvoice("%*i", length, array);


   time = amps_Clock();

   for(;loop;loop--)
       amps_BCast(amps_CommWorld, SOURCE, invoice);

   time -= amps_Clock();

   amps_FreeInvoice(invoice);


   max_invoice = amps_NewInvoice("%d", &time_ticks);

   time_ticks = -(double)(time);
   amps_AllReduce(amps_CommWorld, max_invoice, amps_Max);

   if(!me)
      amps_Printf("  wall clock time   = %lf seconds\n",
		  time_ticks/AMPS_TICKS_PER_SEC);

   amps_Finalize();

   return 0;
}

