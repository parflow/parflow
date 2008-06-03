/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
double          ring_sync();

#include <stdio.h>
#include "amps.h"

int main (argc, argv)
int argc;
char *argv[];
{
   int reps, first, last, incr;
   int len;

   double t;
   double mean_time, rate;

   int left, right;
   int myrank = amps_Rank(amps_CommWorld);


   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   reps = 100;
   first = 0;
   last = 2048;
   incr = 33;

   left = myrank ? 
      myrank - 1: amps_Size(amps_CommWorld)-1;
   right = (myrank ==  amps_Size(amps_CommWorld) - 1) ?
      0: myrank + 1;
   
   if (myrank == 0)
      printf("\n#\tdist\tlen\ttime\t\tave time (us)\trate\n");

   for (len = first; len <= last; len += incr)
   {
      amps_Sync(amps_CommWorld);
      t = ring_sync(reps, len, left, right);
      mean_time = t;
      mean_time = mean_time / reps;	/* take average over trials */
      /* convert to microseconds */
      rate = (double) (len) / (mean_time * (1e-6));
      if (myrank == 0)
      {
	 printf("\t%d\t%d\t%f\t%f\t%.2f\n",
		amps_Size(amps_CommWorld), len, t, mean_time, rate);
	 fflush(stdout);
      }
   }

   amps_Finalize();
}

double          ring_sync(reps, len, left, right)
int             reps, len, left, right;
{
   amps_Invoice invoice;
   int myrank;
   amps_Clock_t start_clock;
   double          elapsed_time;
   long            i, msg_id, myproc;
   char            *sbuffer, *rbuffer;
   char            *temp_buffer;

   temp_buffer = (char *)malloc(len*sizeof(double));

   invoice = amps_NewInvoice("%*d", len, temp_buffer);
   
   myrank = amps_Rank(amps_CommWorld);

   amps_Sync(amps_CommWorld);
   start_clock = amps_Clock();

   if (myrank == 0)
      amps_Recv(amps_CommWorld, 1, invoice);
   else 
      if (myrank == 1)
	 amps_Send(amps_CommWorld, 0, invoice);


   for (i = 0; i < reps; i++)
   {
      amps_Send(amps_CommWorld, right, invoice);
      amps_Recv(amps_CommWorld, left, invoice);
   }

   elapsed_time = amps_Clock() - start_clock;

   return (elapsed_time);
}

