/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routines for handling ParFlow timing.
 *
 *****************************************************************************/

#include <string.h>
#include "parflow.h"
#include "timing.h"


#ifdef PF_TIMING

/*--------------------------------------------------------------------------
 * NewTiming
 *--------------------------------------------------------------------------*/

void  NewTiming()
{
   timing = ctalloc(TimingType, 1);

   /* The order of these registers need to be in sync with the defines
    * found in solver.h 
    */

   RegisterTiming("Solver Setup");
   RegisterTiming("Solver");
   RegisterTiming("Solver Cleanup");
   RegisterTiming("Matvec");
   RegisterTiming("PFSB I/O");
   RegisterTiming("PFB I/O");
#ifdef VECTOR_UPDATE_TIMING
   RegisterTiming("VectorUpdate");
#endif

}


/*--------------------------------------------------------------------------
 * RegisterTiming
 *--------------------------------------------------------------------------*/

int  RegisterTiming(name)
char  *name;
{
   amps_Clock_t     *old_time     = (timing -> time);
   amps_CPUClock_t  *old_cpu_time = (timing -> cpu_time);
   FLOPType         *old_flops    = (timing -> flops);
   char            **old_name     = (timing -> name);
   int               old_size     = (timing -> size);

   int  i;


   (timing -> time)     = ctalloc(amps_Clock_t, (old_size+1));
   (timing -> cpu_time) = ctalloc(amps_CPUClock_t, (old_size+1));
   (timing -> flops)    = ctalloc(FLOPType, (old_size+1));
   (timing -> name)     = ctalloc(char *, (old_size+1));

   (timing -> size)++;

   for (i = 0; i < old_size; i++)
   {
      (timing -> time)[i]     = old_time[i];
      (timing -> cpu_time)[i] = old_cpu_time[i];
      (timing -> flops)[i]    = old_flops[i];
      (timing -> name)[i]     = old_name[i];
   }

   tfree(old_time);
   tfree(old_cpu_time);
   tfree(old_flops);
   tfree(old_name);

   (timing -> name)[old_size] = ctalloc(char, 50);
   strncpy((timing -> name)[old_size], name, 49);

   return old_size;
}


/*--------------------------------------------------------------------------
 * PrintTiming
 *--------------------------------------------------------------------------*/

void  PrintTiming()
{
   amps_File     file;
   amps_Invoice  max_invoice;

   double time_ticks;
   double cpu_ticks;
   double mflops;

   int     i;


   max_invoice = amps_NewInvoice("%d%d", &time_ticks, &cpu_ticks);

   IfLogging(0)
      file = OpenLogFile("Timing");

   for (i = 0; i < (timing -> size); i++)
   {
      time_ticks = (double)((timing -> time)[i]);
      cpu_ticks  = (double)((timing -> cpu_time)[i]);
      amps_AllReduce(amps_CommWorld, max_invoice, amps_Max);

      IfLogging(0)
      {
	 amps_Fprintf(file,"%s:\n", (timing -> name)[i]);
	 amps_Fprintf(file,"  wall clock time   = %f seconds\n",
		 time_ticks/AMPS_TICKS_PER_SEC);

	 mflops = time_ticks ? 
	    ((timing -> flops)[i]/(time_ticks/AMPS_TICKS_PER_SEC))/1.0E6
	    : 0.0;

	 amps_Fprintf(file,"  wall MFLOPS = %f (%g)\n", mflops,
		      (timing -> flops)[i]);
#ifdef CPUTiming 
	 if(AMPS_CPU_TICKS_PER_SEC)
         {
	    amps_Fprintf(file,"  CPU  clock time   = %f seconds\n",
			 cpu_ticks/AMPS_CPU_TICKS_PER_SEC);
	    if(cpu_ticks)
	       amps_Fprintf(file,"  cpu  MFLOPS = %f (%g)\n", 
			    ((timing -> flops)[i]/(cpu_ticks/AMPS_CPU_TICKS_PER_SEC))/1.0E6,
			    (timing -> flops)[i]);
	 }
#endif

      }
   }

   IfLogging(0)
      CloseLogFile(file);

#ifdef VECTOR_UPDATE_TIMING
   {
      FILE *file;
      char filename[255];
      int i;

      sprintf(filename, "event.%05d.log", amps_Rank(amps_CommWorld));
      
      file = fopen(filename, "w");
      
      for(i=0; i < NumEvents; i++)
      {
	 fprintf(file, "%d %ld %ld %ld %ld %ld %ld\n", 
		 amps_Rank(amps_CommWorld),
		 EventTiming[i][MatvecStart],
		 EventTiming[i][MatvecEnd],
		 EventTiming[i][InitStart],
		 EventTiming[i][InitEnd],
		 EventTiming[i][FinalizeStart],
		 EventTiming[i][FinalizeEnd] );
      }
      fclose(file);
   }
#endif

   amps_FreeInvoice(max_invoice);
}


/*--------------------------------------------------------------------------
 * FreeTiming
 *--------------------------------------------------------------------------*/

void  FreeTiming()
{
   int  i;


   for (i = 0; i < (timing -> size); i++)
      tfree(TimingName(i));

   tfree(timing -> time);
   tfree(timing -> cpu_time);
   tfree(timing -> flops);
   tfree(timing -> name);

   tfree(timing);
}


#endif
