/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <sys/times.h>
#include <sys/time.h>
#include "amps.h"

#ifdef CASC_HAVE_GETTIMEOFDAY

amps_Clock_t amps_start_clock=0;

void amps_clock_init()
{
   struct timeval r_time;

   /* get the current time */
   gettimeofday(&r_time, 0);

   amps_start_clock = r_time.tv_sec;
}

/**

Returns the current wall clock time on the node.  This may or may
not be synchronized across the nodes.  The value returned is
in some internal units, to convert to seconds divide by
\Ref{AMPS_TICKS_PER_SEC}.

@memo Current time
@return current time in {\em AMPS} ticks
*/
amps_Clock_t amps_Clock()   
{
   struct timeval r_time;
   amps_Clock_t micro_sec;

   /* get the current time */
   gettimeofday(&r_time, 0);

   /* get the seconds part */
   micro_sec = (r_time.tv_sec - amps_start_clock);
   micro_sec = micro_sec*10000;

   /* get the lower order part */
   micro_sec += r_time.tv_usec/100;

   return(micro_sec);
}

#endif

#ifdef AMPS_NX_CLOCK
#include <nx.h>
amps_Clock_t amps_Clock()
{
	return dclock();
}
#endif


#ifndef amps_CPUClock

amps_CPUClock_t amps_CPUClock()   
{
   struct tms cpu_tms;

   times(&cpu_tms);
   
   return(cpu_tms.tms_utime);
}

#endif

