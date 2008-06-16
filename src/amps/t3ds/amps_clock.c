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

#ifdef AMPS_BSD_TIME



amps_Clock_t amps_Clock()   
{
   struct timeval r_time;
   amps_Clock_t micro_sec;

   /* get the current time */
   gettimeofday(&r_time, 0);

   /* get the seconds part */
   /* WARNING: This is not strictly speaking a portable operation since
      overflow may occur. */
   micro_sec = r_time.tv_sec*1000;

   /* get the m seconds part */
   micro_sec += r_time.tv_usec / 1000;

   return(micro_sec);
}

#endif


