/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#include "amps.h"

#include <sys/times.h>
#include <sys/time.h>
#include <unistd.h>

long AMPS_CPU_TICKS_PER_SEC;

#ifdef CASC_HAVE_GETTIMEOFDAY

amps_Clock_t amps_start_clock = 0;

void amps_clock_init()
{
  struct timeval r_time;

  /* get the current time */
  gettimeofday(&r_time, 0);

  amps_start_clock = r_time.tv_sec;

  AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
}

/**
 *
 * Returns the current wall clock time on the node.  This may or may
 * not be synchronized across the nodes.  The value returned is
 * in some internal units, to convert to seconds divide by
 * \Ref{AMPS_TICKS_PER_SEC}.
 *
 * @memo Current time
 * @return current time in {\em AMPS} ticks
 */
amps_Clock_t amps_Clock()
{
  struct timeval r_time;
  amps_Clock_t micro_sec;

  /* get the current time */
  gettimeofday(&r_time, 0);

  /* get the seconds part */
  micro_sec = (r_time.tv_sec - amps_start_clock);
  micro_sec = micro_sec * 10000;

  /* get the lower order part */
  micro_sec += r_time.tv_usec / 100;

  return(micro_sec);
}

#else

void amps_clock_init()
{
  AMPS_CPU_TICKS_PER_SEC = sysconf(_SC_CLK_TCK);
}

amps_CPUClock_t amps_Clock()
{
  struct tms cpu_tms;

  times(&cpu_tms);

  return(cpu_tms.tms_utime);
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

