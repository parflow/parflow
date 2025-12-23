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
/*****************************************************************************
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
   * found in timing.h
   */

  RegisterTiming("Solver Setup");
  RegisterTiming("Solver");
  RegisterTiming("Richards Exclude 1st Time Step");
  RegisterTiming("Solver Cleanup");
  RegisterTiming("Matvec");
  RegisterTiming("PFSB I/O");
  RegisterTiming("PFB I/O");
  RegisterTiming("CLM");
  RegisterTiming("PFSOL Read");
  RegisterTiming("Clustering");
  RegisterTiming("Netcdf I/O");
  RegisterTiming("PDI I/O");
#ifdef VECTOR_UPDATE_TIMING
  RegisterTiming("VectorUpdate");
#endif
}


/*--------------------------------------------------------------------------
 * RegisterTiming
 *--------------------------------------------------------------------------*/

int  RegisterTiming(
                    char *name)
{
  amps_Clock_t     *old_time = (timing->time);
  amps_CPUClock_t  *old_cpu_time = (timing->cpu_time);
  FLOPType         *old_flops = (timing->flops);
  char            **old_name = (timing->name);
  int old_size = (timing->size);

  int i;


  (timing->time) = ctalloc(amps_Clock_t, (old_size + 1));
  (timing->cpu_time) = ctalloc(amps_CPUClock_t, (old_size + 1));
  (timing->flops) = ctalloc(FLOPType, (old_size + 1));
  (timing->name) = ctalloc(char *, (old_size + 1));

  (timing->size)++;

  for (i = 0; i < old_size; i++)
  {
    (timing->time)[i] = old_time[i];
    (timing->cpu_time)[i] = old_cpu_time[i];
    (timing->flops)[i] = old_flops[i];
    (timing->name)[i] = old_name[i];
  }

  tfree(old_time);
  tfree(old_cpu_time);
  tfree(old_flops);
  tfree(old_name);

  (timing->name)[old_size] = ctalloc(char, 50);
  strncpy((timing->name)[old_size], name, 49);

  return old_size;
}


/*--------------------------------------------------------------------------
 * PrintTiming
 *--------------------------------------------------------------------------*/

void  PrintTiming()
{
  amps_File file = NULL;
  amps_Invoice max_invoice;

  double time_ticks[timing->size];
  double cpu_ticks[timing->size];
  double mflops[timing->size];

  int i;

  max_invoice = amps_NewInvoice("%*d%*d", timing->size, &time_ticks, timing->size, &cpu_ticks);

  for (i = 0; i < (timing->size); i++)
  {
    time_ticks[i] = (double)((timing->time)[i]);
    cpu_ticks[i] = (double)((timing->cpu_time)[i]);
  }

  amps_AllReduce(amps_CommWorld, max_invoice, amps_Max);

  for (i = 0; i < (timing->size); i++)
  {
    mflops[i] = time_ticks[i] ?
                ((timing->flops)[i] / (time_ticks[i] / AMPS_TICKS_PER_SEC)) / 1.0E6
                : 0.0;
  }

  IfLogging(0)
  {
    file = OpenLogFile("Timing");

    for (i = 0; i < (timing->size); i++)
    {
      amps_Fprintf(file, "%s:\n", (timing->name)[i]);
      amps_Fprintf(file, "  wall clock time   = %f seconds\n",
                   time_ticks[i] / AMPS_TICKS_PER_SEC);
      amps_Fprintf(file, "  wall MFLOPS = %f (%g)\n", mflops[i],
                   (timing->flops)[i]);
#ifdef CPUTiming
      if (AMPS_CPU_TICKS_PER_SEC)
      {
        amps_Fprintf(file, "  CPU  clock time   = %f seconds\n",
                     cpu_ticks[i] / AMPS_CPU_TICKS_PER_SEC);
        if (cpu_ticks[i])
          amps_Fprintf(file, "  cpu  MFLOPS = %f (%g)\n",
                       ((timing->flops)[i] / (cpu_ticks[i] / AMPS_CPU_TICKS_PER_SEC)) / 1.0E6,
                       (timing->flops)[i]);
      }
#endif
    }

    CloseLogFile(file);

    char filename[2048];
    sprintf(filename, "%s.timing.csv", GlobalsOutFileName);

    if ((file = fopen(filename, "w")) == NULL)
    {
      InputError("Error: can't open output file %s%s\n", filename, "");
    }

    fprintf(file, "Timer,Time (s),MFLOPS (mops/s),FLOP (op)\n");
    for (i = 0; i < (timing->size); i++)
    {
      fprintf(file, "%s,%f,%f,%g\n", timing->name[i],
              time_ticks[i] / AMPS_TICKS_PER_SEC,
              mflops[i], (timing->flops)[i]);
    }

    fclose(file);
  }

#ifdef VECTOR_UPDATE_TIMING
  {
    FILE *file;
    char filename[255];
    int i;

    sprintf(filename, "event.%05d.log", amps_Rank(amps_CommWorld));

    file = fopen(filename, "w");

    for (i = 0; i < NumEvents; i++)
    {
      fprintf(file, "%d %ld %ld %ld %ld %ld %ld\n",
              amps_Rank(amps_CommWorld),
              EventTiming[i][MatvecStart],
              EventTiming[i][MatvecEnd],
              EventTiming[i][InitStart],
              EventTiming[i][InitEnd],
              EventTiming[i][FinalizeStart],
              EventTiming[i][FinalizeEnd]);
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
  int i;


  for (i = 0; i < (timing->size); i++)
    tfree(TimingName(i));

  tfree(timing->time);
  tfree(timing->cpu_time);
  tfree(timing->flops);
  tfree(timing->name);

  tfree(timing);
}


#endif
