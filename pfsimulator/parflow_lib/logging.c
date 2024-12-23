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
* NewLogging, FreeLogging, OpenLogFile, CloseLogFile
*
* Routines for printing vectors to standard out.
*
*****************************************************************************/

#include "parflow.h"
#include "pfversion.h"

/*--------------------------------------------------------------------------
 * NewLogging
 *--------------------------------------------------------------------------*/

void   NewLogging()
{
  FILE  *log_file;



  GlobalsLoggingLevel = 1;

  if (!amps_Rank(amps_CommWorld))
  {
    char filename[2048];
    sprintf(filename, "%s.log", GlobalsOutFileName);

    if ((log_file = fopen(filename, "w")) == NULL)
    {
      InputError("Error: can't open output file %s%s\n", filename, "");
    }

    fprintf(log_file, "*****************************************************************************\n");
    fprintf(log_file, "ParFlow Output Log\n");

    PrintVersionInfo(log_file);

    fprintf(log_file, "*****************************************************************************\n");

    fclose(log_file);
  }
}


/*--------------------------------------------------------------------------
 * FreeLogging
 *--------------------------------------------------------------------------*/

void   FreeLogging()
{
}


/*--------------------------------------------------------------------------
 * OpenLogFile
 *--------------------------------------------------------------------------*/

FILE *OpenLogFile(char *module_name)
{
  FILE *log_file;

  char filename[2048];

  sprintf(filename, "%s.log", GlobalsOutFileName);

  if (!amps_Rank(amps_CommWorld))
  {
    log_file = amps_SFopen(filename, "a");

    fprintf(log_file, "==============================================\n");
    fprintf(log_file, "BEGIN Log Info (%s)\n", module_name);
    fprintf(log_file, "\n");

    return log_file;
  }
  else
    return NULL;
}


/*--------------------------------------------------------------------------
 * CloseLogFile
 *--------------------------------------------------------------------------*/

int  CloseLogFile(FILE *log_file)
{
  if (!amps_Rank(amps_CommWorld))
  {
    fprintf(log_file, "\n");
    fprintf(log_file, "END Log Info\n");
    fprintf(log_file, "==============================================\n");

    return fclose(log_file);
  }
  else
    return 0;
}

/**
 * Print version information to the specified file.
 *
 *  \param log_file File to print to
 */
void PrintVersionInfo(FILE *log_file)
{
  fprintf(log_file, "\tVersion         : %s\n", PARFLOW_VERSION_STRING);
  fprintf(log_file, "\tCompiled on     : %s %s\n", __DATE__, __TIME__);

#ifdef CFLAGS
  fprintf(log_file, "\tWith C flags    : %s\n", CFLAGS);
#endif

#ifdef FFLAGS
  fprintf(log_file, "\tWith F77 flags  : %s\n", FFLAGS);
#endif

#if defined(PARFLOW_HAVE_KOKKOS) && !defined(PARFLOW_HAVE_RMM)
  fprintf(log_file, "\tWith acc backend: KOKKOS\n");
#elif defined(PARFLOW_HAVE_KOKKOS) && defined(PARFLOW_HAVE_RMM)
  fprintf(log_file, "\tWith acc backend: KOKKOS+RMM\n");
#elif defined(PARFLOW_HAVE_CUDA) && !defined(PARFLOW_HAVE_RMM)
  fprintf(log_file, "\tWith acc backend: CUDA\n");
#elif defined(PARFLOW_HAVE_CUDA) && defined(PARFLOW_HAVE_RMM)
  fprintf(log_file, "\tWith acc backend: CUDA+RMM\n");
#elif defined(PARFLOW_HAVE_OMP)
  fprintf(log_file, "\tWith acc backend: OMP\n");
#endif
}
