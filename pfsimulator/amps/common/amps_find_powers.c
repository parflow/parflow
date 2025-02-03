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

#define AMPS_HIGHEST_LONG 0x07fffffff
#define AMPS_HIGHEST_LONG_LOG 31


void amps_FindPowers(
                     int  N,
                     int *log,
                     int *Nnext,
                     int *Nprev)
{
  long high, high_log, low, low_log;

  for (high = AMPS_HIGHEST_LONG, high_log = AMPS_HIGHEST_LONG_LOG,
       low = 1, low_log = 0; (high > N) && (low < N);
       high >>= 1, low <<= 1, high_log--, low_log++)
    ;

  if (high <= N)
  {
    if (high == N)      /* no defect, power of 2! */
    {
      *Nnext = N;
      *log = (int)high_log;
    }
    else
    {
      *Nnext = (int)high << 1;
      *log = (int)high_log + 1;
    }
  }
  else   /* condition low >= N satisfied */
  {
    if (low == N)       /* no defect, power of 2! */
    {
      *Nnext = N;
      *log = (int)low_log;
    }
    else
    {
      *Nnext = (int)low;
      *log = (int)low_log;
    }
  }

  if (N == *Nnext)      /* power of 2 */
    *Nprev = N;
  else
  {
    *Nprev = *Nnext >> 1;     /* Leading bit of N is previous power of 2 */
  }
}
