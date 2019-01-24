/*BHEADER*********************************************************************
 *  This file is part of Parflow. For details, see
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

#ifndef _MELISSA_HEADER
#define _MELISSA_HEADER

#include "parflow.h"

#include "simulation_snapshot.h"

/**
 * If MELISSA_ACTIVE is not 0 parflow will try to communicate with a Melissa server
 */
extern int MELISSA_ACTIVE;

/**
 * Sets MELISSA_ACTIVE depending on compile options in the input database.
 * If MELISSA_ACTIVE initializes Melissa.
 */
void NewMelissa(void);

#ifdef HAVE_MELISSA

/**
 * Tell Melissa how many bytes of which variable it will receive.
 * must be called once.
 * We transmit also saturation data. We assume that saturation and pressure data are of the
 * same shape.
 */
void MelissaInit(Vector const * const pressure, Vector const * const saturation, Vector const * const evap_trans_sum);
/**
 * Send pressure to Melissa
 */
int MelissaSend(const SimulationSnapshot * snapshot);

/**
 * Frees memory allocated by NewMelissa()
 */
void FreeMelissa(void);

#ifdef __DEBUG
#define D(x ...) printf("=======%d:", amps_Rank(amps_CommWorld)); printf(x); \
  printf("\n"); printf(" %s:%d\n", __FILE__, __LINE__)
#else
#define D(...)
#endif

#endif

#endif
