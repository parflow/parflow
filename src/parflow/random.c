/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * SeedRand, Rand
 *
 * Routines for generating random numbers.
 *
 *****************************************************************************/

#include "amps.h"
#include <math.h>


/*--------------------------------------------------------------------------
 * Static variables
 *--------------------------------------------------------------------------*/

amps_ThreadLocalDcl(static int, Seed);

#define M 1048576
#define L 1027

/*--------------------------------------------------------------------------
 * SeedRand:
 *   The seed must always be positive.
 *
 *   Note: the internal seed must be positive and odd, so it is set
 *   to (2*input_seed - 1);
 *--------------------------------------------------------------------------*/

void  SeedRand(seed)
int   seed;
{
   amps_ThreadLocal(Seed) = (2*seed - 1) % M;
}

/*--------------------------------------------------------------------------
 * Rand
 *--------------------------------------------------------------------------*/

double  Rand()
{
   amps_ThreadLocal(Seed) = (L * amps_ThreadLocal(Seed)) % M;

   return ( ((double) amps_ThreadLocal(Seed)) / ((double) M) );
}
