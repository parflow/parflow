/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include "amps.h"

#define AMPS_HIGHEST_LONG 0x07fffffff
#define AMPS_HIGHEST_LONG_LOG 31


void amps_FindPowers(N, log, Nnext, Nprev)
int N;
int *log, *Nnext, *Nprev;
{
    long high, high_log, low, low_log;

    for(high = AMPS_HIGHEST_LONG, high_log = AMPS_HIGHEST_LONG_LOG,
        low  = 1,            low_log  = 0 ; (high > N) && (low < N);
        high >>= 1, low <<= 1, high_log--, low_log++)
                ;

    if(high <= N)
    {
        if(high == N)   /* no defect, power of 2! */
        {
            *Nnext = N;
            *log   = high_log;
	 }
        else
        {
            *Nnext = high << 1;
            *log   = high_log + 1;
	 }
     }
    else /* condition low >= N satisfied */
    {
        if(low == N)    /* no defect, power of 2! */
        {
            *Nnext = N;
            *log   = low_log;
	 }
        else
        {
            *Nnext = low;
            *log   = low_log;
	 }
     }

    if(N == *Nnext)     /* power of 2 */
        *Nprev = N;
    else
    {
        *Nprev = *Nnext >> 1; /* Leading bit of N is previous power of 2 */
     }
 }
