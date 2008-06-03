/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `stats.c'
 *
 * (C) 1995 Regents of the University of California.
 *
 * $Revision: 1.1.1.1 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef STATS_HEADER
#define STATS_HEADER

#include "databox.h"

#include <stdio.h>
#include <math.h>


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* stats.c */
void Stats ANSI_PROTO((Databox *databox, double *min, double *max, double *mean,
                       double *sum, double *variance, double *stdev));

#undef ANSI_PROTO

#endif
