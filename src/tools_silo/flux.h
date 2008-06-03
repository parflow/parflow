/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `flux.c'
 *
 * (C) 1995 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Revision: 1.1.1.1 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef FLUX_HEADER
#define FLUX_HEADER

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

/* flux.c */
Databox *CompFlux ANSI_PROTO((Databox *k , Databox *h ));

#undef ANSI_PROTO

#endif
