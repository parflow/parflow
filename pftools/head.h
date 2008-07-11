/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `head.c'
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: head.h,v $
 Revision 1.4  1996/08/10 06:35:02  mccombjr
 *** empty log message ***

 * Revision 1.3  1996/04/25  01:05:49  falgout
 * Added general BC capability.
 *
 * Revision 1.2  1995/12/21  00:56:38  steve
 * Added copyright
 *
 * Revision 1.1  1993/08/20  21:22:32  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef HEAD_HEADER
#define HEAD_HEADER

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


/* head.c */
Databox *HHead ANSI_PROTO((Databox *h , GridType grid_type ));
Databox *PHead ANSI_PROTO((Databox *h , GridType grid_type ));

#undef ANSI_PROTO

#endif

