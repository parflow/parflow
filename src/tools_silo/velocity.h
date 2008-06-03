/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `velocity.c'
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: velocity.h,v $
 Revision 1.1.1.1  2006/02/14 23:05:52  kollet
 CLM.PF_1.0

 Revision 1.1.1.1  2006/02/14 18:51:22  kollet
 CLM.PF_1.0

 Revision 1.6  1997/03/07 00:31:01  shumaker
 Added two new pftools, pfgetsubbox and pfbfcvel

 * Revision 1.5  1995/12/21  00:56:38  steve
 * Added copyright
 *
 * Revision 1.4  1995/08/17  21:48:59  falgout
 * Added a vertex velocity capability.
 *
 * Revision 1.3  1993/09/01  01:29:54  falgout
 * Added ability to compute velocities as well as velocity magnitudes.
 *
 * Revision 1.2  1993/08/20  21:22:32  falgout
 * Added PrintParflowB, modified velocity computation, added HHead and PHead.
 *
 * Revision 1.1  1993/04/13  18:00:21  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef VELOCITY_HEADER
#define VELOCITY_HEADER

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


/* velocity.c */
Databox **CompBFCVel ANSI_PROTO((Databox *k , Databox *h ));
Databox **CompCellVel ANSI_PROTO((Databox *k , Databox *h ));
Databox **CompVertVel ANSI_PROTO((Databox *k , Databox *h ));
Databox *CompVMag ANSI_PROTO((Databox *vx , Databox *vy , Databox *vz ));

#undef ANSI_PROTO

#endif

