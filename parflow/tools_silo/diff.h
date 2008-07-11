/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/*****************************************************************************
 * Header file for `diff.c'
 *
 * (C) 1993 Regents of the University of California.
 *
 *----------------------------------------------------------------------------
 * $Revision: 1.1.1.1 $
 *
 *----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef DIFF_HEADER
#define DIFF_HEADER

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


/* diff.c */
void SigDiff ANSI_PROTO((Databox *v1 , Databox *v2 , int m , double absolute_zero, Tcl_DString *result, FILE *fp ));
double DiffElt ANSI_PROTO((Databox *v1 , Databox *v2 , int i, int j, int k, int m , double absolute_zero));
void MSigDiff ANSI_PROTO((Databox *v1 , Databox *v2 , int m, double absolute_zero, Tcl_DString *result));

#undef ANSI_PROTO

#endif

