/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1 $
 *********************************************************************EHEADER*/

#ifndef ENLARGE_HEADER
#define ENLARGE_HEADER

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

Databox *EnlargeBox ANSI_PROTO((Databox *inbox , 
				int new_nx , int new_nj , int new_nz ));

#undef ANSI_PROTO

#endif

