/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
 *********************************************************************EHEADER*/
#ifndef LOAD_HEADER
#define LOAD_HEADER

#include "databox.h"
#include "grid.h"
#include "usergrid.h"


#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* load.c */
void LoadParflowB ANSI_PROTO((char *filename , SubgridArray *all_subgrids , Background *background , Databox *databox ));

#undef ANSI_PROTO

#endif
