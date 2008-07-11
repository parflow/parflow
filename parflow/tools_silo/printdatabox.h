/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `printdatabox.c'
 *
 * (C) 1993 Regents of the University of California.
 *
 * $Revision: 1.1.1.1 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef PRINTDATABOX_HEADER
#define PRINTDATABOX_HEADER


#ifdef PF_HAVE_HDF
#include <hdf.h>
#endif

#include <stdio.h>

#include "databox.h"

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/


#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* printdatabox.c */
void PrintSimpleA P((FILE *fp , Databox *v ));
void PrintSimpleB P((FILE *fp , Databox *v ));
void PrintParflowB P((FILE *fp , Databox *v ));
void PrintAVSField P((FILE *fp , Databox *v ));
int  PrintSDS P((char *filename , int type , Databox *v ));
 
#undef P

#endif
