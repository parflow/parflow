/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.4 $
 *********************************************************************EHEADER*/
#ifndef FILE_HEADER
#define FILE_HEADER

/*-----------------------------------------------------------------------
 * Supported File types 
 *-----------------------------------------------------------------------*/
#define ParflowB     1
#define SimpleA      2
#define SimpleB      3

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* file.c */
int FileType ANSI_PROTO((char *filename ));
Databox *Read ANSI_PROTO((int type , char *filename ));

#undef ANSI_PROTO

#endif
