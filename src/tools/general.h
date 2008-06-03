/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/

#ifndef _GENERAL_HEADER
#define _GENERAL_HEADER

#include <stdlib.h>
#include <stdio.h>

/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

#define talloc(type, count) \
((count) ? (type *) malloc((unsigned int)(sizeof(type) * (count))) : NULL)

#define ctalloc(type, count) \
((count) ? (type *) calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else


/*--------------------------------------------------------------------------
 * Define max and min functions
 *--------------------------------------------------------------------------*/

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef min
#define min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef round
#define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif


/*--------------------------------------------------------------------------
 * Define various flags
 *--------------------------------------------------------------------------*/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ON  1
#define OFF 0


#endif

