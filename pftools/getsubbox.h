
#ifndef GETSUBBOX_HEADER
#define GETSUBBOX_HEADER

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

Databox *CompSubBox ANSI_PROTO((Databox *fun , 
				int il , int jl , int kl , 
				int iu , int ju , int ku ));

#undef ANSI_PROTO

#endif

