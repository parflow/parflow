/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _USERGRID_HEADER
#define _USERGRID_HEADER

/*--------------------------------------------------------------------------
 * Background:
 *   Structure describing a uniform grid in real-space.
 *--------------------------------------------------------------------------*/

typedef struct
{
   double   X,  Y,  Z;    /* Bottom-lower-left corner in real-space */
   double   DX, DY, DZ;   /* Spacing in each coordinate direction */

} Background;

/*--------------------------------------------------------------------------
 * Accessor macros: Background
 *--------------------------------------------------------------------------*/

#define BackgroundX(background)   ((background) -> X)
#define BackgroundY(background)   ((background) -> Y)
#define BackgroundZ(background)   ((background) -> Z)

#define BackgroundDX(background)  ((background) -> DX)
#define BackgroundDY(background)  ((background) -> DY)
#define BackgroundDZ(background)  ((background) -> DZ)


#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif


/* usergrid.c */

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
/* usergrid.c */
Background *ReadBackground P((Tcl_Interp *interp ));
void FreeBackground P((Background *background ));
Subgrid *ReadUserSubgrid P((Tcl_Interp *interp ));
Grid *ReadUserGrid P((Tcl_Interp *interp ));
void FreeUserGrid P((Grid *user_grid ));
SubgridArray *DistributeUserGrid P((Grid *user_grid , int num_procs , int P , int Q , int R ));
 
#undef P

#endif
