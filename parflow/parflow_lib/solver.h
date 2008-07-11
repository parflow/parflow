/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Stuctures for the solver.
 *
 *****************************************************************************/

#ifndef _SOLVER_HEADER
#define _SOLVER_HEADER


/*--------------------------------------------------------------------------
 * Some global information
 *--------------------------------------------------------------------------*/

#define ArithmeticMean(a, b)  ( 0.5*((a) + (b)) )
#define GeometricMean(a, b)   ( sqrt((a)*(b)) )
#define HarmonicMean(a, b)    ( ((a) + (b)) ? (2.0*(a)*(b))/((a) + (b)) : 0 )
#define UpstreamMean(a, b, c, d) ( (( a - b ) >= 0) ? c : d )

#define CellFaceConductivity  HarmonicMean


#endif





