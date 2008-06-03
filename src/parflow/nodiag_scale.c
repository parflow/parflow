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
 * FILE:	nodiag_scale.c
 *
 * WRITTEN BY:	Bill Bosl
 *		phone: (510) 423-2873
 *		e-mail: wjbosl@llnl.gov
 *
 * FUNCTIONS IN THIS FILE:
 * NoDiagScale, InitNoDiagScale, FreeNoDiagScale, NewNoDiagScale
 *
 * DESCRIPTION:
 * This module does nothing to the input matrix. It's primary
 * purpose is to give a "do nothing" alternative for the 
 * diagonal scaling options.
 *
 *****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * NoDiagScale
 *--------------------------------------------------------------------------*/

void         NoDiagScale(x, A, b, flag)
Matrix      *A;
Vector      *x;
Vector      *b;
int	     flag;
{
  return;
}

/*--------------------------------------------------------------------------
 * NoDiagScaleInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *NoDiagScaleInitInstanceXtra(grid)
Grid      *grid;
{
   return ThisPFModule;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  NoDiagScaleFreeInstanceXtra()
{
   return;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *NoDiagScaleNewPublicXtra(char *name)
{
   return ThisPFModule;
}


/*--------------------------------------------------------------------------
 * NoDiagScaleFreePublicXtra
 *--------------------------------------------------------------------------*/

void  NoDiagScaleFreePublicXtra()
{
   return;
}

/*--------------------------------------------------------------------------
 * NoDiagScaleSizeOfTempData
 *--------------------------------------------------------------------------*/

int  NoDiagScaleSizeOfTempData()
{
   return 0;
}
