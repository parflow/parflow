/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * NewDatabox and FreeDatabox
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: databox.c,v $
 Revision 1.1.1.1  2006/02/14 23:05:52  kollet
 CLM.PF_1.0

 Revision 1.1.1.1  2006/02/14 18:51:22  kollet
 CLM.PF_1.0

 Revision 1.14  1997/09/20 00:16:28  ssmith
 This checkin updates the TCL/TK interface to use the new package
 mechanism in TCL rather than re-writing the main TCL routine.
 Also namespace control was partially implemented to help prevent
 name conflicts.
 Globals where removed.

 Revision 1.13  1996/09/12 23:32:04  steve
 Removed old pfio stuff

 * Revision 1.12  1996/08/21  21:21:59  steve
 * Fixed Compile problems under IRIX
 *
 * Revision 1.11  1996/08/10  06:34:57  mccombjr
 * *** empty log message ***
 *
 * Revision 1.10  1995/12/21  00:56:38  steve
 * Added copyright
 *
 * Revision 1.9  1995/11/29  19:13:59  steve
 * Fixed env names; removed some debugging prints; added Win32 batch files
 *
 * Revision 1.8  1995/11/14  02:47:43  steve
 * Fixed ^M problems
 *
 * Revision 1.7  1995/11/14  02:12:06  steve
 * Fixed Compiler warnings under Win32
 *
 * Revision 1.6  1995/10/11  18:01:14  falgout
 * Added a PrintGrid routine.
 *
 * Revision 1.5  1995/06/27  21:36:13  falgout
 * Added (X, Y, Z) coordinates to databox structure.
 *
 * Revision 1.4  1994/05/18  23:33:28  falgout
 * Changed Error stuff, and print interactive message to stderr.
 *
 * Revision 1.3  1994/02/09  23:26:52  ssmith
 * Cleaned up pftools and added pfload/pfunload
 *
 * Revision 1.2  1993/04/13  17:41:05  falgout
 * added dx, dy, dz
 *
 * Revision 1.1  1993/04/02  23:13:24  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include <malloc.h>
#include <stdlib.h>
#include "databox.h"


/*-----------------------------------------------------------------------
 * create new Databox structure
 *-----------------------------------------------------------------------*/

Databox         *NewDatabox(nx, ny, nz, x, y, z, dx, dy, dz)
int             nx, ny, nz;
double          x,  y,  z;
double          dx, dy, dz;
{
   Databox         *new;

   if ((new = calloc(1, sizeof (Databox))) == NULL)
      return((Databox *)NULL);

   if ((DataboxCoeffs(new) = calloc((nx * ny * nz), sizeof (double))) == NULL){
      free(new);
      return((Databox *)NULL);
   }

   DataboxNx(new) = nx;
   DataboxNy(new) = ny;
   DataboxNz(new) = nz;

   DataboxX(new)  = x;
   DataboxY(new)  = y;
   DataboxZ(new)  = z;

   DataboxDx(new) = dx;
   DataboxDy(new) = dy;
   DataboxDz(new) = dz;

   return new;
}


/*-----------------------------------------------------------------------
 * print Databox grid info
 *-----------------------------------------------------------------------*/

void            GetDataboxGrid(interp, databox)
Tcl_Interp     *interp;
Databox        *databox;
{
   char         triple[256];
   Tcl_DString  result;
 
   Tcl_DStringInit(&result);

   sprintf(triple, "%d %d %d",
	   DataboxNx(databox), DataboxNy(databox), DataboxNz(databox));
   Tcl_DStringAppendElement(&result, triple);
   sprintf(triple, "%f %f %f",
	   DataboxX(databox), DataboxY(databox), DataboxZ(databox));
   Tcl_DStringAppendElement(&result, triple);
   sprintf(triple, "%f %f %f",
	   DataboxDx(databox), DataboxDy(databox), DataboxDz(databox));
   Tcl_DStringAppendElement(&result, triple); 
   Tcl_DStringResult(interp, &result); 
}

/*-----------------------------------------------------------------------
 * free Databox structure
 *-----------------------------------------------------------------------*/

void         FreeDatabox(databox)
Databox      *databox;
{
   free(DataboxCoeffs(databox));

   free(databox);
}


