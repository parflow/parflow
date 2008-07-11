/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.12 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Error header file
 *
 * (C) 1993 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Log: error.h,v $
 Revision 1.12  1997/09/20 00:16:29  ssmith
 This checkin updates the TCL/TK interface to use the new package
 mechanism in TCL rather than re-writing the main TCL routine.
 Also namespace control was partially implemented to help prevent
 name conflicts.
 Globals where removed.

 Revision 1.11  1997/06/05 23:31:04  ssmith
 Added reload feature

 Revision 1.10  1997/05/13 21:47:06  ssmith
 Added elnarge box command

 Revision 1.9  1997/03/07 00:30:59  shumaker
 Added two new pftools, pfgetsubbox and pfbfcvel

 * Revision 1.8  1996/08/21  18:45:47  steve
 * Fixed some compiler warnings and problem with running in different directories
 *
 * Revision 1.7  1996/08/10  06:35:07  mccombjr
 * *** empty log message ***
 *
 * Revision 1.6  1995/12/21  00:56:38  steve
 * Added copyright
 *
 * Revision 1.5  1995/11/14  02:47:42  steve
 * Fixed ^M problems
 *
 * Revision 1.4  1995/11/14  02:12:05  steve
 * Fixed Compiler warnings under Win32
 *
 * Revision 1.3  1995/10/23  18:11:49  steve
 * Added support for HDF
 * Made varargs be ANSI
 *
 * Revision 1.2  1994/05/18  23:33:28  falgout
 * Changed Error stuff, and print interactive message to stderr.
 *
 * Revision 1.1  1993/04/02  23:13:24  falgout
 * Initial revision
 *
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef ERROR_HEADER
#define ERROR_HEADER

#include <stdio.h>
#include <stdarg.h>

#ifdef PF_HAVE_HDF
#include <hdf.h>
#endif

#include <tcl.h>
#include "databox.h"

/*-----------------------------------------------------------------------
 * Error messages 
 *-----------------------------------------------------------------------*/

#if PF_HAVE_HDF

static char *LOADSDSUSAGE  = "Usage: pfloadsds filename [ds_num]\n";
static char *SAVESDSUSAGE  = "Usage: pfsavesds dataset -filetype filename\n       file types: float64 float32 int32 uint32 int16 uint16 int8 uint8\n";

#endif
static char *BFCVELUSAGE     = "Usage: pfbfcvel conductivity phead\n";
static char *GETSUBBOXUSAGE   = "Usage: pfgetsubbox dataset il jl kl iu ju ku\n"; 
static char *ENLARGEBOXUSAGE   = "Usage: pfenlargebox dataset new_nx new_ny new_nz\n"; 
static char *LOADPFUSAGE   = "Usage: pfload [-filetype] filename\n       file types: pfb pfsb sa sb rsa\n";
static char *RELOADUSAGE   = "Usage: pfreload dataset\n";
static char *SAVEPFUSAGE   = "Usage: pfsave dataset -filetype filename\n       file types: pfb sa sb\n";
static char *GETLISTUSAGE  = "Usage: pfgetlist [dataset]\n";
static char *GETELTUSAGE   = "Usage: pfgetelt dataset i j k\n"; 
static char *GETGRIDUSAGE  = "Usage: pfgetgrid dataset\n";
static char *GRIDTYPEUSAGE = "Usage: pfgridtype [vertex | cell]\n";
static char *CVELUSAGE     = "Usage: pfcvel conductivity phead\n";
static char *VVELUSAGE     = "Usage: pfvvel conductivity phead\n";
static char *VMAGUSEAGE    = "Usage: pfvmag datasetx datasety datasetz\n";
static char *HHEADUSAGE    = "Usage: pfhhead dataset\n";
static char *PHEADUSAGE    = "Usage: pfphead dataset\n";
static char *FLUXUSAGE     = "Usage: pfflux conductivity hhead\n";
static char *AXPYUSAGE     = "Usage: pfaxpy alpha datasetx datasety\n";
static char *GETSTATSUSAGE = "Usage: pfstats dataset\n";
static char *MDIFFUSAGE    = "Usage: pfmdiff datasetp datasetq sig_digs [abs_zero]\n";
static char *SAVEDIFFUSAGE = "Usage: pfsavediff datasetp datasetq sig_digs [abs_zero] -file filename\n";
static char *DIFFELTUSAGE  = "Usage: pfdiffelt datasetp datasetq i j k sig_digs [abs_zero]\n";
static char *NEWGRIDUSAGE  = "Usage: pfnewgrid {nx ny nz} {x y z} {dx dy dz} label\n       Types: int nx, ny, nz;  double x, y, z, dx, dy, dz;\n";
static char *NEWLABELUSAGE = "Usage: pfnewlabel dataset newlabel\n";
static char *DELETEUSAGE   = "Usage: pfdelete dataset\n";

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif

/* Function prototypes for error checking functions */

int  SameDimensions ANSI_PROTO((Databox *databoxp, Databox *databoxq));
int  InRange ANSI_PROTO((int i, int j, int k, Databox *databox));
int  IsValidFileType ANSI_PROTO((char *option));
char *GetValidFileExtension ANSI_PROTO((char *filename));

/* Function prototypes for creating error messages  */

void InvalidOptionError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void InvalidArgError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void WrongNumArgsError ANSI_PROTO((Tcl_Interp *interp, char *usage));
void MissingOptionError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void MissingFilenameError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void InvalidFileExtensionError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void NotAnIntError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void NotADoubleError ANSI_PROTO((Tcl_Interp *interp, int argnum, char *usage));
void NumberNotPositiveError ANSI_PROTO((Tcl_Interp *interp, int argnum));
void SetNonExistantError ANSI_PROTO((Tcl_Interp *interp, char *hashkey)); 
void MemoryError ANSI_PROTO((Tcl_Interp *interp));
void ReadWriteError ANSI_PROTO((Tcl_Interp *interp));
void OutOfRangeError ANSI_PROTO((Tcl_Interp *interp, int i, int j, int k));
void DimensionError ANSI_PROTO((Tcl_Interp *interp));

#undef ANSI_PROTO


#endif
