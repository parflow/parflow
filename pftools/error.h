/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/
/*****************************************************************************
* Error header file
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Log: error.h,v $
* Revision 1.12  1997/09/20 00:16:29  ssmith
* This checkin updates the TCL/TK interface to use the new package
* mechanism in TCL rather than re-writing the main TCL routine.
* Also namespace control was partially implemented to help prevent
* name conflicts.
* Globals where removed.
*
* Revision 1.11  1997/06/05 23:31:04  ssmith
* Added reload feature
*
* Revision 1.10  1997/05/13 21:47:06  ssmith
* Added elnarge box command
*
* Revision 1.9  1997/03/07 00:30:59  shumaker
* Added two new pftools, pfgetsubbox and pfbfcvel
*
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

#include "parflow_config.h"

#include <stdio.h>
#include <stdarg.h>

#ifdef HAVE_HDF4
#include <hdf.h>
#endif

#include <tcl.h>
#include "databox.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------
 * Error messages
 *-----------------------------------------------------------------------*/

#if HAVE_HD5

static char *LOADSDSUSAGE = "Usage: pfloadsds filename [ds_num]\n";
static char *SAVESDSUSAGE = "Usage: pfsavesds dataset -filetype filename\n       file types: float64 float32 int32 uint32 int16 uint16 int8 uint8\n";

#endif
static char *BFCVELUSAGE = "Usage: pfbfcvel conductivity phead\n";
static char *GETSUBBOXUSAGE = "Usage: pfgetsubbox dataset il jl kl iu ju ku\n";
static char *ENLARGEBOXUSAGE = "Usage: pfenlargebox dataset new_nx new_ny new_nz\n";
static char *LOADPFUSAGE = "Usage: pfload [-filetype] filename\n       file types: pfb pfsb sa sb rsa\n";
static char *RELOADUSAGE = "Usage: pfreload dataset\n";
static char *SAVEPFUSAGE = "Usage: pfsave dataset -filetype filename\n       file types: pfb sa sb\n";
static char *GETLISTUSAGE = "Usage: pfgetlist [dataset]\n";
static char *GETELTUSAGE = "Usage: pfgetelt dataset i j k\n";
static char *GETGRIDUSAGE = "Usage: pfgetgrid dataset\n";
static char *SETGRIDUSAGE = "Usage: pfsetgrid { nx ny nz } { x y z } { dx dy dz } dataset\n       Types: int nx, ny, nz;  double x, y, z, dx, dy, dz;\n";
static char *GRIDTYPEUSAGE = "Usage: pfgridtype [vertex | cell]\n";
static char *CVELUSAGE = "Usage: pfcvel conductivity phead\n";
static char *VVELUSAGE = "Usage: pfvvel conductivity phead\n";
static char *VMAGUSEAGE = "Usage: pfvmag datasetx datasety datasetz\n";
static char *HHEADUSAGE = "Usage: pfhhead dataset\n";
static char *PHEADUSAGE = "Usage: pfphead dataset\n";
static char *FLUXUSAGE = "Usage: pfflux conductivity hhead\n";
static char *AXPYUSAGE = "Usage: pfaxpy alpha datasetx datasety\n";
static char *SUMUSAGE = "Usage: pfsum datasetx\n";
static char *CELLSUMUSAGE = "Usage: pfcellsum datasetx datasety mask\n";
static char *CELLDIFFUSAGE = "Usage: pfcelldiff datasetx datasety mask\n";
static char *CELLMULTUSAGE = "Usage: pfcellmult datasetx datasety mask\n";
static char *CELLDIVUSAGE = "Usage: pfcelldiv datasetx datasety mask\n";
static char *CELLSUMCONSTUSAGE = "Usage: pfcellsumconst datasetx val mask\n";
static char *CELLDIFFCONSTUSAGE = "Usage: pfcelldiffconst datasetx val mask\n";
static char *CELLMULTCONSTUSAGE = "Usage: pfcellmultconst datasetx val mask\n";
static char *CELLDIVCONSTUSAGE = "Usage: pfcelldivconst datasetx val mask\n";
static char *GETSTATSUSAGE = "Usage: pfstats dataset\n";
static char *MDIFFUSAGE = "Usage: pfmdiff datasetp datasetq sig_digs [abs_zero]\n";
static char *SAVEDIFFUSAGE = "Usage: pfsavediff datasetp datasetq sig_digs [abs_zero] -file filename\n";
static char *DIFFELTUSAGE = "Usage: pfdiffelt datasetp datasetq i j k sig_digs [abs_zero]\n";
static char *NEWGRIDUSAGE = "Usage: pfnewgrid {nx ny nz} {x y z} {dx dy dz} label\n       Types: int nx, ny, nz;  double x, y, z, dx, dy, dz;\n";
static char *NEWLABELUSAGE = "Usage: pfnewlabel dataset newlabel\n";
static char *PFCOMPUTETOPUSAGE = "Usage: pfcomputetop mask\n";
static char *PFCOMPUTEBOTTOMUSAGE = "Usage: pfcomputebottom mask\n";
static char *PFCOMPUTEDOMAINUSAGE = "Usage: pfcomputedomain top\n";
static char *PFPRINTDOMAINUSAGE = "Usage: pfcomputedomain subgrid_array\n";
static char *PFEXTRACT2DDOMAINUSAGE = "Usage: pfexctract2Ddomain subgrid_array\n";
static char *PFBUILDDOMAINUSAGE = "Usage: pfbuilddomain\n";
static char *PFDISTONDOMAINUSAGE = "Usage: pfdistondomain filename subgrid_array\n";
static char *PFEXTRACTTOPUSAGE = "Usage: pfextracttop top dataset\n";
static char *PFSURFACESTORAGEUSAGE = "Usage: pfsuracestorge top pressure\n";
static char *PFSUBSURFACESTORAGEUSAGE = "Usage: pfsubsuracestorge mask porosity pressure saturation specific_storage\n";
static char *PFGWSTORAGEUSAGE = "Usage: pfgwstorge mask porosity pressure saturation specific_storage\n";
static char *PFSURFACERUNOFFUSAGE = "Usage: pfsuracerunoff top slope_x slope_y mannings pressure\n";
static char *PFWATERTABLEDEPTHUSAGE = "Usage: pfwatertabledepth top saturation\n";
static char *PFSLOPEXUSAGE = "Usage: pfslopex dem\n";
static char *PFSLOPEYUSAGE = "Usage: pfslopey dem\n";
static char *PFUPSTREAMAREAUSAGE = "Usage: pfupstreamarea dem sx sy\n";
static char *PFFILLFLATSUSAGE = "Usage: pffillflats dem \n";
static char *PFPITFILLDEMUSAGE = "Usage: pfpitfilldem dem dpit maxiter\n";
static char *PFMOVINGAVGDEMUSAGE = "Usage: pfmovingavgdem dem wsize maxiter\n";
static char *PFSATTRANSUSAGE = "Usage: pfsattrans nlayers mask perm\n";
static char *PFTOPODEFTOWTUSAGE = "Usage: pftopowt deficit porosity ssat sres mask top \n";
static char *PFTOPODEFUSAGE = "Usage: pftopodeficit profile m trans dem sx sy recharge ssat sres porosity mask\n              profile = Exponential or Linear\n";
static char *PFTOPOINDEXUSAGE = "Usage: pftopoindex dem sx sy\n";
static char *PFEFFRECHARGEUSAGE = "Usage: pfeffectiverecharge precip et sx sy dem\n";
static char *PFTOPORECHARGEUSAGE = "Usage: pftoporecharge nriver riverfile trans dem sx sy\n";
static char *PFHYDROSTATUSAGE = "Usage: pfhydrostatic wtdepth mask top\n";
static char *PFSLOPEXD4USAGE = "Usage: pfslopexD4 dem\n";
static char *PFSLOPEYD4USAGE = "Usage: pfslopeyD4 dem\n";
static char *PFSLOPED8USAGE = "Usage: pfslopeD8 dem\n";
static char *PFSEGMENTD8USAGE = "Usage: pfsegmentD8 dem\n";
static char *PFCHILDD8USAGE = "Usage: pfchildD8 dem\n";
static char *PFFLINTSLAWDEMUSAGE = "Usage: pfflintslaw dem c p\n";
static char *PFFLINTSLAWFITUSAGE = "Usage: pfflintslawfit dem c0 p0 maxiter\n";
static char *PFFLINTSLAWBYBASINUSAGE = "Usage: pfflintslawbybasin dem c0 p0 maxiter\n";
static char *DELETEUSAGE = "Usage: pfdelete dataset\n";


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

/* Function prototypes for error checking functions */

int SameDimensions(Databox *databoxp, Databox *databoxq);
int InRange(int i, int j, int k, Databox *databox);
int IsValidFileType(char *option);
char *GetValidFileExtension(char *filename);

/* Function prototypes for creating error messages  */

void InvalidOptionError(Tcl_Interp *interp, int argnum, char *usage);
void InvalidArgError(Tcl_Interp *interp, int argnum, char *usage);
void WrongNumArgsError(Tcl_Interp *interp, char *usage);
void MissingOptionError(Tcl_Interp *interp, int argnum, char *usage);
void MissingFilenameError(Tcl_Interp *interp, int argnum, char *usage);
void InvalidFileExtensionError(Tcl_Interp *interp, int argnum, char *usage);
void NotAnIntError(Tcl_Interp *interp, int argnum, char *usage);
void NotADoubleError(Tcl_Interp *interp, int argnum, char *usage);
void NumberNotPositiveError(Tcl_Interp *interp, int argnum);
void SetNonExistantError(Tcl_Interp *interp, char *hashkey);
void MemoryError(Tcl_Interp *interp);
void ReadWriteError(Tcl_Interp *interp);
void OutOfRangeError(Tcl_Interp *interp, int i, int j, int k);
void DimensionError(Tcl_Interp *interp);

#ifdef __cplusplus
}
#endif

#endif
