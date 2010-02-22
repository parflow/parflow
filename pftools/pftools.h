/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/
/******************************************************************************
 * Header file for `pftools' program
 *
 * (C) 1993 Regents of the University of California.
 *
 *-----------------------------------------------------------------------------
 * $Revision: 1.24 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef PFTOOLS_HEADER
#define PFTOOLS_HEADER

#include "parflow_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#ifdef HAVE_HDF4
#include <hdf.h>
#endif

#include <tcl.h>

#include "databox.h"

#ifndef NULL
#define NULL ((void *)0)
#endif
#define WS " \t\n"
#define MAX_KEY_SIZE 32


/*-----------------------------------------------------------------------
 * Data structure and accessor macros 
 *-----------------------------------------------------------------------*/

typedef struct
{
   Tcl_HashTable   members;
   GridType        grid_type;
   int             total_members;
   int             num;

} Data;

#define DataMembers(data)   ((data) -> members)
#define DataGridType(data)  ((data) -> grid_type)
#define DataTotalMem(data)  ((data) -> total_members)
#define DataNum(data)       ((data) -> num)
#define DataMember(data, hashkey, entryPtr) \
   ( ((entryPtr = Tcl_FindHashEntry(&DataMembers(data), hashkey)) != 0)	\
   ? (Databox *)Tcl_GetHashValue(entryPtr)				\
   : (Databox *) NULL)
#define FreeData(data) (free((Data *)data))


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* pftools.c */
int PFDistCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
char *GetString P((Tcl_Interp *interp , char *key ));
int GetInt P((Tcl_Interp *interp , char *key ));
int GetIntDefault P((Tcl_Interp *interp , char *key , int def ));
double GetDouble P((Tcl_Interp *interp , char *key ));
Data *InitPFToolsData P((void ));
int AddData P((Data *data , Databox *databox , char *label , char *hashkey ));
void PFTExitProc P((ClientData clientData ));
int keycompare P((const void *key1 , const void *key2 ));
int GetSubBoxCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int EnlargeBoxCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int ReLoadPFCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int LoadPFCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int LoadSDSCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SavePFCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SaveSDSCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GetListCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GetEltCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GetGridCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GridTypeCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CVelCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int VVelCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int BFCVelCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int VMagCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int HHeadCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int PHeadCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int FluxCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int NewGridCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int NewLabelCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int AxpyCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SumCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellSumCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellDiffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellMultCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellDivCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellSumConstCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellDiffConstCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellMultConstCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int CellDivConstCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GetStatsCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int MDiffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SaveDiffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int DiffEltCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int DeleteCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int ComputeTopCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int ExtractTopCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SurfaceStorageCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SubsurfaceStorageCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int GWStorageCommand P((ClientData clientData, Tcl_Interp *interp , int argc , char *argv []));
int SurfaceRunoffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int WaterTableDepthCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));

void Axpy(double alpha, Databox *X,  Databox *Y);
void Sum(Databox *X,  double *sum);
void CellSum(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellDiff(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellMult(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellDiv(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellSumConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellDiffConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellMultConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellDivConst(Databox *X, double val, Databox *mask, Databox *sum);
#undef P

#endif

