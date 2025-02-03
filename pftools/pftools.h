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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif
#define WS " \t\n"
#define MAX_KEY_SIZE 32

/*-----------------------------------------------------------------------
 * Data structure and accessor macros
 *-----------------------------------------------------------------------*/

typedef struct {
  Tcl_HashTable members;
  GridType grid_type;
  int total_members;
  unsigned long num;
} Data;

#define DataMembers(data)   ((data)->members)
#define DataGridType(data)  ((data)->grid_type)
#define DataTotalMem(data)  ((data)->total_members)
#define DataNum(data)       ((data)->num)
#define DataMember(data, hashkey, entryPtr)                                 \
        (((entryPtr = Tcl_FindHashEntry(&DataMembers(data), hashkey)) != 0) \
   ? (Databox*)Tcl_GetHashValue(entryPtr)                                   \
   : (Databox*)NULL)
#define FreeData(data) (free((Data*)data))


/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

/* pftools.c */
int PFDistCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
char *GetString(Tcl_Interp *interp, char *key);
int GetInt(Tcl_Interp *interp, char *key);
int GetIntDefault(Tcl_Interp *interp, char *key, int def);
double GetDouble(Tcl_Interp *interp, char *key);
Data *InitPFToolsData(void);
int AddData(Data *data, Databox *databox, char *label, char *hashkey);
void PFTExitProc(ClientData clientData);
int keycompare(const void *key1, const void *key2);
int GetSubBoxCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int EnlargeBoxCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ReLoadPFCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int LoadPFCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int LoadSDSCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SavePFCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SaveSDSCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GetListCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GetEltCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GetGridCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SetGridCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GridTypeCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CVelCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int VVelCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int BFCVelCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int VMagCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int HHeadCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int PHeadCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int FluxCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int NewGridCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int NewLabelCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int AxpyCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SumCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellSumCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellDiffCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellMultCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellDivCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellSumConstCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellDiffConstCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellMultConstCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int CellDivConstCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GetStatsCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int MDiffCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SaveDiffCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int DiffEltCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int DeleteCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ComputeTopCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ComputeBottomCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ComputeDomainCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int PrintDomainCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int Extract2DDomainCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int BuildDomainCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int PFDistOnDomainCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ExtractTopCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SurfaceStorageCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SubsurfaceStorageCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int GWStorageCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SurfaceRunoffCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int WaterTableDepthCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);

int SavePFVTKCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int MakePatchySolidCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int pfsolFmtConvert(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);

void Axpy(double alpha, Databox *X, Databox *Y);
void Sum(Databox *X, double *sum);
void CellSum(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellDiff(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellMult(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellDiv(Databox *X, Databox *Y, Databox *mask, Databox *sum);
void CellSumConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellDiffConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellMultConst(Databox *X, double val, Databox *mask, Databox *sum);
void CellDivConst(Databox *X, double val, Databox *mask, Databox *sum);

int SlopeXUpwindCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SlopeYUpwindCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SlopeXD4Command(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SlopeYD4Command(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SlopeD8Command(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int UpstreamAreaCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int FillFlatsCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int PitFillCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int MovingAvgCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int SegmentD8Command(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int ChildD8Command(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int FlintsLawCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int FlintsLawFitCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int FlintsLawByBasinCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);

int SatTransmissivityCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int TopoIndexCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int EffectiveRechargeCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int TopoRechargeCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int TopoDeficitCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int TopoDeficitToWTCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);
int HydroStatFromWTCommand(ClientData clientData, Tcl_Interp *interp, int argc, char *argv []);

#ifdef __cplusplus
}
#endif

#endif
