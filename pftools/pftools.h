/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.24 $
 *********************************************************************EHEADER*/
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
#include <tk.h>

#include "databox.h"
#include "readdatabox.h"
#include "printdatabox.h"
#include "velocity.h"
#include "head.h"
#include "flux.h"
#include "stats.h"
#include "diff.h"
#include "error.h"
#include "getsubbox.h"
#include "enlargebox.h"
#include "file.h"
#include "load.h"

#include "region.h"
#include "grid.h"
#include "usergrid.h"

#include "general.h"


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
        ((int)(entryPtr = Tcl_FindHashEntry(&DataMembers(data), hashkey)) \
        ? (Databox *)Tcl_GetHashValue(entryPtr) \
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
int GetStatsCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int MDiffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int SaveDiffCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int DiffEltCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));
int DeleteCommand P((ClientData clientData , Tcl_Interp *interp , int argc , char *argv []));

#undef P

#endif

