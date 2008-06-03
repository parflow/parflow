/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * Header file for `well.c'
 *
 * (C) 1996 Regents of the University of California.
 *
 * see info_header.h for complete information
 *
 *                               History
 *-----------------------------------------------------------------------------
 $Revision: 1.3 $
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#ifndef PFWELL_DATA_HEADER
#define PFWELL_DATA_HEADER

#include <string.h>
#include <stdarg.h>
#include <malloc.h>

#include "general.h"

#include "well.h"

/*----------------------------------------------------------------
 * Well List Member structure
 *----------------------------------------------------------------*/

typedef struct well_list_member
{
   double                    time;
   WellDataPhysical        **well_data_physicals;
   WellDataValue           **well_data_values;
   WellDataStat            **well_data_stats;
   struct well_list_member  *next_well_list_member;
} WellListMember;


/*--------------------------------------------------------------------------
 * Accessor macros: WellListMember
 *--------------------------------------------------------------------------*/
#define WellListMemberTime(well_list_member)\
        ((well_list_member) -> time)

#define WellListMemberWellDataPhysicals(well_list_member)\
        ((well_list_member) -> well_data_physicals)
#define WellListMemberWellDataPhysical(well_list_member,i)\
        ((well_list_member) -> well_data_physicals[i])

#define WellListMemberWellDataValues(well_list_member)\
        ((well_list_member) -> well_data_values)
#define WellListMemberWellDataValue(well_list_member,i)\
        ((well_list_member) -> well_data_values[i])

#define WellListMemberWellDataStats(well_list_member)\
        ((well_list_member) -> well_data_stats)
#define WellListMemberWellDataStat(well_list_member,i)\
        ((well_list_member) -> well_data_stats[i])

#define WellListMemberNextWellListMember(well_list_member)\
        ((well_list_member) -> next_well_list_member)

/*-----------------------------------------------------------------------
 * function prototypes
 *-----------------------------------------------------------------------*/

#ifdef __STDC__
# define        ANSI_PROTO(s) s
#else
# define ANSI_PROTO(s) ()
#endif

void WriteWellListData ANSI_PROTO((FILE *fd, double time, WellDataStat *well_data_stat, int num_phases, int num_components));

#undef ANSI_PROTO

#endif
