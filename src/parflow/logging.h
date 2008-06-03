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
 * Header info for logging
 *
 *****************************************************************************/

#ifndef _LOGGING_HEADER
#define _LOGGING_HEADER

#define IfLogging(level) \
if ((level <= GlobalsLoggingLevel) && (!amps_Rank(amps_CommWorld)))

#endif
