/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

amps_File amps_Fopen(filename, type)
char *filename;
char *type;
{
   char temp[255];

   sprintf(temp, "%s.%05d", filename, amps_Rank(amps_CommWorld));

   return fopen(temp, type);
}
