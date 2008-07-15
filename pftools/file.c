/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.5 $
 *********************************************************************EHEADER*/
#include <string.h>
#include <stdlib.h>

#include "databox.h"
#include "readdatabox.h"
#include "file.h"


int FileType(filename)
char *filename;
{
   char *ptr;

   if( (ptr = strrchr(filename, '.')) == NULL)
      return -1;

   ptr++;

   if      (strcmp(ptr, "pfb") == 0)
      return ParflowB;
   else if (strcmp(ptr, "sa") == 0)
      return SimpleA;
   else if (strcmp(ptr, "sb") == 0)
      return SimpleB;
   else
      return -1;
}

/*-----------------------------------------------------------------------
 * Read the input file
 *-----------------------------------------------------------------------*/

Databox *Read(type, filename)
int type;
char *filename;
{

   Databox *indatabox;

   switch(type)
   {
   case ParflowB:
      indatabox = ReadParflowB(filename);
      break;
   case SimpleA:
      indatabox = ReadSimpleA(filename);
      break;
   case SimpleB:
      indatabox = ReadSimpleB(filename);
      break;
   default:
      printf("Cannot read from that file type\n");
      exit(1);
   }

   return indatabox;
}

