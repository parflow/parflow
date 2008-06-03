/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

int main (argc, argv)
int argc;
char *argv[];
{
   int   num;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      exit(1);
   }
   
   amps_Finalize();

   return result;
}
