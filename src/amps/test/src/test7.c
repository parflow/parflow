/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/*
   This test prints out information size and rank information on all
   the nodes and then exits 
*/

#include <stdio.h>
#include "amps.h"

int main (argc, argv)
int argc;
char *argv[];
{
   int   num;
   int   me;

   int result = 0;

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      exit(1);
   }
   
   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);
   
   amps_Printf("Node %d: number procs = %d\n", me, num);
      
   amps_Finalize();

   return result;
}
