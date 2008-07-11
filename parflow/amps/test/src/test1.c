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
   the nodes and then amps_Exits 
*/

#include <stdio.h>
#include "amps.h"

int main (argc, argv)
int argc;
char *argv[];
{
   int   num;
   int   me;
   int   i;

   int result = 0;

   char *ptr;

   /* To make sure that malloc checking is on */

   if (amps_Init(&argc, &argv))
   {
      amps_Printf("Error amps_Init\n");
      amps_Exit(1);
   }

   ptr = malloc(20);

   num = amps_Size(amps_CommWorld);

   me = amps_Rank(amps_CommWorld);
   
   amps_Printf("Node %d: number procs = %d\n", me, num);

   for(i = 1; i < argc; i++)
   {
      amps_Printf("arg[%d] = %s\n", i, argv[i]);
   }

   amps_Finalize();

   return result;
}

