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
 * Various general routines used within ParFlow.
 *
 *****************************************************************************/

#include <math.h>
#include "parflow.h"


#ifdef PF_MEMORY_ALLOC_CHECK

#define MEM_CHECK_SIZE 100000

/*--------------------------------------------------------------------------
 * malloc_chk
 *--------------------------------------------------------------------------*/

char  *malloc_chk(size, file, line)
int    size;
char  *file;
int    line;
{
   char  *ptr;


   if (size)
   {
      ptr = malloc(size);

      if (ptr == NULL)
	 amps_Printf("Error: out of memory in %s at line %d\n", file, line);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}


/*--------------------------------------------------------------------------
 * calloc_chk
 *--------------------------------------------------------------------------*/

char  *calloc_chk(count, elt_size, file, line)
int    count;
int    elt_size;
char  *file;
int    line;
{
   char  *ptr;
   int    size = count*elt_size;


   if (size)
   {
      ptr = calloc(count, elt_size);

      if (ptr == NULL)
	 amps_Printf("Error: out of memory in %s at line %d\n", file, line);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

#endif


/*--------------------------------------------------------------------------
 * Exp2:
 *   Return the integer e satisfying p = 2^e, where p >= 1 is an integer.
 *   If no such integer e exists, return -1.
 *--------------------------------------------------------------------------*/

int  Exp2(p)
int  p;
{
   int  e = 0;


   while (p > 1)
   {
      if (p % 2)
         return -1;
      else
      {
         e += 1;
         p /= 2;
      }
   }

   return  e;
}


