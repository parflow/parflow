/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

/******************************************************************************
 *
 * Various general routines used within ParFlow.
 *
 *****************************************************************************/

#include <math.h>
#include "parflow.h"

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif

amps_ThreadLocalDcl(int, s_max_memory);

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



/*
 *************************************************************************
 *                                                                       
 * Records memory usage
 *                                                                       
 *************************************************************************
*/

void recordMemoryInfo() 
{
#ifdef HAVE_MALLINFO
   /*
    * Access information from mallinfo
    */   
   struct mallinfo my_mallinfo = mallinfo();

   /* Get all the memory currently allocated to user by malloc, etc. */
   int used_mem = my_mallinfo.uordblks;

   /* Record high-water mark for memory used. */
   if(amps_ThreadLocal(s_max_memory) < used_mem) 
   {
      amps_ThreadLocal(s_max_memory) = used_mem;
   }
#endif
}


/*
*************************************************************************
*                                                                       *
* Prints maximum memory used (i.e. high-water mark).  The max is        *
* determined each time the "printMemoryInfo" or "recordMemoryInfo"      *
* functions are called.                                                 *
*                                                                       *
*************************************************************************
*/
void printMaxMemory(FILE *log_file)  
{
#ifdef HAVE_MALLINFO



   /*
    * Step through all nodes (>0) and send max memory to processor 0,
    * which subsequently writes it out.
    */
   int maxmem = 0;

   recordMemoryInfo();

   amps_Invoice invoice = amps_NewInvoice("%i", &maxmem);

   if (amps_Rank(amps_CommWorld) != 0) {
	 maxmem = (int)amps_ThreadLocal(s_max_memory);
	 amps_Send(amps_CommWorld, 0, invoice);
   } else {

      int p = 0;

      if(log_file) {
	 fprintf(log_file, 
		 "Maximum memory used on processor %d : %d MB\n", 
		 p, 
		 (int)amps_ThreadLocal(s_max_memory)/(1024*1024) );
      }
	 
      for (p = 1; p < amps_Size(amps_CommWorld); p++) {
	 amps_Recv(amps_CommWorld, p, invoice);
	 if(log_file) {
	    fprintf(log_file, 
		    "Maximum memory used on processor %d : %d MB\n", 
		    p, 
		    maxmem/(1024*1024) );
	 }
      }
      
   }
   
   amps_FreeInvoice(invoice);

#endif
}


/*
 *************************************************************************
 *                                                                       *
 * Prints memory usage to specified output stream.  Each time this       *
 * method is called, it prints in the format:                            *
 *                                                                       *
 *    253.0MB (265334688) in 615 allocs, 253.9MB reserved (871952 unused)*
 *                                                                       *
 * where                                                                 *
 *                                                                       *
 *    253.0MB is how many megabytes your current allocation has malloced.* 
 *    2653346688 is the precise size (in bytes) of your current alloc.   *
 *    615 is the number of items allocated with malloc.                  *
 *    253.9MB is the current memory reserved by the system for mallocs.  *
 *    871952 is the bytes currently not used in this reserved memory.    *
 *                                                                       *
 *************************************************************************
*/
void printMemoryInfo(FILE *log_file) 
{
   
#ifdef HAVE_MALLINFO

   /* Get malloc info structure */
   struct mallinfo my_mallinfo = mallinfo();
   
   /* Get total memory reserved by the system for malloc currently*/
   int reserved_mem = my_mallinfo.arena;
   
   /* Get all the memory currently allocated to user by malloc, etc. */
#if 0
   int used_mem = my_mallinfo.hblkhd + my_mallinfo.usmblks +
      my_mallinfo.uordblks;
#endif
   int used_mem = my_mallinfo.uordblks;
   
   /* Get memory not currently allocated to user but malloc controls */
   int free_mem = my_mallinfo.fsmblks + my_mallinfo.fordblks;
   
   /* Get number of items currently allocated */
   int number_allocated = my_mallinfo.ordblks + my_mallinfo.smblks;

   
   /* Record high-water mark for memory used. */
   if(amps_ThreadLocal(s_max_memory) < used_mem) 
   {
      amps_ThreadLocal(s_max_memory) = used_mem;
   }
   
   /* Print out concise malloc info line */
   fprintf(log_file, 
	   "Memory in use : %d MB in %d allocs, %d MB reserved ( %d unused)\n",
	   used_mem/(1024*1024), 
	   number_allocated, 
	   reserved_mem/(1024*1024), 
	   free_mem/(1024*1024));
#endif
}
