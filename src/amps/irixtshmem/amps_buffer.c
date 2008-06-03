/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

void amps_FreeBufferFreeList()
{
   amps_Buffer *ptr;

   while( ptr = amps_BufferFreeList)
   {
      amps_BufferFreeList = ptr -> next;
      free(ptr);
   }
}

int amps_AddToBufferList(buf)
char *buf;
{
    amps_Buffer *ptr;

    if(ptr = amps_BufferFreeList)
	amps_BufferFreeList = ptr -> next;
    else
	ptr = (amps_Buffer *)calloc(1, sizeof(amps_Buffer));

	    
    ptr -> buffer = buf;		
    ptr -> next = NULL;
    ptr -> prev = amps_BufferListEnd;

    if(amps_BufferListEnd)
       amps_BufferListEnd -> next = ptr;

    amps_BufferListEnd = ptr;
    
    if(!amps_BufferList)
       amps_BufferList = ptr;

    return 0;
}

char *amps_CheckBufferList(src)
int src;
{
   amps_Buffer *ptr;

   ptr = amps_BufferList;
   while(ptr)
   {
      if( *(int *)(ptr -> buffer) == src)
      {
	 /* remove the found buffer from the list */
	 if(ptr -> prev == NULL)
	    /* start of list */
	    amps_BufferList = ptr -> next;
	 else
	    ptr -> prev -> next = ptr -> next;

	 if(ptr -> next == NULL)
	    amps_BufferListEnd = ptr->prev;
	 else
	    ptr -> next -> prev = ptr -> prev;
	    
	 /* add it to the free list */
	 ptr -> next = amps_BufferFreeList;
	 amps_BufferFreeList = ptr;
	 
	 /* return the buffer that was found */
	 return (ptr -> buffer) + sizeof(double);
      }
      ptr = ptr -> next;
   }
   /* did not find on list */
   return NULL;
}
