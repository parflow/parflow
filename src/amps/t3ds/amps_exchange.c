/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"

#define AMPS_COPY(type, src, len, src_stride, dest, dest_stride) \
{ \
     type *ptr_src, *ptr_dest; \
     if( ((src_stride) == 1) && ((dest_stride == 1))) \
	bcopy((src), (dest), (len)*sizeof(type)); \
     else \
	for(ptr_src = (type*)(src), (ptr_dest) = (type*)(dest); \
	    (ptr_dest) < (type*)(dest) + (len)*(dest_stride); \
	    (ptr_src) += (src_stride), (ptr_dest) += (dest_stride)) \
		    *(ptr_dest) = *(ptr_src); \
} 


void amps_vector_copy(type, items, dim, ptr_src, len, src_stride, dst, ptr_dst)
int type;
amps_PackageItem *items;
int dim;
char **ptr_src;
int *len;
int *src_stride;
int dst;
char **ptr_dst;

{
   int i;
   int dst_stride;

   dst_stride = items[dim].stride;   

   if(dim == 0)
   {
      if(type == AMPS_INVOICE_DOUBLE_CTYPE)
      {
	if(((*src_stride) == 1) && (dst_stride == 1))
	 {
	   shmem_put((long *)*ptr_dst, (long *)*ptr_src, *len, dst);

	   *(double **)ptr_src += (*len)-1;
	   *(double **)ptr_dst += (*len)-1;
	 }
	 else 
	 {
	   shmem_iput((long *)*ptr_dst, (long*)*ptr_src, 
		     dst_stride, *src_stride, *len, dst);

	   *(double **)ptr_src += ((*len)-1) * (*src_stride);
	   *(double **)ptr_dst += ((*len)-1) * dst_stride;
	 } 
      }
      else
	 printf("AMPS Error: invalid vector type\n");
   }
   else
   {
      for(i = 0; i < len[dim]-1; i++)
      {
      amps_vector_copy(type, 
		       items, 
		       dim-1, 
		       ptr_src, 
		       len, 
		       src_stride, 
		       dst, 
		       ptr_dst);

	 *(double **)ptr_src += src_stride[dim];
	 *(double **)ptr_dst += dst_stride;
      }

      amps_vector_copy(type, 
		       items, 
		       dim-1, 
		       ptr_src, 
		       len, 
		       src_stride, 
		       dst, 
		       ptr_dst);
   }
}

void _amps_wait_exchange(amps_Handle handle)
{
   amps_Package package = handle -> package;
   amps_InvoiceEntry *ptr;   
   amps_PackageItem *items;

   char *src;
   int src_len, src_stride;
   char *dst;
   int dst_stride;

   int i, j;
   int item;

   /* exchange here so things have greater likely good of being in cache */
   for(i = 0; i < package -> num_send; i++)
   {

      items = package -> recv_info[i].items;
      ptr = package -> send_invoices[i] -> list;      

      item = 0;
      /* For each of the src copy the data the package information */
      for(j = 0; j < package -> send_invoices[i] -> num; j++, ptr = ptr -> next)
	{
	  if(items[item].type > AMPS_INVOICE_LAST_CTYPE )
	    {
	      
	      dst = items[item].data;
	      src = ptr -> data;

	      amps_vector_copy(items[item].type-AMPS_INVOICE_LAST_CTYPE, 
			       &items[item], 
			       items[item].dim-1, 
			       &src, 
			       ptr -> ptr_len, 
			       ptr -> ptr_stride,
			       package -> dest[i],
			       &dst);

	    item += items[item].dim;
	 }
	 else
	 {
	    src = items[item].data;
	    
	    src_len = items[item].len;
	    src_stride = items[item].stride;
	    
	    dst = ptr -> data;
	    
	    dst_stride = (ptr -> stride_type == AMPS_INVOICE_POINTER) ? 
	       *(ptr -> ptr_stride) : ptr -> stride;

	    printf("AMPS_ERROR: not supported\n");
	    exit(1);

#if 0
	    
	    switch(items[item].type)
	    {
	    case AMPS_INVOICE_CHAR_CTYPE:
	       AMPS_COPY(char, src, src_len, src_stride, dst, dst_stride);
	       break;
	    case AMPS_INVOICE_SHORT_CTYPE:
	       AMPS_COPY(short, src, src_len, src_stride, dst, dst_stride);
	       break;
	    case AMPS_INVOICE_INT_CTYPE:
	       AMPS_COPY(int, src, src_len, src_stride, dst, dst_stride);
	       break;
	    case AMPS_INVOICE_LONG_CTYPE:
	       AMPS_COPY(long, src, src_len, src_stride, dst, dst_stride);
	       break;
	    case AMPS_INVOICE_FLOAT_CTYPE:
	       AMPS_COPY(float, src, src_len, src_stride, dst, dst_stride);
	       break;
	    case AMPS_INVOICE_DOUBLE_CTYPE:
	       AMPS_COPY(double, src, src_len, src_stride, dst, dst_stride);
	       break;
	    }
#endif

	    item++;
	 }
      }
   }

   shmem_udcflush();

   /* Need to sync here so we know everyone is done */
   amps_Sync(amps_CommWorld);
}

amps_Handle amps_IExchangePackage(amps_Package package)
{

  barrier();

  return( amps_NewHandle(NULL, 0, NULL, package));
}

