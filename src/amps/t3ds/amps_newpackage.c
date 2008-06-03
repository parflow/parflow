/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include "amps.h"


amps_Package amps_NewPackage(amps_Comm comm,
			     int num_send,
			     int *dest,
			     amps_Invoice *send_invoices,
			     int num_recv,
			     int *src,
			     amps_Invoice *recv_invoices)
{
   amps_Package package;

   amps_PackageItem *items;
   amps_DstInfo *src_info;
   amps_InvoiceEntry *ptr;
   int num_package_items;
   int i,j;
   int dim;
   int item;
   int num;
   int num_remaining;

   package = (amps_Package)malloc(sizeof(amps_PackageStruct));
      
   package -> num_send = num_send;
   package -> dest = dest;
   package -> send_invoices = send_invoices;

   package -> num_recv = num_recv;
   package -> src = src;
   package -> recv_invoices = recv_invoices;

   /* For each of the destinations pack up and send the package inforamation 
      (ie locations, lengths and strides ) */

   if(num_recv)
     {
       for(i = 0; i < num_recv; i++)
	 {
	   /* allocate an array to send */
	   num_package_items = amps_num_package_items(recv_invoices[i]);
	 
	   item = 0;
	   items = malloc(sizeof(amps_PackageItem)*num_package_items);
	 
	   /* Pack up invoice */
	   ptr = recv_invoices[i] -> list;
	 
	   while(ptr != NULL)
	     {
	    
	       if( ptr -> type > AMPS_INVOICE_LAST_CTYPE)
		 {
		   items[item].type = ptr -> type;
	       
		   items[item].data =  
		     (ptr -> data_type == AMPS_INVOICE_POINTER) ?
		       *((char **)(ptr -> data)) : ptr -> data;
	    
		   /* Store the dim of the vector */
		   dim = items[item].dim = 
		     ( ptr -> dim_type == AMPS_INVOICE_POINTER) ?
		       *(ptr -> ptr_dim) : ptr -> dim;
	       
		   /* Pack the vector len and strides into following package
		      items */
		   for(j = 0; j < dim; j++)
		     {
		       items[item].len  = ptr -> ptr_len[j];
		       items[item].stride = ptr -> ptr_stride[j];
		       
		       item++;
		     }
		 }
	       else
		 {
		   
		   items[item].type = ptr -> type;
		   
		   items[item].data =  
		     (ptr -> data_type == AMPS_INVOICE_POINTER) ?
		       *((char **)(ptr -> data)) : ptr -> data;
	       
		   items[item].len = 
		     (ptr -> len_type == AMPS_INVOICE_POINTER) ? 
		       *(ptr -> ptr_len) : ptr ->len;
	       
		   items[item].stride = 
		     (ptr -> stride_type == AMPS_INVOICE_POINTER) ?
		       *(ptr -> ptr_stride) :  ptr -> stride;
	       
		   item++;
		 }	       
	       
	       ptr = ptr -> next;
	     }

	   /* send the package items to the dest */
	   pvm_initsend( AMPS_ENCODING);
	   pvm_pkbyte((char*)items, 
		      sizeof(amps_PackageItem)*num_package_items, 1);
	   pvm_send(src[i], amps_ExchangeTag);

	   free(items);
	 }
     }

   if(num_send)
     {
       package -> recv_info = malloc(num_send* sizeof(amps_DstInfo));
       
       /* For each of the src recv the package information */

       for(i = 0; i < num_send; i++)
	 {
	   pvm_recv(dest[i], amps_ExchangeTag);
	   num_package_items = amps_num_package_items(send_invoices[i]);
	   
	   package -> recv_info[i].items = malloc(sizeof(amps_PackageItem)
						  *num_package_items);
       
	   pvm_upkbyte((char*)package -> recv_info[i].items,
		       sizeof(amps_PackageItem)*num_package_items, 1);
	 }
     }


#if 0
       num_remaining = num_send;
       while(num_remaining)
	 {
	   for(i = 0; i < num_send; i++)
	     {
	       if( !package -> recv_info[i].items 
		  && pvm_nrecv(dest[i], amps_ExchangeTag))
		 {

		   num_package_items = amps_num_package_items(send_invoices[i]);
	   
		   package -> recv_info[i].items = malloc(sizeof(amps_PackageItem)
							  *num_package_items);
		   
		   pvm_upkbyte((char*)package -> recv_info[i].items,
			      sizeof(amps_PackageItem)*num_package_items, 1);
		   
		   num_remaining--;
		 }
	     }
	   
	 }

     }
#endif
   
   return package;
 }


void amps_FreePackage(amps_Package package)
{
  int i;
  
  if(package)
    {
      if(i = package -> num_send)
	{
	  while(i--)
	      free(package -> recv_info[i].items);
	  free(package -> recv_info);
	}
      free(package);
    }
}

