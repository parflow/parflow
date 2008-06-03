/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <string.h>
#include <stdarg.h>

#include "amps.h"

	
int amps_unpack(comm, inv, buffer, buf_size)
amps_Comm comm;
amps_Invoice inv;
char *buffer;
int buf_size;
{
   amps_InvoiceEntry *ptr;
   int len, stride;
   int  malloced = FALSE;
   int size;
   char *data;
   char *temp_pos;
   int dim;

   MPI_Datatype mpi_type;
   MPI_Datatype *base_type;
   MPI_Datatype *new_type;
   MPI_Datatype *temp_type;

   int element_size;
   int position;

   int base_size;

   int i;
   
   /* we are unpacking so signal this operation */
   
   inv -> flags &= ~AMPS_PACKED;
   
   /* for each entry in the invoice pack that entry into the letter         */
   ptr = inv -> list;
   position = 0;
   while(ptr != NULL)
   {
      /* invoke the packing convert out for the entry */
      /* if user then call user ones */
      /* else switch on builtin type */
      if(ptr -> len_type == AMPS_INVOICE_POINTER)
	 len = *(ptr -> ptr_len);
      else
	 len = ptr ->len;
      
      if(ptr -> stride_type == AMPS_INVOICE_POINTER)
	 stride = *(ptr -> ptr_stride);
      else
	 stride = ptr -> stride;
      
      switch(ptr->type)
      {
      case AMPS_INVOICE_CHAR_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(char)
						     *len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_BYTE, &mpi_type);

	       MPI_Type_commit(&mpi_type);

	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);

	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_BYTE, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      case AMPS_INVOICE_SHORT_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(short)* 
						  len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_SHORT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_SHORT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      case AMPS_INVOICE_INT_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(int)* 
						  len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_INT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_INT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      case AMPS_INVOICE_LONG_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(long)*
						  len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_LONG, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_LONG, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      case AMPS_INVOICE_FLOAT_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(float)*
						  len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_FLOAT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_FLOAT, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      case AMPS_INVOICE_DOUBLE_CTYPE:
	 if(!ptr->ignore)
	 {
	    if( ptr -> data_type == AMPS_INVOICE_POINTER)
	    {
	       *((void **)(ptr -> data)) = malloc(sizeof(double)*
						  len*stride);
	       malloced = TRUE;

	       MPI_Type_vector(len, 1, stride, MPI_DOUBLE, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  *((void **)(ptr -> data)), 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	    else
	    {
	       MPI_Type_vector(len, 1, stride, MPI_DOUBLE, &mpi_type);

	       MPI_Type_commit(&mpi_type);
	       MPI_Unpack(buffer, buf_size, &position, 
			  ptr -> data, 1, mpi_type, comm);
	       MPI_Type_free(&mpi_type);
	    }
	 }
	 break;
      default:
	 dim = ( ptr -> dim_type == AMPS_INVOICE_POINTER) ?
	     *(ptr -> ptr_dim) : ptr -> dim;

	 size = amps_vector_sizeof_local( comm, 
				   ptr -> type - AMPS_INVOICE_LAST_CTYPE,
				   NULL, &temp_pos, dim, 
				   ptr -> ptr_len, ptr -> ptr_stride);

	 if( ptr -> data_type == AMPS_INVOICE_POINTER )
	    data = *(char **)(ptr -> data) = (char *)malloc(size);
	 else 
	    data = ptr -> data;

	 base_type = calloc(1, sizeof(MPI_Datatype));
	 new_type = calloc(1, sizeof(MPI_Datatype));

	 len = ptr -> ptr_len[0];
	 stride = ptr -> ptr_stride[0];

	 switch(ptr -> type - AMPS_INVOICE_LAST_CTYPE)
	 {
	 case AMPS_INVOICE_CHAR_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_BYTE, base_type);
	       element_size = sizeof(char);
	    }
	    break;
	 case AMPS_INVOICE_SHORT_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_SHORT, base_type);
	       element_size = sizeof(short);
	    }
	    break;
	 case AMPS_INVOICE_INT_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_INT, base_type);
	       element_size = sizeof(int);
	    }
	    break;
	 case AMPS_INVOICE_LONG_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_LONG, base_type);
	       element_size = sizeof(long);
	    }
	    break;
	 case AMPS_INVOICE_FLOAT_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_FLOAT, base_type);
	       element_size = sizeof(float);
	    }
	    break;
	 case AMPS_INVOICE_DOUBLE_CTYPE:
	    if(!ptr->ignore)
	    {
	       MPI_Type_vector(len, 1, stride, MPI_DOUBLE, base_type);
	       element_size = sizeof(double);
	    }
	    break;
	 }

	 base_size = element_size*( len + (len - 1) * (stride-1));
	 
	 for(i = 1; i < dim; i++)
	 {
	    MPI_Type_hvector(ptr -> ptr_len[i], 1, 
				base_size + 
				(ptr -> ptr_stride[i]-1) * element_size,
			     *base_type, new_type);

	    base_size = base_size*ptr -> ptr_len[i]
	       + (ptr -> ptr_stride[i]-1)*(ptr->ptr_len[i] - 1)*element_size;
	    MPI_Type_free(base_type);
	    temp_type = base_type;
	    base_type = new_type;
	    new_type = temp_type;
	 }
	 
	 MPI_Type_commit(base_type);

	 MPI_Unpack(buffer, size, &position, data, 1, *base_type, comm);

	 MPI_Type_free(base_type);

	 free(base_type);
	 free(new_type);
      }
      ptr = ptr -> next;
   }  
   
   if(malloced)
   {
      inv -> combuf_flags |= AMPS_INVOICE_OVERLAYED;
      inv -> combuf = buffer;
      inv -> comm = comm;
   }
   return 0;
}

