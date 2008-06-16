/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifdef AMPS_MALLOC_DEBUG
#include <gmalloc.h>
#endif

#include <stdio.h>
#include <sys/times.h>
#include <pvm3.h>


#define amps_ThreadLocal(arg) arg
#define amps_ThreadLocalDcl(type, arg) type arg

#define FALSE 0
#define TRUE  1

#define amps_Max 1
#define amps_Min 2
#define amps_Add 3

#define amps_MsgTag 7

typedef char *amps_Comm;
typedef FILE *amps_File;

extern amps_Comm amps_CommWorld;

extern int amps_tid;
extern int *amps_tids;
extern int amps_rank;
extern int amps_size;

#define AMPS_ENCODING PvmDataDefault

/* These are the built-in types that are supported */

#define AMPS_INVOICE_CHAR_CTYPE                1
#define AMPS_INVOICE_SHORT_CTYPE               2
#define AMPS_INVOICE_INT_CTYPE                 3
#define AMPS_INVOICE_LONG_CTYPE                4
#define AMPS_INVOICE_DOUBLE_CTYPE              5
#define AMPS_INVOICE_FLOAT_CTYPE               6
#define AMPS_INVOICE_LAST_CTYPE                7

/* Flags for use with user-defined flag                                      */
/* ?????? following is very confusing rename them SGS */
#define AMPS_INVOICE_OVERLAY                   4
#define AMPS_INVOICE_NON_OVERLAYED             8

/* Flags for use with Pfmp_Invoice flag field                                 */
#define AMPS_INVOICE_USER_TYPE                 1

/* Flags for use with data types */
#define AMPS_INVOICE_CONSTANT 0
#define AMPS_INVOICE_POINTER 1
#define AMPS_INVOICE_DATA_POINTER 2

#define AMPS_INVOICE_ALLOCATED 1
#define AMPS_INVOICE_OVERLAYED 2


/* This structure is used to keep track of the entries in an invoice         */
typedef struct amps_invoice
{
    long   flags;     /* some flags for this invoice */
    
    long   combuf_flags; /* flags indicating state of the communications
			    buffer */

    struct amps_invoice_entry *list;
    struct amps_invoice_entry *end_list;
    int num;          /* number of items in the list                        */

    amps_Comm comm;
    
} amps_InvoiceStruct;

typedef amps_InvoiceStruct *amps_Invoice;
/* Each entry in the invoice has one of these                                */
typedef struct amps_invoice_entry
{
    int     type;         /* type that this invoice points to */
    
    long    flags;        /* flags indicating state of the invoice           */
    
    int     data_type;    /* what type of pointer do we have                 */
    long    data_flags;   /* flags indicating state of the data pointer      */
    void   *data;

    void   *extra;
    
    int     len_type;
    int     len;
    int    *ptr_len;
    
    int     stride_type;
    int     stride;
    int     *ptr_stride;

    int     dim_type;
    int     dim;
    int     *ptr_dim;
    
    int     ignore;      /* do we ignore this invoice?                       */
    
    struct amps_invoice_entry *next;
    
} amps_InvoiceEntry;

/*===========================================================================*/
/* Package structure is used by the Exchange functions.  Contains several    */
/* Invoices plus the src or dest rank.                                       */
/*===========================================================================*/


typedef struct
{
   int            num_send;
   int           *dest;
   amps_Invoice  *send_invoices;

   int            num_recv;
   int           *src;
   amps_Invoice  *recv_invoices;

   int            recv_remaining;
   struct amps_HandleObject **recv_handles;

} amps_PackageStruct;

typedef amps_PackageStruct *amps_Package;

typedef struct
{
   int type;
   amps_Comm comm;
   int id;
   amps_Invoice invoice;
   amps_Package package;

} amps_HandleObject;

#ifdef AMPS_BSD_TIME
#define AMPS_TICKS_PER_SEC 10000
#endif

typedef long amps_Clock_t;

typedef clock_t amps_CPUClock_t;
extern long AMPS_CPU_TICKS_PER_SEC;

typedef amps_HandleObject *amps_Handle;

#include "amps_proto.h"

/*****************************************************************************
 *
 *   PACKING structures and defines
 *
 *****************************************************************************/
 
#define AMPS_PACKED 2

#define AMPS_IGNORE  -1

#define PACK_HOST_TYPE 1
#define PACK_NO_CONVERT_TYPE 2

#define AMPS_ALIGN(type, dest) \
    ((sizeof(type) - \
      ((unsigned long)(dest) % sizeof(type))) \
     % sizeof(type));

#define AMPS_SIZEOF(type, len, stride) \
    ((sizeof(type)*(len)*(stride)))

/*---------------------------------------------------------------------------*/
/* Macros for Invoice creation and deletion.                                 */
/*---------------------------------------------------------------------------*/

#define amps_append_invoice amps_new_invoice

/*---------------------------------------------------------------------------*/
/* Internal macros used to clear buffer and letter spaces.                   */
/*---------------------------------------------------------------------------*/
#if SGS
#define AMPS_CLEAR_INVOICE(invoice) \
    { \
	  amps_ClearInvoice(invoice); \
    }

#define AMPS_PACK_FREE_LETTER(comm, invoice,amps_letter) \
    if( (invoice) -> combuf_flags & AMPS_INVOICE_OVERLAYED) \
        (invoice) -> combuf_flags |= AMPS_INVOICE_ALLOCATED; \
    else \
    { \
        (invoice) -> combuf_flags &= ~AMPS_INVOICE_ALLOCATED; \
	pvm_freebuf(amps_letter); \
    } 

#endif

#define amps_Rank(comm) amps_rank
#define amps_Size(comm) amps_size

#define amps_gettid(comm, rank) amps_tids[(rank)]

/* In PVM we always send */
#define amps_ISend(comm, dest, invoice) 0, amps_Send((comm), (dest), (invoice))

#define amps_FreeHandle(handle) free((handle));

#define amps_Sync(comm) pvm_barrier((comm), amps_size)

#define amps_Exit(code) exit(code)

#define amps_Fclose(file)  fclose((file))
#define amps_Fprintf fprintf
#define amps_Fscanf fscanf

#define amps_FFclose(file) fclose((file))


/******************************************************************************
 * Read and Write routines to write to files in XDR format.
 *****************************************************************************/


#define amps_SizeofChar sizeof(char)
#define amps_SizeofShort sizeof(short)
#define amps_SizeofInt sizeof(int)
#define amps_SizeofLong sizeof(long)
#define amps_SizeofFloat sizeof(float)
#define amps_SizeofDouble sizeof(double)

/*---------------------------------------------------------------------------*/
/* The following routines are used to actually write data to a file.         */
/* We use XDR like representation for all values written.                    */
/*---------------------------------------------------------------------------*/

#define amps_SizeofChar sizeof(char)
#define amps_SizeofShort sizeof(short)
#define amps_SizeofInt sizeof(int)
#define amps_SizeofLong sizeof(long)
#define amps_SizeofFloat sizeof(float)
#define amps_SizeofDouble sizeof(double)

#define amps_WriteChar(file, ptr, len) \
    fwrite( (ptr), sizeof(char), (len), (FILE *)(file) )

#define amps_WriteShort(file, ptr, len) \
    fwrite( (ptr), sizeof(short), (len), (FILE *)(file) )

#define amps_WriteLong(file, ptr, len) \
    fwrite( (ptr), sizeof(long), (len), (FILE *)(file) )

#define amps_WriteFloat(file, ptr, len) \
    fwrite( (ptr), sizeof(float), (len), (FILE *)(file) )

#ifndef AMPS_BYTE_SWAP

#define amps_WriteInt(file, ptr, len) \
    fwrite( (ptr), sizeof(int), (len), (FILE *)(file) )

#define amps_WriteDouble(file, ptr, len) \
    fwrite( (ptr), sizeof(double), (len), (FILE *)(file) )

#endif


#define amps_ReadChar(file, ptr, len) \
    fread( (ptr), sizeof(char), (len), (FILE *)(file) )

#define amps_ReadShort(file, ptr, len) \
    fread( (ptr), sizeof(short), (len), (FILE *)(file) )

#define amps_ReadLong(file, ptr, len) \
    fread( (ptr), sizeof(long), (len), (FILE *)(file) )

#define amps_ReadFloat(file, ptr, len) \
    fread( (ptr), sizeof(float), (len), (FILE *)(file) )

#ifndef AMPS_BYTE_SWAP

#define amps_ReadInt(file, ptr, len) \
    fread( (ptr), sizeof(int), (len), (FILE *)(file) )

#define amps_ReadDouble(file, ptr, len) \
    fread( (ptr), sizeof(double), (len), (FILE *)(file) )

#endif

#if SGS
#define amps_free(comm, combuf) pvm_freebuf((combuf))
#endif

#define amps_Error(name, type, comment, operation) 

#ifdef AMPS_MEMORY_ALLOC_CHECK
 
#define amps_TAlloc(type, count) \
{ \
     (type *) ptr; \
     if ( (ptr = (type *) malloc((unsigned int)(sizeof(type) * (count)))) \
         == NULL) \
     amps_Printf("Error: out of memory in <%s> at line %d\n", \
                 __FILE__, __LINE__); \
     ptr; \
} 
 
#define amps_CTAlloc(type, count) \
{ \
     (type *) ptr; \
     if ( (ptr = (type *) calloc((unsigned int)(count), (unsigned int)sizeof(type))) == NULL ) \
     amps_Printf("Error: out of memory in <%s> at line %d\n", \
                 __FILE__, __LINE__); \
     ptr; \
} 
 
/* note: the `else' is required to guarantee termination of the `if' */
#define amps_TFree(ptr) if (ptr) free(ptr); else
 
/*--------------------------------------
 * Do not check memory allocation
 *--------------------------------------*/
 
#else
 
#define amps_Talloc(type, count) \
((count) ? (type *) malloc((unsigned int)(sizeof(type) * (count))) : NULL)
 
#define amps_CTAlloc(type, count) \
((count) ? (type *) calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)
 
/* note: the `else' is required to guarantee termination of the `if' */
#define amps_TFree(ptr) if (ptr) free(ptr); else
 
#endif
