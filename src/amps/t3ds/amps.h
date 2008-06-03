/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <pvm3.h>
#include <mpp/shmem.h>

#ifdef AMPS_MALLOC_DEBUG
#include <gmalloc.h>
#endif

#define amps_ThreadLocal(arg) arg
#define amps_ThreadLocalDcl(type, arg) type arg

#define FALSE 0
#define TRUE  1

#define amps_Max 1
#define amps_Min 2
#define amps_Add 3

#define amps_MsgTag 7
#define amps_ExchangeTag 17

#define amps_ReduceBufSize 20

typedef char *amps_Comm;
typedef FILE *amps_File;

#define amps_CommWorld NULL

extern int amps_rank;
extern void *redWrk;
extern void *redWrk_buf;
extern long redSync[_SHMEM_REDUCE_SYNC_SIZE];
extern long barSync[_SHMEM_BARRIER_SYNC_SIZE];
extern long bcaSync[_SHMEM_BCAST_SYNC_SIZE];

extern int amps_size;

#define AMPS_ENCODING PvmDataRaw

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


#ifdef CRAY_TIME
#define amps_Clock() rtclock()
#define amps_CPUClock() cpused()
#define AMPS_TICKS_PER_SEC 1.5E8
#define AMPS_CPU_TICKS_PER_SEC 1.5E8
#endif

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
    int    *ptr_dim;

    
    int     ignore;      /* do we ignore this invoice?                       */
    
    struct amps_invoice_entry *next;
    
} amps_InvoiceEntry;

/*===========================================================================*/
/* Package structure is used by the Exchange functions.  Contains several    */
/* Invoices plus the src or dest rank.                                       */
/*===========================================================================*/
 
typedef struct _amps_packageitem
{
   int type;   
 
   void *data;
 
   int len;
   int stride;
 
   int dim;
 
} amps_PackageItem;

typedef struct _amps_srcinfo
{

   amps_PackageItem *items;   

} amps_DstInfo;

 
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

   amps_DstInfo  *recv_info;

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
 
typedef amps_HandleObject *amps_Handle;

typedef long amps_Clock_t;

typedef clock_t amps_CPUClock_t;

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

#define amps_Rank(comm) amps_rank
#define amps_Size(comm) amps_size

/* In PVM we always send */
#define amps_ISend(comm, dest, invoice) 0, amps_Send((comm), (dest), (invoice))

#define amps_FreeHandle(handle) free((handle));

#define amps_Sync(comm) barrier()

#define amps_Exit(code) exit(code)

#define amps_Fclose(file)  fclose((file))
#define amps_Fprintf fprintf
#define amps_Fscanf fscanf

#define amps_FFclose(file)  fclose((file))

/******************************************************************************
 * Read and Write routines to write to files in XDR format.
 *****************************************************************************/

#define amps_SizeofChar sizeof(char)
#define amps_SizeofShort sizeof(short)
/* We are writing out in XDR format, CRAY uses 8 byte ints */
#define amps_SizeofInt 4
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

#define amps_WriteDouble(file, ptr, len) \
    fwrite( (ptr), sizeof(double), (len), (FILE *)(file) )

#define amps_ReadChar(file, ptr, len) \
    fread( (ptr), sizeof(char), (len), (FILE *)(file) )

#define amps_ReadShort(file, ptr, len) \
    fread( (ptr), sizeof(short), (len), (FILE *)(file) )

#define amps_ReadLong(file, ptr, len) \
    fread( (ptr), sizeof(long), (len), (FILE *)(file) )

#define amps_ReadFloat(file, ptr, len) \
    fread( (ptr), sizeof(float), (len), (FILE *)(file) )

#define amps_ReadDouble(file, ptr, len) \
    fread( (ptr), sizeof(double), (len), (FILE *)(file) )

#define amps_Error(name, type, comment, operation) 

#define amps_TAlloc(type, count) \
((count) ? (type *)amps_malloc(sizeof(type) * (count)) : NULL)
 
#define amps_CTAlloc(type, count) \
((count) ? (type *)amps_calloc((count), sizeof(type)) : NULL)

#define amps_TFree(ptr)
