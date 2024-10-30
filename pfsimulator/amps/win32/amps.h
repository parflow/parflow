/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/
#ifndef _AMPS_HEADER
#define _AMPS_HEADER

#define new DEBUG_NEW

#include <windows.h>
#include <stdio.h>
#include <malloc.h>
#include <process.h>
#include <limits.h>


extern HANDLE *amps_sync_ready;
extern HANDLE *amps_sync_done;

extern HANDLE *locks;

extern HANDLE *sema;

extern HANDLE bcast_lock;
extern HANDLE *bcast_locks;

extern HANDLE *bcast_sema;


#define MAXPATHLEN 1024

#define AMPS_MAX_MESGS LONG_MAX

int AMPS_USERS_MAIN(int argc, char *argv []);

#define main AMPS_USERS_MAIN

#define amps_Max 1
#define amps_Min 2
#define amps_Add 3

#define AMPS_PID 0

/* These are the built-in types that are supported */

#define AMPS_INVOICE_CHAR_CTYPE                1
#define AMPS_INVOICE_SHORT_CTYPE               2
#define AMPS_INVOICE_INT_CTYPE                 3
#define AMPS_INVOICE_LONG_CTYPE                4
#define AMPS_INVOICE_DOUBLE_CTYPE              5
#define AMPS_INVOICE_FLOAT_CTYPE               6
#define AMPS_INVOICE_LAST_CTYPE                7

/* Flags for use with user-defined flag                                      */
#define AMPS_INVOICE_OVERLAY                   1

/* Flags for use with Pfmp_Invoice flag field                                 */
#define AMPS_INVOICE_USER_TYPE                 1

/* Flags for use with data types */
#define AMPS_INVOICE_CONSTANT 0
#define AMPS_INVOICE_POINTER 1
#define AMPS_INVOICE_DATA_POINTER 2

/* SGS ?????? following are misleading, OVERLAYED indicates overlayed or
 *  malloced, NON_OVERLAYED indicates all malloced non actually overlayed */
#define AMPS_INVOICE_ALLOCATED 1
#define AMPS_INVOICE_OVERLAYED 2
#define AMPS_INVOICE_NON_OVERLAYED 4

typedef int amps_Comm;
typedef FILE *amps_File;

extern amps_Comm amps_CommWorld;
extern int amps_size;
_declspec(thread) extern int amps_rank;

#define amps_Rank(comm) amps_rank

#define amps_Exit(code) exit(code)

/* In CE/RK we always send */
#define amps_ISend(comm, dest, invoice) 0, amps_Send((comm), (dest), (invoice))

#define amps_Size(comm) amps_size

#define amps_new(comm, size) ((char*)malloc((size) + sizeof(double)) + sizeof(double))
#define amps_free(comm, buf) free((char*)(buf) - sizeof(double))

#define amps_FreeHandle(handle) free((handle))

#define amps_Fclose(file)  fclose((file))
#define amps_Fprintf fprintf
#define amps_Fscanf fscanf

#define amps_FFclose(file)  fclose((file))

#define amps_Clock() GetTickCount()
#define amps_CPUClock() GetTickCount()

typedef DWORD amps_Clock_t;
typedef DWORD amps_CPUClock_t;

#define AMPS_TICKS_PER_SEC 1000

#define amps_ThreadLocal(arg) arg
#define amps_ThreadLocalDcl(type, arg) _declspec(thread) type arg


/* This structure is used to keep track of the entries in an invoice         */
typedef struct amps_invoicestruct {
  long flags;          /* some flags for this invoice */

  long combuf_flags;     /* flags indicating state of the communications
                          * buffer */
  void   *combuf;      /* pointer to the communications buffer
                        * associated with this invoice                       */

  struct amps_invoice_entry *list;
  struct amps_invoice_entry *end_list;
  int num;            /* number of items in the list                        */

  amps_Comm comm;
} amps_InvoiceStruct;

typedef amps_InvoiceStruct *amps_Invoice;
/* Each entry in the invoice has one of these                                */
typedef struct amps_invoice_entry {
  int type;               /* type that this invoice points to */

  long flags;             /* flags indicating state of the invoice           */

  int data_type;          /* what type of pointer do we have                 */
  long data_flags;        /* flags indicating state of the data pointer      */
  void   *data;

  void   *extra;

  int len_type;
  int len;
  int    *ptr_len;

  int stride_type;
  int stride;
  int    *ptr_stride;

  int dim_type;
  int dim;
  int    *ptr_dim;

  int ignore;            /* do we ignore this invoice?                       */

  struct amps_invoice_entry *next;
} amps_InvoiceEntry;

typedef struct amps_buffer {
  struct amps_buffer *next, *prev;

  char *buffer;
} amps_Buffer;

/*===========================================================================*/
/* Package structure is used by the Exchange functions.  Contains several    */
/* Invoices plus the src or dest rank.                                       */
/*===========================================================================*/

typedef struct _amps_packageitem {
  int type;

  void *data;

  int len;
  int stride;

  int dim;
} amps_PackageItem;

typedef struct _amps_srcinfo {
  HANDLE send_sema;
  HANDLE recv_sema;

  amps_PackageItem *items;
} amps_SrcInfo;

typedef struct {
  int num_send;
  int           *dest;
  amps_Invoice  *send_invoices;

  int num_recv;
  int           *src;
  amps_Invoice  *recv_invoices;

  int recv_remaining;
  struct amps_HandleObject **recv_handles;

  amps_SrcInfo  **rcv_info;
  amps_SrcInfo  **snd_info;
} amps_PackageStruct;

typedef amps_PackageStruct *amps_Package;

typedef struct {
  int type;
  amps_Comm comm;
  int id;
  amps_Invoice invoice;
  amps_Package package;
} amps_HandleObject;

typedef amps_HandleObject *amps_Handle;

#if 0
extern amps_Buffer **amps_PtrBufferList;
extern amps_Buffer **amps_PtrBufferListEnd;
extern amps_Buffer **amps_PtrBufferFreeList;
#endif

#define amps_BufferList (amps_PtrBufferList[amps_rank])
#define amps_BufferListEnd (amps_PtrBufferListEnd[amps_rank])
#define amps_BufferFreeList (amps_PtrBufferFreeList[amps_rank])

extern amps_Invoice *amps_PtrSyncInvoice;

/****************************************************************************
 *
 *   PACKING structures and defines
 *
 *****************************************************************************/

#define AMPS_PACKED 2

#define AMPS_IGNORE  -1

#define PACK_HOST_TYPE 1
#define PACK_NO_CONVERT_TYPE 2

/*---------------------------------------------------------------------------*/
/* General functions to call methods for specified type                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/* Functions to for align                                                    */
/*---------------------------------------------------------------------------*/
#define AMPS_ALIGN(type, src, dest, len, stride)       \
        ((sizeof(type) -                               \
          ((unsigned long)(dest) % sizeof(type)))      \
         % sizeof(type));

#define AMPS_CALL_CHAR_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(char, (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_SHORT_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(short, (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_INT_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(int, (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_LONG_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(long, (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_FLOAT_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(float, (_src), (_dest), (_len), (_stride))

/* SGS warning!!!!!!:  since malloc returns 4 byte alinged things we should align
 * in a similiar way */

#define AMPS_CALL_DOUBLE_ALIGN(_comm, _src, _dest, _len, _stride) \
        AMPS_ALIGN(float, (_src), (_dest), (_len), (_stride))

/*---------------------------------------------------------------------------*/
/* Functions to for sizeof                                                   */
/*---------------------------------------------------------------------------*/
#define AMPS_SIZEOF(len, stride, size) \
        ((len) * (size))

#define AMPS_CALL_CHAR_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(char))

#define AMPS_CALL_SHORT_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(short))

#define AMPS_CALL_INT_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(int))

#define AMPS_CALL_LONG_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(long))

#define AMPS_CALL_FLOAT_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(float))

#define AMPS_CALL_DOUBLE_SIZEOF(_comm, _src, _dest, _len, _stride) \
        AMPS_SIZEOF((_len), (_stride), sizeof(double))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#define AMPS_CONVERT_OUT(type, cvt, comm, src, dest, len, stride)                                               \
        {                                                                                                       \
          type *ptr_src, *ptr_dest;                                                                             \
          if ((char*)(src) != (char*)(dest))                                                                    \
          if ((stride) == 1)                                                                                    \
          memcpy((dest), (src), (len) * sizeof(type));                                                          \
          else                                                                                                  \
          for (ptr_src = (type*)(src), ptr_dest = (type*)(dest); ptr_src < (type*)(src) + (len) * (stride);     \
               ptr_src += (stride), ptr_dest++)                                                                 \
          *ptr_dest = *ptr_src;                                                                                 \
        }

#define AMPS_CALL_CHAR_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(char, ctohc, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_SHORT_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(short, ctohs, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_INT_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(int, ctohi, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_LONG_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(long, ctohl, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_FLOAT_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(float, ctohf, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_DOUBLE_OUT(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_OUT(double, ctohd, (_comm), (_src), (_dest), (_len), (_stride))



#define AMPS_CONVERT_IN(type, cvt, comm, src, dest, len, stride)                                                      \
        {                                                                                                             \
          type *ptr_src, *ptr_dest;                                                                                   \
          if ((char*)(src) != (char*)(dest))                                                                          \
          if ((stride) == 1)                                                                                          \
          memcpy((dest), (src), (len) * sizeof(type));                                                                \
          else                                                                                                        \
          for (ptr_src = (type*)(src), (ptr_dest) = (type*)(dest); (ptr_dest) < (type*)(dest) + (len) * (stride);     \
               (ptr_src)++, (ptr_dest) += (stride))                                                                   \
          *(ptr_dest) = *(ptr_src);                                                                                   \
        }


#define AMPS_CALL_CHAR_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(char, htocc, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_SHORT_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(short, htocs, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_INT_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(int, htoci, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_LONG_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(long, htocl, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_FLOAT_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(float, htocf, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CALL_DOUBLE_IN(_comm, _src, _dest, _len, _stride) \
        AMPS_CONVERT_IN(double, htocd, (_comm), (_src), (_dest), (_len), (_stride))

#define AMPS_CHECK_OVERLAY(_type, _comm) 1

#define AMPS_CHAR_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(char, _comm)

#define AMPS_SHORT_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(short, _comm)

#define AMPS_INT_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(int, _comm)

#define AMPS_LONG_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(long, _comm)

#define AMPS_FLOAT_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(float, _comm)

#define AMPS_DOUBLE_OVERLAY(_comm) \
        AMPS_CHECK_OVERLAY(double, _comm)


/*---------------------------------------------------------------------------*/
/* Macros for Invoice creation and deletion.                                 */
/*---------------------------------------------------------------------------*/

#define amps_append_invoice amps_new_invoice

/*---------------------------------------------------------------------------*/
/* Internal macros used to clear buffer and letter spaces.                   */
/*---------------------------------------------------------------------------*/

#define AMPS_CLEAR_INVOICE(invoice)                           \
        {                                                     \
          (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED; \
          amps_ClearInvoice(invoice);                         \
        }

#define AMPS_PACK_FREE_LETTER(comm, invoice, amps_letter)       \
        if ((invoice)->combuf_flags & AMPS_INVOICE_OVERLAYED)   \
        (invoice)->combuf_flags |= AMPS_INVOICE_ALLOCATED;      \
        else                                                    \
        {                                                       \
          (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;   \
          amps_free((comm), (amps_letter));                     \
        }                                                       \


/*****************************************************************************
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

#define amps_WriteChar(file, ptr, len) \
        fwrite((ptr), sizeof(char), (len), (FILE*)(file))

#define amps_WriteShort(file, ptr, len) \
        fwrite((ptr), sizeof(short), (len), (FILE*)(file))

void amps_WriteInt();

#define amps_WriteLong(file, ptr, len) \
        fwrite((ptr), sizeof(long), (len), (FILE*)(file))

void amps_WriteDouble();


#define amps_ReadChar(file, ptr, len) \
        fread((ptr), sizeof(char), (len), (FILE*)(file))

#define amps_ReadShort(file, ptr, len) \
        fread((ptr), sizeof(short), (len), (FILE*)(file))

void amps_ReadInt();

#define amps_ReadLong(file, ptr, len) \
        fread((ptr), sizeof(long), (len), (FILE*)(file))

void amps_ReadDouble();

#define amps_Error(name, type, comment, operation)



#if 1

#define amps_TAlloc(type, count) \
        (type*)_amps_TAlloc(count * sizeof(type), __FILE__, __LINE__)

#define amps_CTAlloc(type, count) \
        (type*)_amps_CTAlloc(count * sizeof(type), __FILE__, __LINE__)


/* note: the `else' is required to guarantee termination of the `if' */
#define amps_TFree(ptr) if (ptr) free(ptr); else

/*--------------------------------------
 * Do not check memory allocation
 *--------------------------------------*/

#else

#define amps_TAlloc(type, count) \
        ((count) ? (type*)malloc((unsigned int)(sizeof(type) * (count))) : NULL)

#define amps_CTAlloc(type, count) \
        ((count) ? (type*)calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define amps_TFree(ptr) if (ptr) free(ptr, amps_arena); else

#endif

/*===========================================================================*/
/*===========================================================================*/

typedef struct amps_shmem_buffer {
  struct amps_shmem_buffer *next;
  struct amps_shmem_buffer *prev;

  char *data;

  int count;
} AMPS_ShMemBuffer;

extern AMPS_ShMemBuffer **buffers;
extern AMPS_ShMemBuffer **buffers_end;

extern AMPS_ShMemBuffer **buffers_local;
extern AMPS_ShMemBuffer **buffers_local_end;

extern AMPS_ShMemBuffer **bcast_buffers;
extern AMPS_ShMemBuffer **bcast_buffers_end;

extern AMPS_ShMemBuffer **bcast_buffers_local;
extern AMPS_ShMemBuffer **bcast_buffers_local_end;

/* Fill in some missing functions */
#define sleep(seconds) Sleep((seconds) * 1000)

#define M_PI 3.14159265358979323846

#include "amps_proto.h"
#endif



