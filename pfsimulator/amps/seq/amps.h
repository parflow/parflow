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

#ifndef amps_include
#define amps_include

#include "parflow_config.h"

#include "amps_common.h"

#include <stdio.h>
#include <sys/times.h>

#ifdef AMPS_MALLOC_DEBUG
#include <gmalloc.h>
#else
#include <stdlib.h>
#endif

#define AMPS_EXCHANGE_SPECIALIZED
#define AMPS_FOPEN_SPECIALIZED
#define AMPS_NEWHANDLE_SPECIALIZED
#define AMPS_SFCLOSE_SPECIALIZED
#define AMPS_SFOPEN_SPECIALIZED
#define AMPS_WAIT_SPECIALIZED

#define amps_ThreadLocal(arg) arg
#define amps_ThreadLocalDcl(type, arg) type arg

#define amps_SyncOp 0
#define amps_Max 1
#define amps_Min 2
#define amps_Add 3

#define amps_HostRank -1

#define amps_CommWorld 0

typedef int amps_Comm;
typedef FILE *amps_File;

#define amps_FreeHandle(handle) free((handle));
#define amps_Rank(comm) 0
#define amps_Size(comm) 1

#define amps_SFopen(filename, type) fopen((filename), (type))
#define amps_SFclose(file) fclose((file))

#define amps_Fclose(file)  fclose((file))
#define amps_Fprintf fprintf
#define amps_Fscanf fscanf

#define amps_Sync(comm)

#define amps_Exit(code) exit(code)

#define amps_AllReduce(comm, invoice, operation)

#define amps_BCast(comm, source, invoice) 0

#define amps_NewHandle(comm, id, invoice)

/* If we are doing malloc checking shutdown the malloc logger */
#ifdef AMPS_MALLOC_DEBUG
#define amps_Finalize() malloc_verify(0); malloc_shutdown()
#else
#define amps_Finalize()
#endif

#define amps_Fopen(filename, type) fopen((filename), (type))
#define amps_Init(argc, argv) amps_clock_init(), 0
#define amps_EmbeddedInit() amps_clock_init(), 0

#define amps_IExchangePackage(package) 0

#define amps_Wait(handle) 0

/*---------------------------------------------------------------------------*/
/* Macros for all commands that have no function in sequential code.         */
/*---------------------------------------------------------------------------*/

#define VOID_FUNC(amps_name) printf("AMPS Error: The %s function is not implemented\n", amps_name)

#define amps_IRecv(comm, source, invoice) VOID_FUNC("amps_IRecv")

#define amps_Recv(comm, source, invoice) VOID_FUNC("amps_Recv")

#define amps_ISend(comm, dest, invoice) VOID_FUNC("amps_ISend")
#define amps_Send(comm, dest, invoide) VOID_FUNC("amps_Send")

#define amps_Test(handle) VOID_FUNC("amps_test")

#define amps_new(comm, size) VOID_FUNC("amps_new")

#define amps_free(comm, buf) VOID_FUNC("amps_free")

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE  1
#endif


/* This structure is used to keep track of the entries in an invoice         */
typedef struct amps_invoice {
  long flags;         /* some flags for this invoice */

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
  int     *ptr_stride;

  int dim_type;
  int dim;
  int    *ptr_dim;

  int ignore;            /* do we ignore this invoice?                       */

  struct amps_invoice_entry *next;
} amps_InvoiceEntry;

/*===========================================================================*/
/* Package structure is used by the Exchange functions.  Contains several    */
/* Invoices plus the src or dest rank.                                       */
/*===========================================================================*/

typedef struct {
  int num_send;
  int           *dest;
  amps_Invoice  *send_invoices;

  int num_recv;
  int           *src;
  amps_Invoice  *recv_invoices;

  int recv_remaining;
  struct amps_HandleObject **recv_handles;
} amps_PackageStruct;

typedef amps_PackageStruct *amps_Package;

typedef struct amps_HandleObject {
  int type;
  amps_Comm comm;
  int id;
  amps_Invoice invoice;
  amps_Package package;
} *amps_Handle;

/****************************************************************************
 *
 *   PACKING structures and defines
 *
 *****************************************************************************/

#define AMPS_PACKED 2

#define AMPS_IGNORE  -1

#define PACK_HOST_TYPE 1
#define PACK_NO_CONVERT_TYPE 2

#define AMPS_ALIGN(type, dest)              \
  ((sizeof(type) -                          \
    ((unsigned long)(dest) % sizeof(type))) \
   % sizeof(type));

#define AMPS_SIZEOF(type, len, stride) \
  ((sizeof(type) * (len) * (stride)))

/*---------------------------------------------------------------------------*/
/* Macros for Invoice creation and deletion.                                 */
/*---------------------------------------------------------------------------*/

#define amps_append_invoice amps_new_invoice

/*---------------------------------------------------------------------------*/
/* Internal macros used to clear buffer and letter spaces.                   */
/*---------------------------------------------------------------------------*/
#if SGS
#define AMPS_CLEAR_INVOICE(invoice) \
  {                                 \
    amps_ClearInvoice(invoice);     \
  }

#define AMPS_PACK_FREE_LETTER(comm, invoice, amps_letter) \
  if ((invoice)->combuf_flags & AMPS_INVOICE_OVERLAYED)   \
    (invoice)->combuf_flags |= AMPS_INVOICE_ALLOCATED;    \
  else                                                    \
  {                                                       \
    (invoice)->combuf_flags &= ~AMPS_INVOICE_ALLOCATED;   \
    pvm_freebuf(amps_letter);                             \
  }

#endif

#define amps_FreeHandle(handle) free((handle));

#define amps_Exit(code) exit(code)

#define amps_Fclose(file)  fclose((file))
#define amps_Fprintf fprintf
#define amps_Fscanf fscanf

#define amps_FFclose(file) fclose((file))


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

#define amps_SizeofChar sizeof(char)
#define amps_SizeofShort sizeof(short)
#define amps_SizeofInt sizeof(int)
#define amps_SizeofLong sizeof(long)
#define amps_SizeofFloat sizeof(float)
#define amps_SizeofDouble sizeof(double)

#define amps_WriteChar(file, ptr, len) \
  fwrite((ptr), sizeof(char), (len), (FILE*)(file))

#define amps_WriteShort(file, ptr, len) \
  fwrite((ptr), sizeof(short), (len), (FILE*)(file))

#define amps_WriteLong(file, ptr, len) \
  fwrite((ptr), sizeof(long), (len), (FILE*)(file))

#define amps_WriteFloat(file, ptr, len) \
  fwrite((ptr), sizeof(float), (len), (FILE*)(file))

#ifdef CASC_HAVE_BIGENDIAN

#define amps_WriteInt(file, ptr, len) \
  fwrite((ptr), sizeof(int), (len), (FILE*)(file))

#define amps_WriteDouble(file, ptr, len) \
  fwrite((ptr), sizeof(double), (len), (FILE*)(file))

#endif


#define amps_ReadChar(file, ptr, len) \
  fread((ptr), sizeof(char), (len), (FILE*)(file))

#define amps_ReadShort(file, ptr, len) \
  fread((ptr), sizeof(short), (len), (FILE*)(file))

#define amps_ReadLong(file, ptr, len) \
  fread((ptr), sizeof(long), (len), (FILE*)(file))

#define amps_ReadFloat(file, ptr, len) \
  fread((ptr), sizeof(float), (len), (FILE*)(file))

#ifdef CASC_HAVE_BIGENDIAN

#define amps_ReadInt(file, ptr, len) \
  fread((ptr), sizeof(int), (len), (FILE*)(file))

#define amps_ReadDouble(file, ptr, len) \
  fread((ptr), sizeof(double), (len), (FILE*)(file))

#endif

#define amps_Error(name, type, comment, operation)

#ifdef AMPS_MEMORY_ALLOC_CHECK

#define amps_TAlloc(type, count)                                      \
  {                                                                   \
    (type*)ptr;                                                       \
    if ((ptr = (type*)malloc((unsigned int)(sizeof(type) * (count)))) \
        == NULL)                                                      \
      amps_Printf("Error: out of memory in <%s> at line %d\n",        \
                  __FILE__, __LINE__);                                \
    ptr;                                                              \
  }

#define amps_CTAlloc(type, count)                                                         \
  {                                                                                       \
    (type*)ptr;                                                                           \
    if ((ptr = (type*)calloc((unsigned int)(count), (unsigned int)sizeof(type))) == NULL) \
      amps_Printf("Error: out of memory in <%s> at line %d\n",                            \
                  __FILE__, __LINE__);                                                    \
    ptr;                                                                                  \
  }

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
#define amps_TFree(ptr) if (ptr) free(ptr); else

#endif

/* These are the built-in types that are supported */

#define AMPS_INVOICE_BYTE_CTYPE                1
#define AMPS_INVOICE_CHAR_CTYPE                2
#define AMPS_INVOICE_SHORT_CTYPE               3
#define AMPS_INVOICE_INT_CTYPE                 4
#define AMPS_INVOICE_LONG_CTYPE                5
#define AMPS_INVOICE_DOUBLE_CTYPE              6
#define AMPS_INVOICE_FLOAT_CTYPE               7
#define AMPS_INVOICE_LAST_CTYPE                8

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

#include "amps_proto.h"

#endif
