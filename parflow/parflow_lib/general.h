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
 * General structures and values
 *
 *****************************************************************************/

#ifndef _GENERAL_HEADER
#define _GENERAL_HEADER


/*--------------------------------------------------------------------------
 * Define memory allocation routines
 *--------------------------------------------------------------------------*/

/*--------------------------------------
 * Check memory allocation
 *--------------------------------------*/

#ifdef PF_MEMORY_ALLOC_CHECK

#define talloc(type, count) \
(type *) malloc_chk((unsigned int)((count)*sizeof(type)), __FILE__, __LINE__)

#define ctalloc(type, count) \
(type *) calloc_chk((unsigned int)(count), (unsigned int)sizeof(type),\
		    __FILE__, __LINE__)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else

/*--------------------------------------
 * Do not check memory allocation
 *--------------------------------------*/

#else

#define talloc(type, count) \
((count) ? (type *) malloc((unsigned int)(sizeof(type) * (count))) : NULL)

#define ctalloc(type, count) \
((count) ? (type *) calloc((unsigned int)(count), (unsigned int)sizeof(type)) : NULL)

/* note: the `else' is required to guarantee termination of the `if' */
#define tfree(ptr) if (ptr) free(ptr); else

#endif


/*--------------------------------------------------------------------------
 * TempData macros
 *--------------------------------------------------------------------------*/

#define NewTempData(temp_data_sz)  amps_CTAlloc(double, (temp_data_sz))

#define FreeTempData(temp_data)    amps_TFree(temp_data)


/*--------------------------------------------------------------------------
 * Define various functions
 *--------------------------------------------------------------------------*/

#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef min
#define min(a,b)  (((a)<(b)) ? (a) : (b))
#endif

#ifndef round
#define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif

/* return 2^e, where e >= 0 is an integer */
#define Pow2(e)   (((unsigned int) 0x01) << (e))


/*--------------------------------------------------------------------------
 * Define various flags
 *--------------------------------------------------------------------------*/

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define ON  1
#define OFF 0

#define DO   1
#define UNDO 0

#define XDIRECTION 0
#define YDIRECTION 1
#define ZDIRECTION 2

#define CALCFCN 0
#define CALCDER 1

#define GetInt(key) IDB_GetInt(amps_ThreadLocal(input_database), (key))
#define GetDouble(key) IDB_GetDouble(amps_ThreadLocal(input_database), (key))
#define GetString(key) IDB_GetString(amps_ThreadLocal(input_database), (key))

#define GetIntDefault(key, default) IDB_GetIntDefault(amps_ThreadLocal(input_database), (key), (default))
#define GetDoubleDefault(key, default) IDB_GetDoubleDefault(amps_ThreadLocal(input_database), (key), (default))
#define GetStringDefault(key, default) IDB_GetStringDefault(amps_ThreadLocal(input_database), (key), (default))


#endif

