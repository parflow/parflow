
/*-------------------------------------------------------------------------*/
/**
  @file		ptrace.c
  @author	N. Devillard, V. Chudnovsky
  @date		March 2004
  @version	$Revision: 1.1.1.1 $
  @brief	Add tracing capability to any program compiled with gcc.

  This module is only compiled when using gcc and tracing has been
  activated. It allows the compiled program to output messages whenever
  a function is entered or exited.

  To activate this feature, your version of gcc must support
  the -finstrument-functions flag.

  When using ptrace on a dynamic library, you must set the
  PTRACE_REFERENCE_FUNCTION macro to be the name of a function in the
  library. The address of this function when loaded will be the first
  line output to the trace file and will permit the translation of the
  other entry and exit pointers to their symbolic names. You may set
  the macro PTRACE_INCLUDE with any #include directives needed for
  that function to be accessible to this source file.

  The printed messages yield function addresses, not human-readable
  names. To link both, you need to get a list of symbols from the
  program. There are many (unportable) ways of doing that, see the
  'etrace' project on freshmeat for more information about how to dig
  the information.
*/
/*--------------------------------------------------------------------------*/

/*
        $Id: ptrace.c,v 1.1.1.1 2004-03-16 20:00:07 ndevilla Exp $
        $Author: ndevilla $
        $Date: 2004-03-16 20:00:07 $
        $Revision: 1.1.1.1 $
*/

#if (__GNUC__ > 2) || ((__GNUC__ == 2) && (__GNUC_MINOR__ > 95))

#define REFERENCE_OFFSET "REFERENCE:"
#define FUNCTION_ENTRY   "enter"
#define FUNCTION_EXIT    "exit"
#define END_TRACE        "EXIT"
#define __NON_INSTRUMENT_FUNCTION__    __attribute__((__no_instrument_function__))
#define PTRACE_OFF        __NON_INSTRUMENT_FUNCTION__

/*---------------------------------------------------------------------------
                                                                Includes
 ---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>


#define STR(_x)          #_x
#define DEF(_x)          _x
#define GET(_x,_y)       _x (_y)

/*---------------------------------------------------------------------------
                                                            User Macros
 ---------------------------------------------------------------------------*/
char PTRACE_PIPENAME[FILENAME_MAX];

/* When using ptrace on a dynamic library, the following must be defined:

#include "any files needed for PTRACE_REFERENCE_FUNCTION"
#define PTRACE_REFERENCE_FUNCTION functionName

`*/


/*---------------------------------------------------------------------------
                                                        Function codes
 ---------------------------------------------------------------------------*/

FILE* trace = 0;

/** Final trace close */
static void
__NON_INSTRUMENT_FUNCTION__
gnu_ptrace_close (void);

static void
gnu_ptrace_close (void)
{
  if (trace == 0)
    {
      fprintf (trace, END_TRACE " %ld\n", (long)getpid ());
      fclose (trace);
    }
}

void init_tracefile (const char* filename)
{
  strncpy (PTRACE_PIPENAME,filename,FILENAME_MAX);

  atexit (gnu_ptrace_close);

  if ((trace = fopen (PTRACE_PIPENAME, "w")) == NULL)
    {
      return;
    }

  // Turn off buffering
  setbuf (trace, 0);
}

/** Function called by every function event */
inline static void
__NON_INSTRUMENT_FUNCTION__
gnu_ptrace (const char * what, void * p);

inline static void
gnu_ptrace (const char * what, void * p)
{
  if (trace)
    {
#ifdef PTRACE_REFERENCE_FUNCTION
      fprintf (trace,"%s %s %p\n",
               REFERENCE_OFFSET,
               GET (STR,PTRACE_REFERENCE_FUNCTION),
               (void *)GET (DEF,PTRACE_REFERENCE_FUNCTION));
#endif
      fprintf (trace, "%s %p\n", what, p);
    }
}

/** According to gcc documentation: called upon function entry */

void
__NON_INSTRUMENT_FUNCTION__
__cyg_profile_func_enter (void *this_fn, void *call_site);

void
__cyg_profile_func_enter (void *this_fn, void *call_site)
{
  gnu_ptrace (FUNCTION_ENTRY, this_fn);
  (void)call_site;
}

/** According to gcc documentation: called upon function exit */
void
__NON_INSTRUMENT_FUNCTION__
__cyg_profile_func_exit (void *this_fn, void *call_site);

void
__cyg_profile_func_exit (void *this_fn, void *call_site)
{
  gnu_ptrace (FUNCTION_EXIT, this_fn);
  (void)call_site;
}

#endif
/* vim: set ts=4 et sw=4 tw=75 */
