/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#include <signal.h>
#include <math.h>
#include <stdio.h>

#undef IEEE_HANDLER

#ifdef AMPS_SUNOS_SIGNALS
#define IEEE_HANDLER
#define AMPS_UNIX_SIGNALS
int printf();
#endif

#ifdef AMPS_SOLAROS_SIGNALS
#define IEEE_HANDLER
#define AMPS_UNIX_SIGNALS
#endif

#ifdef IEEE_HANDLER
#include <floatingpoint.h>
#endif

#ifdef AMPS_OSF1_SIGNALS
#endif

#ifdef AMPS_IRIX_SIGNALS 
#include <sigfpe.h>
#define AMPS_UNIX_SIGNALS
#endif

#ifdef AMPS_UNIX_SIGNALS

void handler_ill ()  {
   printf ( "*** EXCEPTION:  Illegal instruction\n" );
   abort();
}
void handler_bus ()  {
   printf ( "*** EXCEPTION:  Bus error\n" );
   abort();
}
void handler_seg ()  {
   printf ( "*** EXCEPTION:  Segmentation violation\n" );
   abort();
}
void handler_sys ()  {
   printf ( "*** EXCEPTION:  Bad arg to system call\n" );
   abort();
}
void handler_fpe ()  {
   printf ( "*** EXCEPTION:  Floating point exception\n" );
   abort();
}

void handler_division ()  {
   printf ( "*** EXCEPTION:  Division by zero\n" );
   abort();
}
void handler_overflow ()  {
   printf ( "*** EXCEPTION:  Overflow\n" );
   abort();
}
void handler_invalid ()  {
   printf ( "*** EXCEPTION:  Invalid operand\n" );
   abort();
}

#endif

/*
 *  --------------------------------------------------------------------
 *  user routine:  Fsignal
 *  input argument is actually not currently used. 
 *  --------------------------------------------------------------------
 */

void Fsignal ()
{

#ifdef AMPS_UNIX_SIGNALS
   signal ( SIGILL,  handler_ill );  /* Illegal instruction         */
   signal ( SIGBUS,  handler_bus );  /* Bus error                   */
   signal ( SIGSEGV, handler_seg );  /* Segmentation violation      */
   signal ( SIGSYS,  handler_sys );  /* Bad argument to system call */
#endif

#ifdef IEEE_HANDLER 

   /*  
    *  AMPS_SUNOS_SIGNALS, AMPS_SOLAROS_SIGNALS
    *
    *  Add calls to IEEE_HANDLER to handle some of the IEEE floating
    *  point exceptions: 
    *  INEXACT (inexact fp approximation (eg. 2/3)) IGNORE exception
    *  DIVISION                                     CATCH  exception
    *  UNDERFLOW                                    IGNORE exception
    *  OVERFLOW                                     CATCH  exception
    *  INVALID                                      CATCH  exception
    *  The keyword ALL covers all five exceptions.
    *
    *  Do not use SIGFPE, else it will catch the exception before
    *  any of the following:
    */ 

   (void) ieee_handler ( "clear", "inexact", (void(*)())0 ); 
   (void) ieee_handler ( "set", "division", handler_division );        
   (void) ieee_handler ( "set", "overflow", handler_overflow );        
   (void) ieee_handler ( "set", "invalid", handler_invalid ); 
#endif

#ifdef AMPS_IRIX_SIGNALS
   signal ( SIGFPE, handler_fpe );  /* Arithmetic exception        */ 
   handle_sigfpes ( _ON, _EN_DIVZERO, 0, _REPLACE_HANDLER_ON_ERROR, 
                    (void(*)()) handler_division );
   handle_sigfpes ( _ON, _EN_OVERFL, 0, _REPLACE_HANDLER_ON_ERROR, 
                    (void(*)()) handler_overflow );
   handle_sigfpes ( _ON, _EN_INVALID, 0, _REPLACE_HANDLER_ON_ERROR, 
                    (void(*)()) handler_invalid );
#endif
}
