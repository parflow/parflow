/****************************************************************************
                  INTERNATIONAL AVS CENTER
	(This disclaimer must remain at the top of all files)

WARRANTY DISCLAIMER

This module and the files associated with it are distributed free of charge.
It is placed in the public domain and permission is granted for anyone to use,
duplicate, modify, and redistribute it unless otherwise noted.  Some modules
may be copyrighted.  You agree to abide by the conditions also included in
the AVS Licensing Agreement, version 1.0, located in the main module
directory located at the International AVS Center ftp site and to include
the AVS Licensing Agreement when you distribute any files downloaded from 
that site.

The International AVS Center, MCNC, the AVS Consortium and the individual
submitting the module and files associated with said module provide absolutely
NO WARRANTY OF ANY KIND with respect to this software.  The entire risk as to
the quality and performance of this software is with the user.  IN NO EVENT
WILL The International AVS Center, MCNC, the AVS Consortium and the individual
submitting the module and files associated with said module BE LIABLE TO
ANYONE FOR ANY DAMAGES ARISING FROM THE USE OF THIS SOFTWARE, INCLUDING,
WITHOUT LIMITATION, DAMAGES RESULTING FROM LOST DATA OR LOST PROFITS, OR ANY
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

This AVS module and associated files are public domain software unless
otherwise noted.  Permission is hereby granted to do whatever you like with
it, subject to the conditions that may exist in copyrighted materials. Should
you wish to make a contribution toward the improvement, modification, or
general performance of this module, please send us your comments:  why you
liked or disliked it, how you use it, and most important, how it helps your
work. We will receive your comments at avs@ncsc.org.

Please send AVS module bug reports to avs@ncsc.org.

******************************************************************************/
#include <stdlib.h>
/* IAC CODE CHANGE : #include <math.h> */
#include <avs/avs_math.h>
#include <avs/flow.h>
#include <avs/avs_data.h>
#include <avs/field.h>

/******************************************************************************/
/*   Author: Lars Bishop (bishop@viz1.ceg.uiuc.edu) */

/* 
 * Floating-point math modules - This file includes one of 4 very simple modules
 * designed to allow basic math and conversion upon real and integer data
 * types.  If you would like to have all 4 of these modules compiled into
 * a single executable file, please see the file "real-math.c.orig_code"
 * in this directory.  The four individual modules are:
 * 1) Real Math - takes in one or two real inputs, and does the selected
 *    math operation, as follows -
 *    Unary ops : output = op(input 1)
 *    Binary ops : (if two inputs connected)  output = input 1 op input 2
 *                 (if one input is connected) output = input 1 op scalar
 *
 * 2) Real to Integer - takes in a real input, and passes the truncated real
 *    as an integer to its output port
 *
 * 3) Integer to Real - takes in an integer input, and passes the integer
 *    as a real to its output port
 * 
 * 4) Print Real - Prints the real-valued input port's value to a widget
 */


intToReal()
{
  int intToRealCompute();	/* declare the compute function (below) */
  int in_port, out_port;	/* temporaries to hold the port numbers */
  
  /* Set the module name and type */
  AVSset_module_name("integer to real", MODULE_FILTER);
  
  /* Create an input port for the required field input */
  in_port =
    AVScreate_input_port("Input Integer",
			 "integer", REQUIRED);
  
  /* Create an output port for the result */
  out_port = AVScreate_output_port("Output Real",
				   "real");
  
  AVSautofree_output(out_port);

  /* Tell avs what subroutine to call to do the compute */
  AVSset_compute_proc(intToRealCompute);
  AVSset_module_flags(COOPERATIVE | REENTRANT);
}


intToRealCompute(input, output)
int input;
float **output;
{
  *output = (float*)malloc(sizeof(float));
  **output = (float)(input);
   
  return(1);
    /* When we're done, we return 1 to indicate success */
}
