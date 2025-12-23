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
/*****************************************************************************
* Error
*
* (C) 1993 Regents of the University of California.
*
* see info_header.h for complete information
*
*                               History
*-----------------------------------------------------------------------------
* $Log: error.c,v $
* Revision 1.8  1997/10/05 20:46:21  ssmith
* Added support for reading and writing AVS Field types.  Written by
* Mike Wittman, integrated by SGS.
*
* Revision 1.7  1997/01/13 22:28:08  ssmith
* Added more support for Win32 to the Tools directory
*
* Revision 1.6  1996/08/10 06:35:26  mccombjr
*** empty log message ***
***
* Revision 1.5  1995/12/21  00:56:38  steve
* Added copyright
*
* Revision 1.4  1995/10/23  18:11:50  steve
* Added support for HDF
* Made varargs be ANSI
*
* Revision 1.3  1994/05/18  23:33:28  falgout
* Changed Error stuff, and print interactive message to stderr.
*
* Revision 1.2  1994/02/09  23:26:52  ssmith
* Cleaned up pftools and added pfload/pfunload
*
* Revision 1.1  1993/04/02  23:13:24  falgout
* Initial revision
*
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "databox.h"

#include <tcl.h>
#include <string.h>

/*-----------------------------------------------------------------------
 * Error checking routines.
 *-----------------------------------------------------------------------*/

/*-----------------------------------------------------------------------
 * Check that the grid dimensions are the same.
 *-----------------------------------------------------------------------*/

int      SameDimensions(
                        Databox *databoxp,
                        Databox *databoxq)
{
  if ((DataboxNx(databoxp) != DataboxNx(databoxq)) ||
      (DataboxNy(databoxp) != DataboxNy(databoxq)) ||
      (DataboxNz(databoxp) != DataboxNz(databoxq)))
  {
    return 0;
  }

  return 1;
}


/*-----------------------------------------------------------------------
 * Make sure a coordinate is within the range of the grid
 *-----------------------------------------------------------------------*/

int     InRange(
                int      i,
                int      j,
                int      k,
                Databox *databox)
{
  if ((i < 0 || i >= DataboxNx(databox)) ||
      (j < 0 || j >= DataboxNy(databox)) ||
      (k < 0 || k >= DataboxNz(databox)))
  {
    return 0;
  }

  return 1;
}


/* Function IsValidFileType - This function is used to make sure a given file */
/* type is a valid one.                                                       */
/*                                                                            */
/* Parameters                                                                 */
/* ----------                                                                 */
/* char *option - The file type option to be validated                        */
/*                                                                            */
/* Return value - int - One if the file type option is valid and zero other-  */
/*                      wise                                                  */

int IsValidFileType(
                    char *option)
{
  if (strcmp(option, "pfb") == 0
      || strcmp(option, "pfsb") == 0
      || strcmp(option, "sa") == 0
      || strcmp(option, "sa2d") == 0     // Added @ IMF
      || strcmp(option, "sb") == 0
      || strcmp(option, "fld") == 0
      || strcmp(option, "vis") == 0
#ifdef HAVE_SILO
      || strcmp(option, "silo") == 0
#endif
      || strcmp(option, "rsa") == 0)
    return(1);
  else
    return(0);
}


/* Function GetValidFileExtension - This function is used to determine the    */
/* extension of any given file name and determine if it is valid.             */
/*                                                                            */
/* Parameters                                                                 */
/* ----------                                                                 */
/* char *filename - The filename whose extension will be determined           */
/*                                                                            */
/* Return value - char * - A valid file extension or null if the file's       */
/*                         extension was invalid                              */

char *GetValidFileExtension(
                            char *filename)
{
  char *extension;

  /* Point the last character of the string */
  extension = filename + (strlen(filename) - 1);

  while (*extension != '.' && extension != filename)
    extension--;

  extension++;

  if (IsValidFileType(extension))
    return(extension);

  else
    return(NULL);
}


/*------------------------------------------------------------------------*/
/* These functions append various error messages to the Tcl result.       */
/*------------------------------------------------------------------------*/

/*------------------------------------------------------------------------
 * This function appends an invalid option error to the Tcl result.
 *------------------------------------------------------------------------*/

void        InvalidOptionError(
                               Tcl_Interp *interp,
                               int         argnum,
                               char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "Error: Argument ", num, " is an invalid option\n",
                   usage, (char*)NULL);
}


/*------------------------------------------------------------------------
 * This function appends an invalid argument error to the Tcl result.
 *------------------------------------------------------------------------*/

void        InvalidArgError(
                            Tcl_Interp *interp,
                            int         argnum,
                            char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "\nError: Argument ", num, " is not a valid argument\n",
                   usage, (char*)NULL);
}

/*------------------------------------------------------------------------
 * This function appends a wrong number of arguments error to the Tcl result.
 *------------------------------------------------------------------------*/

void        WrongNumArgsError(
                              Tcl_Interp *interp,
                              char *      usage)
{
  Tcl_AppendResult(interp, "\nError: Wrong number of arguments\n", usage,
                   (char*)NULL);
}


/*------------------------------------------------------------------------
 * Assign a missing option error message to the Tcl result
 *------------------------------------------------------------------------*/

void        MissingOptionError(
                               Tcl_Interp *interp,
                               int         argnum,
                               char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "\nError: Option missing at argument ", num,
                   "\n", usage, (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign a missing filename error message to the Tcl result
 *-----------------------------------------------------------------------*/

void        MissingFilenameError(
                                 Tcl_Interp *interp,
                                 int         argnum,
                                 char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "\nError: Filename missing after argument ", num,
                   "\n", usage, (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign an invalid file extension error message to the Tcl result
 *-----------------------------------------------------------------------*/

void        InvalidFileExtensionError(
                                      Tcl_Interp *interp,
                                      int         argnum,
                                      char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "\nError: Argument ", num,
                   " is a filename with an unknown extension\n", usage,
                   (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign an invalid type error message to the Tcl result when an
 * argument passed to a function is not an integer when it should be.
 *-----------------------------------------------------------------------*/

void        NotAnIntError(
                          Tcl_Interp *interp,
                          int         argnum,
                          char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp, "\nError: Integer missing at argument ", num, "\n",
                   usage, (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign an invalid type error message to the Tcl result when an
 * argument passed to a function is not a double when it should be.
 *-----------------------------------------------------------------------*/

void        NotADoubleError(
                            Tcl_Interp *interp,
                            int         argnum,
                            char *      usage)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp, "\nError: Double floating point number missing at argument ", num, "\n", usage, (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign a negative number error to the tcl result
 *-----------------------------------------------------------------------*/

void        NumberNotPositiveError(
                                   Tcl_Interp *interp,
                                   int         argnum)
{
  char num[256];

  sprintf(num, "%d", argnum);
  Tcl_AppendResult(interp, "\nError: Argument ", num, " must be greater than or equal to zero\n", (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign an out of memory error to the tcl result
 *-----------------------------------------------------------------------*/

void        MemoryError(
                        Tcl_Interp *interp)
{
  Tcl_SetResult(interp, "\nError: Memory could not be allocated to perform\n       the requested operation\n", TCL_STATIC);
}



/*-----------------------------------------------------------------------
 * Assign an undefined dataset error to the tcl result
 *-----------------------------------------------------------------------*/

void        SetNonExistantError(
                                Tcl_Interp *interp,
                                char *      hashkey)
{
  Tcl_AppendResult(interp, "\nError: `", hashkey,
                   "' is not a valid set name\n", (char*)NULL);
}


/*-----------------------------------------------------------------------
 * Assign an out of memory error to the tcl result
 *-----------------------------------------------------------------------*/

void        ReadWriteError(
                           Tcl_Interp *interp)
{
  Tcl_SetResult(interp, "\nError: The file could not be accessed or there is not enough memory available\n", TCL_STATIC);
}


/*-----------------------------------------------------------------------
 * Assign an element out of range error to the TCL result
 *-----------------------------------------------------------------------*/

void        OutOfRangeError(
                            Tcl_Interp *interp,
                            int         i,
                            int         j,
                            int         k)
{
  char message[256];

  sprintf(message, "\nError: Element (%d, %d, %d) is out of range\n", i, j, k);
  Tcl_SetResult(interp, message, TCL_STATIC);
}


/*-----------------------------------------------------------------------
 * Assign an incompatible dimensions or out of memory error to the TCL
 * result.
 *-----------------------------------------------------------------------*/

void        DimensionError(
                           Tcl_Interp *interp)
{
  Tcl_SetResult(interp, "\nError: The dimensions of the given data sets are not compatible\n", TCL_STATIC);
}
