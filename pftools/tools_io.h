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
/* Basic I/O routines */

#ifndef TOOLS_IO_HEADER
#define TOOLS_IO_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAXPATHLEN
#ifdef MAX_PATH
#define MAXPATHLEN MAX_PATH
#else
#define MAXPATHLEN 1024
#endif
#endif

/*---------------------------------------------------------------------------*/
/* The following routines are used to actually write data to a file.         */
/* We use XDR like representation for all values written.                    */
/*---------------------------------------------------------------------------*/

#define tools_SizeofChar sizeof(char)
#define tools_SizeofShort sizeof(short)
#define tools_SizeofInt sizeof(int)
#define tools_SizeofLong sizeof(long)
#define tools_SizeofFloat sizeof(float)
#define tools_SizeofDouble sizeof(double)


#ifndef CASC_HAVE_BIGENDIAN

void tools_WriteInt(
                    FILE * file,
                    int *  ptr,
                    int    len);

void tools_WriteFloat(
                      FILE * file,
                      float *ptr,
                      int    len);

void tools_WriteDouble(
                       FILE *  file,
                       double *ptr,
                       int     len);

void tools_ReadInt(
                   FILE * file,
                   int *  ptr,
                   int    len);

void tools_ReadDouble(
                      FILE *  file,
                      double *ptr,
                      int     len);


#else
#ifdef TOOLS_CRAY

void tools_WriteInt(
                    FILE * file,
                    int *  ptr,
                    int    len);

void tools_WriteDouble(
                       FILE *  file,
                       double *ptr,
                       int     len);

void tools_ReadInt(
                   FILE * file,
                   int *  ptr,
                   int    len);

void tools_ReadDouble(
                      FILE *  file,
                      double *ptr,
                      int     len);

#else
#ifdef TOOLS_INTS_ARE_64

#define tools_WriteFloat(file, ptr, len) \
  fwrite((ptr), sizeof(float), (len), (FILE*)(file))

#define tools_WriteDouble(file, ptr, len) \
  fwrite((ptr), sizeof(double), (len), (FILE*)(file))

#define tools_ReadDouble(file, ptr, len) \
  fread((ptr), sizeof(double), (len), (FILE*)(file))

#else

/****************************************************************************/
/* Normal I/O for machines that use IEEE                                     */
/****************************************************************************/

#define tools_WriteInt(file, ptr, len) \
  fwrite((ptr), sizeof(int), (len), (FILE*)(file))

#define tools_WriteFloat(file, ptr, len) \
  fwrite((ptr), sizeof(float), (len), (FILE*)(file))

#define tools_WriteDouble(file, ptr, len) \
  fwrite((ptr), sizeof(double), (len), (FILE*)(file))

#define tools_ReadInt(file, ptr, len) \
  fread((ptr), sizeof(int), (len), (FILE*)(file))

#define tools_ReadDouble(file, ptr, len) \
  fread((ptr), sizeof(double), (len), (FILE*)(file))

#endif
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif
