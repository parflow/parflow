/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

#ifndef _GRID_MESSAGE_HEADER
#define _GRID_MESSAGE_HEADER
typedef struct {
  int nx;
  int ny;
  int nz;
  int ix;
  int iy;
  int iz;
  int nX;
  int nY;
  int nZ;
} GridMessageMetadata;

#endif
