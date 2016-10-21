/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

/******************************************************************************
 * Header file to include all header information for parflow netcdf IO.
 *
 *-----------------------------------------------------------------------------
 *
 *****************************************************************************/

#include "parflow.h"

#include <math.h>
#include <netcdf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// TODO: package this in a if have
#include <uuid/uuid.h>
// is standard for UNIX OS
#include <dirent.h>

// error function for the netcdf lib
#define ERRCODE 2
#define ERR(e, line)                                                           \
  {                                                                            \
    printf("Error on line %4d: %d %s\n", line, e, nc_strerror(e));             \
    exit(ERRCODE);                                                             \
  }

#define NC_INDEPENDENT 0
#define NC_COLLECTIVE 1


char *FixDots(char *orig, char *rep, char *with);
uint32_t Adler32(const void *buf, size_t buflength);
int str_ends_with(const char *str, const char *suffix);

int numfiles;
int ncid;
int retval;
int closed;

// NetCDF general settings
char nc_filename[255];
int split_mode;
int old_time;
// offset for the parflow timestep in new NetCDF files after a split
int time_index_offset;
// time offset for a NetCDF dataset that is continued
int time_cont_offset;
char *general_file_name;
uuid_t uuid;
char uuid_str[37];

int latlonid;
char *latlonfile;
char latvar_name[30];
char lonvar_name[30];
int latlondata;

// pointer for the write calls
int *dimids;
// contains 3 or 4 for the current n-dimensional variable
int dimlen;
// process the comitted var name and extract the time_index
// from char to int
int timestep;
// starting offset for the current write out for the time dimension
int time_index;
// ID for the time dimension
int time_dimid;
// ID for the x dimension
int x_dimid;
// ID for the y dimension
int y_dimid;
// ID for the z dimension
int z_dimid;
// length of the x dimension
int x_dimlen;
// length of the y dimension
int y_dimlen;
// length of the z dimension
int z_dimlen;
// actual start date specified in the TCL file
struct tm *start_time;
// id of the time variable belonging to the time dimension
int time_varid;
// value of the current time dimension variable
double timedimvar_value;

// values calculated with Adler32
enum variables_char {
  permx = 149160588,
  permy = 149226125,
  permz = 149291662,
  porosity = 265290634,
  specific = 945686171,
  press = 109380142,
  satur = 109052464,
  mask = 69992877,
  slopex = 201589499,
  slopey = 201655036,
  mannings = 251659100,
  dzmult = 198771456,
  lhtot = 456983705,
  lwradout = 733152736,
  shtot = 459736224,
  soilgrnd = 732366288,
  evaptot = 641664381,
  evapgrnd = 736691665,
  evapsoi = 640746865,
  evapveg = 639894888,
  tranveg = 645268849,
  infl = 318178244,
  sweout = 202769159,
  tgrnd = 147522175,
  tsoil = 149947019,
  qirr = 321258457,
  qirrinst = 746980854,
  evaptrans = 319488981,
  evaptranssum = 557450538,
  overlandsum = 472450225,
  overlandbcflux = 945161886
};
