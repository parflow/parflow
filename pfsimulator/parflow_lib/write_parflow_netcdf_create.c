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

#include "parflow_netcdf.h"

/*
    Settings and hard metadata
*/

void WriteNetCDF_MakeSettings(char *file_prefix, Vector *v) {
  Grid *grid = VectorGrid(v);
  NameArray switch_na;
  char key[IDB_MAX_KEY_LEN];
  char *switch_name;
  int switch_value;
  time_t rawtime;

  numfiles = 0;
  closed = 1;

#ifdef HAVE_NETCDF
  // parse the date from the tcl args and create the string
  time(&rawtime);
  start_time = localtime(&rawtime);
  WriteNetCDF_ResetTime();

  // set file creation information
  char dot[] = ".";
  char under[] = "_";
  general_file_name = FixDots(file_prefix, dot, under);

  // grid values
  x_dimlen = SubgridNX(GridBackground(grid));
  y_dimlen = SubgridNY(GridBackground(grid));
  z_dimlen = SubgridNZ(GridBackground(grid));

  // set split mode
  switch_na = NA_NewNameArray("none daily monthly yearly");
  sprintf(key, "NetCDF.SplitInterval");
  switch_name = GetStringDefault(key, "none");
  switch_value = NA_NameToIndex(switch_na, switch_name);
  switch (switch_value) {
  case 0: {
    split_mode = 0;
    break;
  }
  case 1: {
    split_mode = 1;
    old_time = GetInt("NetCDF.StartDateDay");
    break;
  }
  case 2: {
    split_mode = 2;
    old_time = GetInt("NetCDF.StartDateMonth");
    break;
  }
  case 3: {
    split_mode = 3;
    old_time = GetInt("NetCDF.StartDateYear");
    break;
  }
  default: {
    InputError("Error: Invalid value <%s> for key <%s>\n", switch_name, key);
  }
  }
  NA_FreeNameArray(switch_na);
  time_index_offset = 0;
#endif
}

char *FixDots(char *orig, char *rep, char *with) {
  char *result;
  char *ins;
  char *tmp;
  int len_rep;
  int len_with;
  int len_front;
  int count;

  if (!orig)
    return NULL;
  if (!rep)
    rep = "";
  len_rep = strlen(rep);
  if (!with)
    with = "";
  len_with = strlen(with);

  ins = orig;
  for (count = 0; tmp = strstr(ins, rep); ++count) {
    ins = tmp + len_rep;
  }

  tmp = result = malloc(strlen(orig) + (len_with - len_rep) * count + 1);

  if (!result)
    return NULL;

  while (count--) {
    ins = strstr(orig, rep);
    len_front = ins - orig;
    tmp = strncpy(tmp, orig, len_front) + len_front;
    tmp = strcpy(tmp, with) + len_with;
    orig += len_front + len_rep; // move to next "end of rep"
  }
  strcpy(tmp, orig);
  return result;
}

int str_ends_with(const char *str, const char *suffix) {

  if (str == NULL || suffix == NULL)
    return 0;

  size_t str_len = strlen(str);
  size_t suffix_len = strlen(suffix);

  if (suffix_len > str_len)
    return 0;

  return 0 == strncmp(str + str_len - suffix_len, suffix, suffix_len);
}

/*
                MANAGE METADATA
*/

void WriteNetCDF_GetVarModus(char *file_name) {
  /*
    This function compares all NetCDF print options and sets an according file
    name for either
    only the one var that is printed or multivar.
  */
  char t[] = "True";
  char f[] = "False";

#ifdef HAVE_NETCDF
  // if only one time varying var is printed change filename accordingly
  if (strcmp(t, GetStringDefault("NetCDF.WritePressure", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
          strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
      0) {
    sprintf(nc_filename, "press_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "satur_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "clm_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "evaptrans_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "evaptranssum_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "overlandsum_%s", file_name);
  } else if (strcmp(f, GetStringDefault("NetCDF.WritePressure", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteSaturation", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteCLM", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTrans", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteEvapTransSum", f)) +
                 strcmp(f, GetStringDefault("NetCDF.WriteOverlandSum", f)) +
                 strcmp(t, GetStringDefault("NetCDF.WriteOverlandBCFlux", f)) ==
             0) {
    sprintf(nc_filename, "overlandbcflux_%s", file_name);
  } else {
    sprintf(nc_filename, "multivar_%s", file_name);
  }
#endif
}

int WriteNetCDF_CheckTimeCompliance(char *file, char *frequency_str) {
  int varid, dimid;
  int is_split = 1;
  int is_freq = 0;
  int is_dist = 0;

#ifdef HAVE_NETCDF
  // open file and check for matching timestamps
  // if time matches we can just break while and continue easily
  nc_open_par(file, NC_NOCLOBBER | NC_MPIIO | NC_NETCDF4 | NC_WRITE,
              amps_CommWorld, amps_Info, &ncid);

  // inquire ids and data length
  char tmp_str[] = "time";
  size_t tmp_len;
  nc_inq_varid(ncid, tmp_str, &varid);
  nc_inq_dimid(ncid, tmp_str, &dimid);
  nc_inq_dimlen(ncid, dimid, &tmp_len);

  // time var start
  char att_name[] = "units";
  char date_str[80];
  struct tm ex_date;
  nc_get_att_text(ncid, varid, att_name, date_str);

  // cut to pieces
  strptime(date_str, "days since %Y-%m-%dT%TZ", &ex_date);

  // time offset
  double time_value;
  size_t index[] = {tmp_len - 1};
  nc_get_var1_double(ncid, varid, index, &time_value);

  char test_str[80];
  char test_str2[80];
  // create a new timestamp from the existing file
  ex_date.tm_hour = ex_date.tm_hour + (int)(time_value * 24.);
  ex_date.tm_isdst = 0;
  mktime(&ex_date);

  // check compliancy
  // compare that both frequencys fit
  char comp_freq_str[5];
  nc_get_att_text(ncid, NC_GLOBAL, "frequency", comp_freq_str);

  if (strcmp(comp_freq_str, frequency_str) == 0) {
    is_freq = 1;
  }

  // check for proper distance between time steps
  struct tm freq_time;
  strptime(comp_freq_str, "%hhr", &freq_time);
  int freq_sec = freq_time.tm_hour * 3600;

  WriteNetCDF_ResetTime();
  double diff_time = difftime(mktime(start_time), mktime(&ex_date));

  if (freq_sec <= diff_time) {
    is_dist = 1;
  }

  // make the time offset from the existing dataset
  time_cont_offset =
      ((int)diff_time / (3600 * GetDouble("TimingInfo.DumpInterval"))) +
      tmp_len;

  // see if split timestep
  switch (split_mode) {
  case 0: {
    if (start_time->tm_mday != old_time) {
      is_split = 0;
      WriteNetCDF_ResetTime();
    }
    break;
  }
  case 2: {
    // monthly splitting
    if (start_time->tm_mon == old_time % 12) {
      is_split = 0;
      WriteNetCDF_ResetTime();
    }
    break;
  }
  case 3: {
    // yearly splitting
    // create the new file if we are in a new year
    if (start_time->tm_year + 1900 != old_time) {
      is_split = 0;
      WriteNetCDF_ResetTime();
    }
    break;
  }
  default: { break; }
  }
#endif

  return (is_split && is_freq && is_dist);
}

void WriteNetCDF_CreateNewFile() {
  char time_string[80];
  char time_file_suffix[80];
  char file_prefix[80];
  char file_type[20];
  char file_name[255];
  struct tm *c_time;
  time_t rawtime;

#ifdef HAVE_NETCDF
  // keep count of the amount of created files
  numfiles++;

  // dataset always starts on the first day of the month
  strftime(time_string, 80, "days since %FT00:00:00Z", start_time);
  strftime(time_file_suffix, 80, "%Y%m%d-%H%M%S", start_time);

  // calc the frequency string
  char frequency_str[5];
  sprintf(frequency_str, "%dhr", (int)GetDouble("TimingInfo.DumpInterval"));

  if (strlen(GetStringDefault("NetCDF.HardFileName", "")) == 0) {
    // make the preliminary file name
    sprintf(file_name, "%s_%s_%s_%s_%s_%s_%s_%s.nc",
            GetStringDefault("NetCDF.Domain", "none"),
            GetStringDefault("NetCDF.DrivingModelID", "none"),
            GetStringDefault("NetCDF.DrivingExperiment", "none"),
            GetStringDefault("NetCDF.DrivingModelEnsemble", "none"),
            GetStringDefault("NetCDF.ModelID", "none"),
            GetStringDefault("NetCDF.VersionID", "none"), frequency_str,
            time_file_suffix);

    // complete the nc_filename variable
    WriteNetCDF_GetVarModus(file_name);
  } else {
    // set a hard name and conditionally set a timestamp to not overwrite data
    if (0 == strcmp("none", GetStringDefault("NetCDF.SplitInterval", "none"))) {
      sprintf(nc_filename, "%s.nc", GetString("NetCDF.HardFileName"));
    } else {
      sprintf(nc_filename, "%s_%s.nc", GetString("NetCDF.HardFileName"),
              time_file_suffix);
    }
  }

  // create the initial file or start appending to a new file
  if (strlen(GetStringDefault("NetCDF.AppendDirectory", "")) > 0) {
    // search the complete directory set by the user
    DIR *dir;
    struct dirent *ent;
    int fit = 0;

    if ((dir = opendir(".")) != NULL) {
      // print all the files and directories within directory
      while ((ent = readdir(dir)) != NULL) {
        // if the filetype matches open the NetCDF file
        if (str_ends_with(ent->d_name, ".nc")) {
          if (WriteNetCDF_CheckTimeCompliance(ent->d_name, frequency_str))
            break;
          else
            FreeNetCDF();
        }
      }
      closedir(dir);
    } else {
      /* could not open directory */
      ERR(-77, __LINE__);
    }
  }

  if (closed) {
    if ((retval = nc_create_par(&nc_filename, NC_MPIIO | NC_NETCDF4,
                                amps_CommWorld, amps_Info, &ncid)))
      ERR(retval, __LINE__);
    time_cont_offset = 0;
  } else {
    nc_inq_dimid(ncid, "x", &x_dimid);
    nc_inq_dimid(ncid, "y", &y_dimid);
    nc_inq_dimid(ncid, "z", &z_dimid);
    nc_inq_dimid(ncid, "time", &time_dimid);
    return;
  }

  // create the NetCDF dimensions base on the comitted grid
  if ((retval = nc_def_dim(ncid, "x", x_dimlen, &x_dimid)))
    ERR(retval, __LINE__);

  if ((retval = nc_def_dim(ncid, "y", y_dimlen, &y_dimid)))
    ERR(retval, __LINE__);

  if ((retval = nc_def_dim(ncid, "z", z_dimlen, &z_dimid)))
    ERR(retval, __LINE__);

  if ((retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dimid)))
    ERR(retval, __LINE__);

  // create the dimension variables
  // TIME dimension variable
  if ((retval =
           nc_def_var(ncid, "time", NC_DOUBLE, 1, &time_dimid, &time_varid)))
    ERR(retval, __LINE__);

  WriteNetCDF_Metadata(time_varid, "time", time_string, "time");
  nc_put_att_text(ncid, time_varid, "calendar", 8, "standard");
  nc_put_att_text(ncid, time_varid, "axis", 1, "T");

  // GLOBAL attributes
  // convention version
  nc_put_att_text(ncid, NC_GLOBAL, "Conventions", 6, "CF-1.6");
  nc_put_att_text(ncid, NC_GLOBAL, "institution",
                  strlen(GetStringDefault("NetCDF.Institution", "none")),
                  GetStringDefault("NetCDF.Institution", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "institute_id",
                  strlen(GetStringDefault("NetCDF.InstituteID", "none")),
                  GetStringDefault("NetCDF.InstituteID", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "model_id",
                  strlen(GetStringDefault("NetCDF.ModelID", "none")),
                  GetStringDefault("NetCDF.ModelID", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "experiment",
                  strlen(GetStringDefault("NetCDF.Experiment", "none")),
                  GetStringDefault("NetCDF.Experiment", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "experiment_id",
                  strlen(GetStringDefault("NetCDF.ExperimentID", "none")),
                  GetStringDefault("NetCDF.ExperimentID", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "contact",
                  strlen(GetStringDefault("NetCDF.Contact", "none")),
                  GetStringDefault("NetCDF.Contact", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "product",
                  strlen(GetStringDefault("NetCDF.Product", "none")),
                  GetStringDefault("NetCDF.Product", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "driving_model_id",
                  strlen(GetStringDefault("NetCDF.DrivingModelID", "none")),
                  GetStringDefault("NetCDF.DrivingModelID", "none"));
  nc_put_att_text(
      ncid, NC_GLOBAL, "driving_model_ensemble_member",
      strlen(GetStringDefault("NetCDF.DrivingModelEnsemble", "none")),
      GetStringDefault("NetCDF.DrivingModelEnsemble", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "driving_experiment_name",
                  strlen(GetStringDefault("NetCDF.DrivingExperiment", "none")),
                  GetStringDefault("NetCDF.DrivingExperiment", "none"));
  char driving_experiment[80];
  sprintf(driving_experiment, "%s,%s,%s",
          GetStringDefault("NetCDF.DrivingModelID", "none"),
          GetStringDefault("NetCDF.DrivingExperiment", "none"),
          GetStringDefault("NetCDF.DrivingModelEnsemble", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "driving_experiment",
                  strlen(driving_experiment), driving_experiment);
  nc_put_att_text(ncid, NC_GLOBAL, "version_id",
                  strlen(GetStringDefault("NetCDF.VersionID", "none")),
                  GetStringDefault("NetCDF.VersionID", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "domain",
                  strlen(GetStringDefault("NetCDF.Domain", "none")),
                  GetStringDefault("NetCDF.Domain", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "project_id",
                  strlen(GetStringDefault("NetCDF.ProjectID", "none")),
                  GetStringDefault("NetCDF.ProjectID", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "references",
                  strlen(GetStringDefault("NetCDF.References", "none")),
                  GetStringDefault("NetCDF.References", "none"));
  nc_put_att_text(ncid, NC_GLOBAL, "comment",
                  strlen(GetStringDefault("NetCDF.Comment", "none")),
                  GetStringDefault("NetCDF.Comment", "none"));

  // set the creation date of the dataset
  time(&rawtime);
  c_time = localtime(&rawtime);
  strftime(time_string, 80, "%FT%TZ", c_time);
  nc_put_att_text(ncid, NC_GLOBAL, "creation_date", strlen(time_string),
                  time_string);

  // set the uuid
  uuid_generate(uuid);
  uuid_unparse_lower(uuid, uuid_str);
  nc_put_att_text(ncid, NC_GLOBAL, "tracking_id", 37, uuid_str);
  uuid_clear(uuid);

  nc_put_att_text(ncid, NC_GLOBAL, "frequency", 5, frequency_str);

  // GLOBAL TCL attributes
  // recursively calls all HBT elements and write them as global attributes
  // into the NetCDF dataset
  WriteNetCDF_GlobalTCLAttributes(WriteNetCDF_WriteGlobalAttribute,
                                  (amps_ThreadLocal(input_database))->root);

  latlondata = 0;
  if (WriteNetCDF_LatLonDataAvail()) {
    nc_open_par(latlonfile, NC_MPIIO, amps_CommWorld, amps_Info, &latlonid);

    int latvarid, lonvarid, targetlatid, targetlonid;
    float field[y_dimlen * x_dimlen];
    int latlondimarr[] = {y_dimid, x_dimid};

    nc_inq_varid(latlonid, latvar_name, &latvarid);
    nc_get_var_float(latlonid, latvarid, &field[0]);

    nc_def_var(ncid, "lat", NC_FLOAT, 2, latlondimarr, &targetlatid);
    WriteNetCDF_Metadata(targetlatid, "latitude", "degrees_north", "Latitude");
    nc_put_var_float(ncid, targetlatid, &field[0]);

    nc_inq_varid(latlonid, lonvar_name, &lonvarid);
    nc_get_var_float(latlonid, lonvarid, &field[0]);

    nc_def_var(ncid, "lon", NC_FLOAT, 2, latlondimarr, &targetlonid);
    WriteNetCDF_Metadata(targetlonid, "longitude", "degrees_east", "Longitude");
    nc_put_var_float(ncid, targetlonid, &field[0]);

    nc_close(latlonid);
    nc_redef(ncid);

    latlondata = 1;
  }

  sprintf(file_prefix, "%s", GlobalsOutFileName);
#endif

  return;
}

int WriteNetCDF_LatLonDataAvail() {
#ifdef HAVE_NETCDF
  NameArray names=
      NA_NewNameArray(GetStringDefault("NetCDF.LatLonNames", "none"));
  if (names->num != 2) {
    return 0;
  } else {
    latlonfile = (char *)malloc(strlen(GetStringDefault("NetCDF.LatLonFile", "")));
    latlonfile = GetStringDefault("NetCDF.LatLonFile", "");
    sprintf(latvar_name, "%s", NA_IndexToName(names, 0));
    sprintf(lonvar_name, "%s", NA_IndexToName(names, 1));
    return 1;
  }
#endif
  return 0;
}

void WriteNetCDF_Timestamp() {
  size_t startp[1], countp[1];
  char time_string[80];

#ifdef HAVE_NETCDF
  if ((retval = nc_var_par_access(ncid, time_varid, NC_COLLECTIVE)))
    ERR(retval, __LINE__);

  WriteNetCDF_ResetTime();
  strftime(time_string, 80, "%F %T", start_time);

  // decide whether a new file needs to be created
  switch (split_mode) {
  case 0: {
    // no splitting just continues
    break;
  }
  case 1: {
    // daily splitting
    // create the new file if we are on a new day
    if (start_time->tm_mday != old_time) {
      old_time = start_time->tm_mday;
      time_index_offset = time_index;
      // first close the old file
      FreeNetCDF();

      WriteNetCDF_CreateNewFile();
      // collective access optimisation
      if ((retval = nc_var_par_access(ncid, time_varid, NC_COLLECTIVE)))
        ERR(retval, __LINE__);

      // enter data mode to write stuff
      if ((retval = nc_enddef(ncid)))
        ERR(retval, __LINE__);

      WriteNetCDF_ResetTime();
    }
    break;
  }
  case 2: {
    // monthly splitting
    // create the new file if we are in a new month
    if (start_time->tm_mon == old_time % 12) {
      old_time = start_time->tm_mon + 1;
      time_index_offset = time_index;
      // first close the old file
      FreeNetCDF();

      WriteNetCDF_CreateNewFile();
      // collective access optimization
      if ((retval = nc_var_par_access(ncid, time_varid, NC_COLLECTIVE)))
        ERR(retval, __LINE__);

      // enter data mode to write stuff
      if ((retval = nc_enddef(ncid)))
        ERR(retval, __LINE__);

      WriteNetCDF_ResetTime();
    }
    break;
  }
  case 3: {
    // yearly splitting
    // create the new file if we are in a new year
    if (start_time->tm_year + 1900 != old_time) {
      old_time = start_time->tm_year + 1;
      time_index_offset = time_index;
      // first close the old file
      FreeNetCDF();

      WriteNetCDF_CreateNewFile();
      // collective access optimization
      if ((retval = nc_var_par_access(ncid, time_varid, NC_COLLECTIVE)))
        ERR(retval, __LINE__);

      // enter data mode to write stuff
      if ((retval = nc_enddef(ncid)))
        ERR(retval, __LINE__);

      WriteNetCDF_ResetTime();
    }
    break;
  }
  default: {
    // literally nothing
    break;
  }
  }

  // calculate the current value for the time dimension variable
  if (numfiles == 1) {
    timedimvar_value = start_time->tm_mday + 1.0 / 24 * (start_time->tm_hour) -
                       (GetInt("NetCDF.StartDateDay"));
  } else {
    timedimvar_value =
        start_time->tm_mday + 1.0 / 24 * (start_time->tm_hour) - 1;
  }

  startp[0] = time_cont_offset + time_index - time_index_offset;
  countp[0] = 1;
  if ((retval = nc_put_vara_double(ncid, time_varid, &startp[0], &countp[0],
                                   &timedimvar_value)))
    ERR(retval, __LINE__);
#endif
}

void WriteNetCDF_ResetTime() {
  start_time->tm_year = GetInt("NetCDF.StartDateYear") - 1900;
  start_time->tm_mon = GetInt("NetCDF.StartDateMonth") - 1;
  start_time->tm_mday = GetInt("NetCDF.StartDateDay");
  start_time->tm_min = 60 *
                       (time_cont_offset + time_index + time_index_offset +
                        GetDefaultInt("TimingInfo.StartCount", 0)) *
                       (int)GetDouble("TimingInfo.BaseUnit") *
                       (int)GetDouble("TimingInfo.DumpInterval");
  start_time->tm_hour = 0;
  start_time->tm_sec = 0;
  start_time->tm_isdst = 0;
  mktime(start_time);
}

void WriteNetCDF_GlobalTCLAttributes(void (*write_method)(void *),
                                     HBT_element *tree) {
  if (tree != NULL) {
    WriteNetCDF_GlobalTCLAttributes(write_method, tree->left);
    (*write_method)(tree->obj);
    WriteNetCDF_GlobalTCLAttributes(write_method, tree->right);
  }
}

void WriteNetCDF_WriteGlobalAttribute(void *obj) {
  IDB_Entry *entry = (IDB_Entry *)obj;

#ifdef HAVE_NETCDF
  nc_put_att_text(ncid, NC_GLOBAL, entry->key, strlen(entry->value),
                  entry->value);
#endif
}

void WriteNetCDF_Metadata(int varid, char *standard_name, char *units,
                          char *long_name) {
#ifdef HAVE_NETCDF
  nc_put_att_text(ncid, varid, "standard_name", strlen(standard_name),
                  standard_name);
  nc_put_att_text(ncid, varid, "units", strlen(units), units);
  nc_put_att_text(ncid, varid, "long_name", strlen(long_name), long_name);

  int type;
  nc_inq_vartype(ncid, varid, &type);

  if (type == NC_DOUBLE) {
    double fill = -999.9;
    nc_put_att_double(ncid, varid, "_FillValue", NC_DOUBLE, 1, &fill);
  } else if (type == NC_FLOAT) {
    float fill = -999.9;
    nc_put_att_float(ncid, varid, "_FillValue", NC_FLOAT, 1, &fill);
  }

  if (latlondata) {
    nc_put_att_text(ncid, varid, "coordinates", 7, "lon lat");
  }
#endif
}

void SetNetCDF_VariableChunking(char *file_prefix, char *varname,
                                size_t *chunks) {
  int varid;

#ifdef HAVE_NETCDF
  varid = WriteNetCDF_Variable(varname);
  if ((retval = nc_def_var_chunking(ncid, varid, 1, chunks)))
    ERR(retval, __LINE__);
#endif
}

int WriteNetCDF_Variable(char *varname) {
  int varid;

#ifdef HAVE_NETCDF
  // create the variable itself; varname could be changed
  retval = nc_def_var(ncid, varname, NC_FLOAT, dimlen, dimids, &varid);
  if ((retval != NC_NOERR) && (retval != NC_ENAMEINUSE)) {
    ERR(retval, __LINE__);
    // if metadata is defined return varid and abort rest of the function
  } else if (retval == NC_ENAMEINUSE) {
    retval = nc_inq_varid(ncid, varname, &varid);
    if (retval != NC_NOERR)
      ERR(retval, __LINE__);
    return varid;
  }

  // add the additional CF-1.6 metadata
  switch (Adler32(varname, strlen(varname))) {
  case permx:
    WriteNetCDF_Metadata(varid, "x_permeability", "m2",
                         "Permeability in x Direction");
    break;
  case permy:
    WriteNetCDF_Metadata(varid, "y_permeability", "m2",
                         "Permeability in y Direction");
    break;
  case permz:
    WriteNetCDF_Metadata(varid, "z_permeability", "m2",
                         "Permeability in z Direction");
    break;
  case porosity:
    WriteNetCDF_Metadata(varid, "soil_porosity", "1", "Porosity");
    break;
  case specific:
    WriteNetCDF_Metadata(varid, "specific_storage", "L-1", "Specific Storage");
    break;
  case press:
    WriteNetCDF_Metadata(varid, "pressure_head_in_soil", "m", "Pressure Head");
    break;
  case satur:
    WriteNetCDF_Metadata(varid, "volume_fraction_of_water_in_soil", "1",
                         "Saturation");
    break;
  case mask:
    WriteNetCDF_Metadata(varid, "land_sea_mask", "1", "Land Sea Mask");
    break;
  case slopex:
    WriteNetCDF_Metadata(varid, "x_slope", "degrees",
                         "Degree of the Slopes in x Direction");
    break;
  case slopey:
    WriteNetCDF_Metadata(varid, "y_slope", "degrees",
                         "Degree of the Slopes in y Direction");
    break;
  case mannings:
    WriteNetCDF_Metadata(varid, "mannings", "m s-1", "Mannings Roughness");
    break;
  case dzmult:
    WriteNetCDF_Metadata(varid, "dz_multiplier", "L", "dZ Multiplier");
    break;
  case lhtot:
    WriteNetCDF_Metadata(varid, "surface_upward_latent_heat_flux", "W m-2",
                         "Surface Upward Latent Heat Flux");
    break;
  case lwradout:
    WriteNetCDF_Metadata(varid, "surface_upwelling_longwave_flux_in_air",
                         "W m-2", "Surface Upwelling Longwave Radiation");
    break;
  case shtot:
    WriteNetCDF_Metadata(varid, "surface_upward_sensible_heat_flux", "W m-2",
                         "Surface Upward Sensible Heat Flux");
    break;
  case soilgrnd:
    WriteNetCDF_Metadata(varid, "ground_upward_heat_flux", "W m-2",
                         "Ground Upward Heat Flux");
    break;
  case evaptot:
    WriteNetCDF_Metadata(varid, "total_evaporation", "mm s-1",
                         "Total Evaporation");
    break;
  case evapgrnd:
    WriteNetCDF_Metadata(
        varid, "water_evaporation_flux_from_soil_no_sublimation", "mm s-1",
        "Water Evaporation from Soil without Sublimation");
    break;
  case evapsoi:
    WriteNetCDF_Metadata(varid, "water_evaporation_flux_from_soil", "mm s-1",
                         "Water Evaporation from Soil");
    break;
  case evapveg:
    WriteNetCDF_Metadata(varid, "water_evaporation_flux_from_vegetation",
                         "mm s-1", "Water Evaporation from Vegetation");
    break;
  case tranveg:
    WriteNetCDF_Metadata(varid, "water_transpiration_flux_from_vegetation",
                         "mm s-1", "Water Transpiration from Vegetation");
    break;
  case infl:
    WriteNetCDF_Metadata(varid, "soil_infiltration", "mm s-1",
                         "Soil Infiltration");
    break;
  case sweout:
    WriteNetCDF_Metadata(varid, "snow_water_equivalent", "mm",
                         "Snow Water Equivalent");
    break;
  case tgrnd:
    WriteNetCDF_Metadata(varid, "surface_temperature", "K",
                         "Surface Temperature");
    break;
  case tsoil:
    WriteNetCDF_Metadata(varid, "soil_temperature", "K", "Temperature of Soil");
    break;
  case qirr:
    WriteNetCDF_Metadata(varid, "surface_irrigation", "m",
                         "Surface Irrigation");
    break;
  case qirrinst:
    WriteNetCDF_Metadata(varid, "instant_irrigation", "m",
                         "Instant Irrigation");
    break;
  case evaptrans:
    WriteNetCDF_Metadata(varid, "rainfall_evaporation_flux", "L3 T-1",
                         "Rainfall and Evaporation Flux");
    break;
  case evaptranssum:
    WriteNetCDF_Metadata(varid, "cumulative_rainfall_evaporation_flux", "L3",
                         "Cumulative Rainfall and Evaporation Flux");
    break;
  case overlandsum:
    WriteNetCDF_Metadata(varid, "cumulative_overland_outflow", "L3",
                         "Cumulative Overland Outflow");
    break;
  case overlandbcflux:
    WriteNetCDF_Metadata(varid, "overland_bc_flux", "W m-2", "OverlandBCFlux");
    break;
  default:
    break;
  }
#endif

  return varid;
}

void FreeNetCDF() {
#ifdef HAVE_NETCDF
  nc_close(ncid);
#endif
  closed = 1;
}

void OpenNetCDF(char *file) {
#ifdef HAVE_NETCDF
  nc_open_par(file, NC_MPIIO | NC_NETCDF4 | NC_NOWRITE, amps_CommWorld,
              amps_Info, &ncid);
#endif
  closed = 0;
}

// use this to hash varnames and compare them in a switch case
uint32_t Adler32(const void *buf, size_t buflength) {
  const uint8_t *buffer = (const uint8_t *)buf;

  uint32_t s1 = 1;
  uint32_t s2 = 0;
  size_t n;

  for (n = 0; n < buflength; n++) {
    s1 = (s1 + buffer[n]) % 65521;
    s2 = (s2 + s1) % 65221;
  }

  return (s2 << 16) | s1;
}
