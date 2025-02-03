/**********************************************************************
 *
 * Copyright (c) 2019 Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither name of Kitware nor the names of any contributors may be used to
 *   endorse or promote products derived from this software without specific prior
 *   written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***********************************************************************/

#include "parflow.h"
#include "pfversion.h"
#include "amps.h"
#include "metadata.h"

#include "cJSON.h"

#include <string.h> // for memset

cJSON* metadata = NULL;
cJSON* js_inputs = NULL;
cJSON* js_outputs = NULL;
cJSON* js_parflow = NULL;
cJSON* js_domains = NULL;

// Print a message on rank 0 and return 0. Note that ##__VA_ARGS__ is
// used to allow an empty __VA_ARGS__ , which is not standard but is
// supported by most major compilers (clang, gcc, msvc).
#define METADATA_ERROR(message, ...)                               \
        {                                                          \
          if (!amps_Rank(amps_CommWorld) && message && message[0]) \
          {                                                        \
            amps_Printf(message, ## __VA_ARGS__);                  \
          }                                                        \
          return 0;                                                \
        }

static void MetadataAddParflowBuildInfo(cJSON* pf)
{
  cJSON* build = cJSON_CreateObject();

  cJSON_AddItemToObject(pf, "build", build);
  cJSON_AddItemToObject(build, "version",
                        cJSON_CreateString(PARFLOW_VERSION_STRING));
  cJSON_AddItemToObject(build, "compiled",
                        cJSON_CreateString(__DATE__ " " __TIME__));
}

void MetadataAddParflowDomainInfo(cJSON* pf, PFModule* solver, Grid* localGrid)
{
  (void)solver;
  Background* bg = GlobalsBackground;
  int extent[3] = {
    BackgroundNX(bg),
    BackgroundNY(bg),
    BackgroundNZ(bg)
  };
  double spacing[3] = {
    BackgroundDX(bg),
    BackgroundDY(bg),
    BackgroundDZ(bg)
  };
  double subOrigin[3] = {
    BackgroundX(bg),
    BackgroundY(bg),
    BackgroundZ(bg)
  };
  double topOrigin[3] = {
    BackgroundX(bg),
    BackgroundY(bg),
    BackgroundZ(bg) + spacing[2] * extent[2]
  };

  int ni = GetInt("Process.Topology.P");
  int nj = GetInt("Process.Topology.Q");
  int nk = GetInt("Process.Topology.R");

  int my_rank = amps_Rank(amps_CommWorld);
  int rr = my_rank;
  // int ss = amps_node_size;
  int* idivs = (int*)malloc(sizeof(int) * ni);
  int* jdivs = (int*)malloc(sizeof(int) * nj);
  int* kdivs = (int*)malloc(sizeof(int) * nk);
  int* gidivs = rr == 0 ? (int*)malloc(sizeof(int) * (ni + 1)) : NULL;
  int* gjdivs = rr == 0 ? (int*)malloc(sizeof(int) * (nj + 1)) : NULL;
  int* gkdivs = rr == 0 ? (int*)malloc(sizeof(int) * (nk + 1)) : NULL;
  int xi = (rr < ni ? rr : -1);
  int yj = (rr / ni < nj && rr % ni == 0 ? rr / ni : -1);
  int zk = (rr / ni / nj < nk && rr % (ni * nj) == 0 ? rr / ni / nj : -1);

  SubgridArray* subgrids = GridSubgrids(localGrid);
  Subgrid* subgrid = NULL;
  int g;
  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
  }

  int origin_i = SubgridIX(subgrid);
  int origin_j = SubgridIY(subgrid);
  int origin_k = SubgridIZ(subgrid);

  cJSON* topsurf = cJSON_CreateObject();
  cJSON_AddItemToObject(pf, "surface", topsurf);
  cJSON_AddItemToObject(topsurf, "cell-extent",
                        cJSON_CreateIntArray(extent, 2));
  cJSON_AddItemToObject(topsurf, "spacing",
                        cJSON_CreateDoubleArray(spacing, 3));
  cJSON_AddItemToObject(topsurf, "origin",
                        cJSON_CreateDoubleArray(topOrigin, 3));
  cJSON* topSGD = cJSON_CreateArray();
  cJSON_AddItemToObject(topsurf, "subgrid-divisions", topSGD);

  cJSON* subsurf = cJSON_CreateObject();
  cJSON_AddItemToObject(pf, "subsurface", subsurf);
  cJSON_AddItemToObject(subsurf, "cell-extent",
                        cJSON_CreateIntArray(extent, 3));
  cJSON_AddItemToObject(subsurf, "spacing",
                        cJSON_CreateDoubleArray(spacing, 3));
  cJSON_AddItemToObject(subsurf, "origin",
                        cJSON_CreateDoubleArray(subOrigin, 3));
  cJSON* subSGD = cJSON_CreateArray();
  cJSON_AddItemToObject(subsurf, "subgrid-divisions", subSGD);

  memset(idivs, 0, sizeof(int) * ni);
  memset(jdivs, 0, sizeof(int) * nj);
  memset(kdivs, 0, sizeof(int) * nk);
  /* Have nodes along axes fill in offsets; others leave empty */
  if (xi >= 0)
  {
    idivs[xi] = origin_i;
  }
  if (yj >= 0)
  {
    jdivs[yj] = origin_j;
  }
  if (zk >= 0)
  {
    kdivs[zk] = origin_k;
  }

#ifdef PARFLOW_HAVE_MPI
  /* Optimization would be to make this a single reduction operation */
  MPI_Reduce(idivs, gidivs, ni, MPI_INT, MPI_SUM, 0, amps_CommWorld);
  MPI_Reduce(jdivs, gjdivs, nj, MPI_INT, MPI_SUM, 0, amps_CommWorld);
  MPI_Reduce(kdivs, gkdivs, nk, MPI_INT, MPI_SUM, 0, amps_CommWorld);
#else
  /* This is broken for parallel layers other than MPI.  AMPS does not
   * have a Reduce operation; only ReduceAll */
  for (int i = 0; i < ni; i++)
  {
    gidivs[i] = idivs[i];
  }

  for (int j = 0; j < nj; j++)
  {
    gjdivs[j] = jdivs[j];
  }

  for (int k = 0; k < nk; k++)
  {
    gkdivs[k] = kdivs[k];
  }
#endif

  if (my_rank == 0)
  {
    /* now add trailing entry for "final" offset: */
    gidivs[ni] = extent[0];
    gjdivs[nj] = extent[1];
    gkdivs[nk] = extent[2];

    cJSON_AddItemToArray(topSGD, cJSON_CreateIntArray(gidivs, ni + 1));
    cJSON_AddItemToArray(topSGD, cJSON_CreateIntArray(gjdivs, nj + 1));

    cJSON_AddItemToArray(subSGD, cJSON_CreateIntArray(gidivs, ni + 1));
    cJSON_AddItemToArray(subSGD, cJSON_CreateIntArray(gjdivs, nj + 1));
    cJSON_AddItemToArray(subSGD, cJSON_CreateIntArray(gkdivs, nk + 1));
  }

  free(idivs);
  free(jdivs);
  free(kdivs);
  free(gidivs);
  free(gjdivs);
  free(gkdivs);
}

static void cJSON_AddIDBEntries(cJSON* json, HBT_element* info, int onlyUsed)
{
  if (!info)
  {
    return;
  }

  cJSON_AddIDBEntries(json, info->left, onlyUsed);
  IDB_Entry* entry = (IDB_Entry*)info->obj;
  if (entry && entry->key && entry->value && (!onlyUsed || entry->used))
  {
    cJSON_AddItemToObject(json, entry->key, cJSON_CreateString(entry->value));
  }
  cJSON_AddIDBEntries(json, info->right, onlyUsed);
}

static cJSON* cJSON_CreateIDBDict(IDB* info, int onlyUsed)
{
  cJSON* dict = cJSON_CreateObject();

  cJSON_AddIDBEntries(dict, info->root, onlyUsed);
  return dict;
}

static void MetadataInitParflowInputInfo(cJSON* pf, PFModule* solver)
{
  (void)solver;
  IDB* pfidb = amps_ThreadLocal(input_database);
  cJSON* cfg = cJSON_CreateObject();

  /* If the user specified an elevation field, record its location
   * so post-processing tools can use it even though the simulation
   * instead relies on slopes computed from elevation.
   *
   * Keep this above the cJSON_CreateIDBDict() call so the
   * dictionary entry is recorded.
   */
  const char* elevFileName = GetStringDefault("TopoSlopes.Elevation.FileName", "none");
  if (elevFileName && strncmp("none", elevFileName, 5) != 0)
  {
    cJSON* field_item = cJSON_CreateObject();
    cJSON* field_files = cJSON_CreateArray();
    cJSON_AddItemToObject(pf, "elevation", field_item);
    cJSON_AddItemToObject(field_item, "type", cJSON_CreateString("pfb"));
    cJSON_AddItemToObject(field_item, "place", cJSON_CreateString("cell"));
    cJSON_AddItemToObject(field_item, "domain", cJSON_CreateString("surface"));
    // TODO: Could add units here if user specifies them somehow.
    cJSON_AddItemToObject(field_item, "data", field_files);
    cJSON* file_descr = cJSON_CreateObject();
    cJSON_AddItemToArray(field_files, file_descr);
    cJSON_AddItemToObject(file_descr, "file", cJSON_CreateString(elevFileName));
  }

  cJSON_AddItemToObject(pf, "configuration", cfg);
  cJSON_AddItemToObject(cfg, "type", cJSON_CreateString("pfidb"));
  cJSON_AddItemToObject(cfg, "data", cJSON_CreateIDBDict(pfidb, /* onlyUsed */ 0));

  // TODO: Introspect solver module to understand more of the fields present?
}

static void MetadataInitParflowOutputInfo(cJSON* pf, PFModule* solver)
{
  (void)pf;
  (void)solver;
  // TODO: Introspect solver module to understand the fields present?
}

static void MetadataFiniParflowInputInfo(cJSON* pf, PFModule* solver)
{
  (void)pf;
  (void)solver;
}

static void MetadataFiniParflowOutputInfo(cJSON* pf, PFModule* solver)
{
  (void)pf;
  (void)solver;
}

static void DeleteMetadata()
{
  cJSON_Delete(metadata);
  js_inputs = NULL;
  js_outputs = NULL;
  js_parflow = NULL;
  js_domains = NULL;
}

void NewMetadata(PFModule* solver)
{
  if (metadata)
  {
    DeleteMetadata();
  }

  metadata = cJSON_CreateObject();
  js_parflow = cJSON_CreateObject();
  js_domains = cJSON_CreateObject();
  js_inputs = cJSON_CreateObject();
  js_outputs = cJSON_CreateObject();
  cJSON_AddItemToObject(metadata, "parflow", js_parflow);
  cJSON_AddItemToObject(metadata, "domains", js_domains);
  cJSON_AddItemToObject(metadata, "inputs", js_inputs);
  cJSON_AddItemToObject(metadata, "outputs", js_outputs);

  MetadataAddParflowBuildInfo(js_parflow);
  // MetadataAddParflowDomainInfo(js_domains, solver);
  MetadataInitParflowInputInfo(js_inputs, solver);
  MetadataInitParflowOutputInfo(js_outputs, solver);
}

void WriteMetadata(PFModule* solver, const char* prefix)
{
  // Only rank 0 should write the metadata summary:
  if (!amps_Rank(MPI_CommWorld))
  {
    // First, write to a dummy file:
    char* json = cJSON_Print(metadata);
    FILE* meta = fopen("tmp.pfmetadata", "w");
    fprintf(meta, "%s", json);
    fclose(meta);
    free(json);
    // Now move the dummy file into place.
    // Doing things this way ensures no one can open a
    // partially written file (which would produce JSON
    // parse errors).
    char mfname[2048];
    snprintf(mfname, 2047, "%s.pfmetadata", prefix);
    rename("tmp.pfmetadata", mfname);
  }
}

void UpdateMetadata(PFModule* solver, const char* prefix, int ts)
{
  (void)ts;
  WriteMetadata(solver, prefix);
}

void FinalizeMetadata(PFModule* solver, const char* prefix)
{
  MetadataFiniParflowInputInfo(js_inputs, solver);
  MetadataFiniParflowOutputInfo(js_outputs, solver);

  WriteMetadata(solver, prefix);
  DeleteMetadata();
}

int MetaDataHasField(cJSON* node, const char* fieldName)
{
  return cJSON_GetObjectItem(node, fieldName) == NULL;
}

int MetadataAddStaticField(
                           MetadataItem parent,
                           const char*  file_prefix,
                           const char*  field_name,
                           const char*  field_units,
                           const char*  field_placement,
                           const char*  field_domain,
                           int          num_field_components,
                           const char** field_component_postfixes)
{
  int ii;

  if (
      num_field_components <= 0 ||
      !file_prefix || !field_name ||
      !field_placement || !field_domain ||
      !field_component_postfixes)
  {
    return 0;
  }
  cJSON* field_item = cJSON_CreateObject();
  cJSON* field_files = cJSON_CreateArray();
  cJSON_AddItemToObject(parent, field_name, field_item);
  cJSON_AddItemToObject(field_item, "type", cJSON_CreateString("pfb"));
  cJSON_AddItemToObject(field_item, "place", cJSON_CreateString(field_placement));
  cJSON_AddItemToObject(field_item, "domain", cJSON_CreateString(field_domain));
  if (field_units && field_units[0])
  {
    cJSON_AddItemToObject(field_item, "units", cJSON_CreateString(field_units));
  }
  cJSON_AddItemToObject(field_item, "data", field_files);
  for (ii = 0; ii < num_field_components; ++ii)
  {
    cJSON* file_descr = cJSON_CreateObject();
    cJSON_AddItemToArray(field_files, file_descr);
    char temp[2048];
    snprintf(temp, 2047, "%s.%s.pfb", file_prefix, field_component_postfixes[ii]);
    cJSON_AddItemToObject(file_descr, "file", cJSON_CreateString(temp));
    if (num_field_components > 1)
    {
      if (num_field_components < 4)
      {
        temp[0] = 'x' + ii;
        temp[1] = '\0';
      }
      else
      {
        snprintf(temp, 2047, "%d", ii);
      }
      cJSON_AddItemToObject(file_descr, "component", cJSON_CreateString(temp));
    }
  }
  return 1;
}

int MetadataAddDynamicField(
                            MetadataItem parent,
                            const char*  file_prefix,
                            double       time,
                            int          step,
                            const char*  field_name,
                            const char*  field_units,
                            const char*  field_placement,
                            const char*  field_domain,
                            int          num_field_components,
                            const char** field_component_postfixes)
{
  (void)time;

  int ii;
  if (
      !file_prefix || !field_name ||
      !field_placement || !field_domain)
  {
    return 0;
  }
  cJSON* field_item = cJSON_GetObjectItem(parent, field_name);
  if (!field_item)
  { // We truly are adding the field for the first time:
    if (
        num_field_components <= 0 ||
        !field_component_postfixes)
    {
      fprintf(stderr,
              "No components provided for initial addition of field \"%s\"\n",
              field_name);
      return 0;
    }
    field_item = cJSON_CreateObject();
    cJSON* field_files = cJSON_CreateArray();
    cJSON_AddItemToObject(parent, field_name, field_item);
    cJSON_AddItemToObject(field_item, "type", cJSON_CreateString("pfb"));
    cJSON_AddItemToObject(field_item, "place", cJSON_CreateString(field_placement));
    cJSON_AddItemToObject(field_item, "domain", cJSON_CreateString(field_domain));
    cJSON_AddItemToObject(field_item, "time-varying", cJSON_CreateTrue());
    if (field_units && field_units[0])
    {
      cJSON_AddItemToObject(field_item, "units", cJSON_CreateString(field_units));
    }
    cJSON_AddItemToObject(field_item, "data", field_files);
    for (ii = 0; ii < num_field_components; ++ii)
    {
      cJSON* file_descr = cJSON_CreateObject();
      cJSON* time_range = cJSON_CreateArray();
      cJSON_AddItemToArray(field_files, file_descr);
      char temp[2048];
      snprintf(temp, 2047, "%s.%s.%%05d.pfb", file_prefix, field_component_postfixes[ii]);
      cJSON_AddItemToObject(file_descr, "file-series", cJSON_CreateString(temp));
      cJSON_AddItemToObject(file_descr, "time-range", time_range);
      if (num_field_components > 1)
      {
        if (num_field_components < 4)
        {
          temp[0] = 'x' + ii;
          temp[1] = '\0';
        }
        else
        {
          snprintf(temp, 2047, "%d", ii);
        }
        cJSON_AddItemToObject(file_descr, "component", cJSON_CreateString(temp));
      }
      cJSON_AddItemToArray(time_range, cJSON_CreateNumber(step));
      cJSON_AddItemToArray(time_range, cJSON_CreateNumber(step));
    }
  }
  else
  { // Assume we are adding a time step; other metadata is used to verify sanity.
    cJSON* checkType;
    cJSON* checkPlace;
    cJSON* checkDomain;
    cJSON* checkTimeVarying;
    cJSON* fdata;
    cJSON* fentry;
    if (
        !(checkType = cJSON_GetObjectItem(field_item, "type")) ||
        checkType->type != cJSON_String ||
        !checkType->valuestring ||
        strcmp(checkType->valuestring, "pfb"))
    {
      fprintf(stderr, "Trying to change type from %s to pfb\n", checkType->valuestring);
      return 0;
    }

    if (
        !(checkPlace = cJSON_GetObjectItem(field_item, "place")) ||
        checkPlace->type != cJSON_String ||
        !checkPlace->valuestring ||
        strcmp(checkPlace->valuestring, field_placement))
    {
      fprintf(stderr, "Trying to change type from \"%s\" to \"%s\"\n",
              checkPlace->valuestring, field_placement);
      return 0;
    }

    if (
        !(checkDomain = cJSON_GetObjectItem(field_item, "domain")) ||
        checkDomain->type != cJSON_String ||
        !checkDomain->valuestring ||
        strcmp(checkDomain->valuestring, field_domain))
    {
      fprintf(stderr, "Trying to change type from \"%s\" to \"%s\"\n",
              checkDomain->valuestring, field_domain);
      return 0;
    }

    if (
        !(checkTimeVarying = cJSON_GetObjectItem(field_item, "time-varying")) ||
        checkTimeVarying->type != cJSON_True)
    {
      fprintf(stderr, "Trying to change field from static to time-varying\n");
      return 0;
    }

    if (
        !(fdata = cJSON_GetObjectItem(field_item, "data")) ||
        fdata->type != cJSON_Array)
    {
      fprintf(stderr, "File data not present or wrong type\n");
      return 0;
    }

    // Now, for each file in the series, update the end time to include "step"
    for (fentry = fdata->child; fentry; fentry = fentry->next)
    {
      cJSON* times;
      if (
          fentry->type != cJSON_Object ||
          !cJSON_GetObjectItem(fentry, "file-series") ||
          !(times = cJSON_GetObjectItem(fentry, "time-range")) ||
          times->type != cJSON_Array ||
          !times->child || !times->child->next ||
          times->child->next->type != cJSON_Number
          )
      {
        continue;
      }
      // FIXME: Handle skipped-step and variable-size steps/dumps?

      // This is kinda hackish, but much more efficient than cJSON_ReplaceItemInArray():
      times->child->next->valueint = step;
      times->child->next->valuedouble = (double)step;
    }
  }
  return 1;
}

// Unlike other MetadataAdd.*Field methods, this one knows
// ahead of time which time steps are available (or at least
// have been specified as available).
int MetadataAddForcingField(
                            cJSON*       parent,
                            const char*  field_name,
                            const char*  field_units,
                            const char*  field_placement,
                            const char*  field_domain,
                            int          clm_metforce,
                            int          clm_metsub,
                            const char*  clm_metpath,
                            const char*  clm_metfile,
                            int          clm_istep_start,
                            int          clm_fstep_start,
                            int          clm_metnt,
                            size_t       num_field_components,
                            const char** field_component_postfixes
                            )
{
  int ii;

  if (!field_name)
  {
    METADATA_ERROR(
                   "Unable to add metadata for null field.\n");
  }
  if (num_field_components <= 0)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; invalid components (%d).\n",
                   field_name, num_field_components);
  }
  if (!clm_metfile)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; no CLM Met file.\n", field_name);
  }
  if (!clm_metpath)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; no CLM Met path.\n", field_name);
  }
  if (!field_placement)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; null field placement.\n",
                   field_name);
  }
  if (!field_domain)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; null field domain.\n", field_name);
  }
  if (!field_component_postfixes)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; null field component_postfixes.\n",
                   field_name);
  }
  if (clm_metforce < 2 || clm_metforce > 3)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; Unhandled CLM Met forcing %d.\n",
                   field_name, clm_metforce);
  }
  if (clm_metnt <= 0)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; %d timesteps provided (> 0 required).\n",
                   field_name, clm_metnt);
  }
  if (clm_istep_start > clm_metnt)
  {
    METADATA_ERROR(
                   "Unable to add metadata for \"%s\"; "
                   "start time (%d) beyond forcing timesteps provided (%d).\n",
                   field_name, clm_istep_start, clm_metnt);
  }

  cJSON* field_item = cJSON_CreateObject();
  cJSON* field_files = cJSON_CreateArray();
  cJSON_AddItemToObject(parent, field_name, field_item);
  cJSON_AddItemToObject(field_item, "type", cJSON_CreateString(clm_metforce == 2 ? "pfb" : "pfb 2d timeseries"));
  cJSON_AddItemToObject(field_item, "time-varying", cJSON_CreateTrue());
  cJSON_AddItemToObject(field_item, "place", cJSON_CreateString(field_placement));
  cJSON_AddItemToObject(field_item, "domain", cJSON_CreateString(field_domain));
  if (field_units && field_units[0])
  {
    cJSON_AddItemToObject(field_item, "units", cJSON_CreateString(field_units));
  }
  cJSON_AddItemToObject(field_item, "data", field_files);
  for (ii = 0; ii < num_field_components; ++ii)
  {
    const char* time_key = NULL;
    cJSON* time_descr;
    cJSON* file_descr = cJSON_CreateObject();
    cJSON_AddItemToArray(field_files, file_descr);
    char temp[2048];
    if (clm_metforce == 2)
    {
      int timeRange[] = { clm_istep_start, clm_istep_start };
      time_key = "time-range";
      time_descr = cJSON_CreateIntArray(timeRange, 2);
      if (clm_metsub)
      {
        snprintf(temp, 2047, "%s/%s/%s.%s.%%06d.pfb",
                 clm_metpath, field_component_postfixes[ii],
                 clm_metfile, field_component_postfixes[ii]
                 );
      }
      else
      {
        snprintf(temp, 2047, "%s/%s.%s.%%06d.pfb",
                 clm_metpath, clm_metfile, field_component_postfixes[ii]);
      }
    }
    else // if clm_metforce == 3
    {
      int timesBetween[] = { clm_fstep_start, clm_istep_start, clm_metnt };
      time_key = "times-between";
      time_descr = cJSON_CreateIntArray(timesBetween, 3);
      if (clm_metsub)
      {
        snprintf(temp, 2047, "%s/%s/%s.%s.%%06d_to_%%06d.pfb",
                 clm_metpath, field_component_postfixes[ii],
                 clm_metfile, field_component_postfixes[ii]
                 );
      }
      else
      {
        snprintf(temp, 2047, "%s/%s.%s.%%06d_to_%%06d.pfb",
                 clm_metpath, clm_metfile, field_component_postfixes[ii]
                 );
      }
    }
    cJSON_AddItemToObject(file_descr, "file-series", cJSON_CreateString(temp));
    if (time_key)
    {
      cJSON_AddItemToObject(file_descr, time_key, time_descr);
    }
    if (num_field_components > 1)
    {
      if (num_field_components < 4)
      {
        temp[0] = 'x' + ii;
        temp[1] = '\0';
      }
      else
      {
        snprintf(temp, 2047, "%d", ii);
      }
      cJSON_AddItemToObject(file_descr, "component", cJSON_CreateString(temp));
    }
  }
  return 1;
}

int MetadataUpdateForcingField(
                               cJSON*      parent,
                               const char* field_name,
                               int         update_timestep
                               )
{
  if (!parent || !field_name)
  {
    return 0;
  }
  cJSON* field_item = cJSON_GetObjectItem(parent, field_name);
  if (!field_item)
  {
    return 0;
  }

  cJSON* fdata;
  cJSON* fentry;
  if (
      !(fdata = cJSON_GetObjectItem(field_item, "data")) ||
      fdata->type != cJSON_Array)
  {
    fprintf(stderr, "File data not present or wrong type\n");
    return 0;
  }

  // Now, for each file in the series, update the end time to include "step"
  for (fentry = fdata->child; fentry; fentry = fentry->next)
  {
    cJSON* times;
    if (
        fentry->type != cJSON_Object ||
        !cJSON_GetObjectItem(fentry, "file-series"))
    {
      continue;
    }
    // FIXME: Handle skipped-step and variable-size steps/dumps?
    if (
        (times = cJSON_GetObjectItem(fentry, "time-range")) &&
        times->type == cJSON_Array &&
        times->child && times->child->next &&
        times->child->type == cJSON_Number &&
        times->child->next->type == cJSON_Number
        )
    {
      // This is kinda hackish, but much more efficient than cJSON_ReplaceItemInArray():
      if (update_timestep < times->child->valueint)
      {
        times->child->valueint = update_timestep;
        times->child->valuedouble = (double)update_timestep;
      }
      else if (update_timestep > times->child->next->valueint)
      {
        times->child->next->valueint = update_timestep;
        times->child->next->valuedouble = (double)update_timestep;
      }
    }
    else if (
             (times = cJSON_GetObjectItem(fentry, "times-between")) &&
             times->type == cJSON_Array &&
             times->child && times->child->next && times->child->next->next &&
             times->child->type == cJSON_Number &&
             times->child->next->type == cJSON_Number
             )
    {
      // This is kinda hackish, but much more efficient than cJSON_ReplaceItemInArray():
      if (update_timestep < times->child->valueint)
      {
        times->child->valueint = update_timestep;
        times->child->valuedouble = (double)update_timestep;
      }
      else if (update_timestep > times->child->next->valueint)
      {
        times->child->next->valueint = update_timestep;
        times->child->next->valuedouble = (double)update_timestep;
      }
    }
  }
  return 1;
}
