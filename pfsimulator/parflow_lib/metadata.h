#ifndef pf_metadata_h
#define pf_metadata_h

#include "cJSON.h"

struct PFModule;
struct Grid;

// Hide cJSON from the public API so we can switch to other JSON libraries as needed.
typedef cJSON* MetadataItem;

extern MetadataItem metadata;
extern MetadataItem js_inputs;
extern MetadataItem js_outputs;
extern MetadataItem js_parflow;
extern MetadataItem js_domains;

void NewMetadata(PFModule* solver);
void MetadataAddParflowDomainInfo(MetadataItem pf, PFModule* solver, Grid* localGrid);
void UpdateMetadata(PFModule* solver, const char* prefix, int ts);
void FinalizeMetadata(PFModule* solver, const char* prefix);

int MetaDataHasField(MetadataItem node, const char* field_name);

int MetadataAddStaticField(
  MetadataItem parent,
  const char* file_prefix,
  const char* field_name,
  const char* field_units,
  const char* field_placement,
  const char* field_domain,
  int num_field_components,
  const char** field_component_postfixes);
int MetadataAddDynamicField(
  MetadataItem parent,
  const char* file_prefix,
  double t,
  int step,
  const char* field_name,
  const char* field_units,
  const char* field_placement,
  const char* field_domain,
  int num_field_components,
  const char** field_component_postfixes);
int MetadataAddForcingField(
  cJSON* parent,
  const char* field_name,
  const char* field_units,
  const char* field_place,
  const char* field_domain,
  int clm_metforce,
  int clm_metsub,
  const char* clm_metpath,
  const char* clm_metfile,
  int clm_istep_start,
  int clm_fstep_start,
  int clm_metnt,
  size_t num_field_components,
  const char** field_component_postfixes);
int MetadataUpdateForcingField(
  cJSON* parent,
  const char* field_name,
  int update_timestep
);

#endif /* pf_metadata_h */
