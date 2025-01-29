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
#ifndef pf_metadata_h
#define pf_metadata_h

#include "cJSON.h"

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
                           const char*  file_prefix,
                           const char*  field_name,
                           const char*  field_units,
                           const char*  field_placement,
                           const char*  field_domain,
                           int          num_field_components,
                           const char** field_component_postfixes);
int MetadataAddDynamicField(
                            MetadataItem parent,
                            const char*  file_prefix,
                            double       t,
                            int          step,
                            const char*  field_name,
                            const char*  field_units,
                            const char*  field_placement,
                            const char*  field_domain,
                            int          num_field_components,
                            const char** field_component_postfixes);
int MetadataAddForcingField(
                            cJSON*       parent,
                            const char*  field_name,
                            const char*  field_units,
                            const char*  field_place,
                            const char*  field_domain,
                            int          clm_metforce,
                            int          clm_metsub,
                            const char*  clm_metpath,
                            const char*  clm_metfile,
                            int          clm_istep_start,
                            int          clm_fstep_start,
                            int          clm_metnt,
                            size_t       num_field_components,
                            const char** field_component_postfixes);
int MetadataUpdateForcingField(
                               cJSON*      parent,
                               const char* field_name,
                               int         update_timestep
                               );

#endif /* pf_metadata_h */
