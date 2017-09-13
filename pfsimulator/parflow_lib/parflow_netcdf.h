/*BHEADER**********************************************************************
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

#ifdef PARFLOW_HAVE_NETCDF
#include <netcdf.h>
#include <netcdf_par.h>
#else
#define MAX_NC_VARS 8192
#endif

typedef struct
{
	char *varName;
	int ncType;
	int dimSize;
	int *dimIDs;
} varNCData;

void WritePFNC(char * file_prefix, char* file_postfix, double t, Vector  *v, int numVarTimeVariant,
			char *varName, int dimensionality, int timDimensionality);
void CreateNCFile(char *file_name, Vector *v);
void CloseNC(int ncID);
int LookUpInventory(char * varName, varNCData **myVarNCData);
void PutDataInNC(int varID, Vector *v, double t, varNCData *myVarNCData);
void find_variable_length( int nid, int varid, long dim_lengths[MAX_NC_VARS] );
void CreateNCFileNode(char *file_name, Vector *v);
void PutDataInNCNode(int varID, double *data_nc_node, int *nodeXIndices, int *nodeYIndices, int *nodeZIndices,
    			int *nodeXCount, int *nodeYCount, int *nodeZCount, double t, varNCData *myVarNCData);
void ReadPFNC(char *fileName, Vector *v, char *varName, int tStep);
void OpenNCFile(char *file_name, int *ncID);
void ReadNCFile(int ncID, int varID, Subvector *subvector, Subgrid *subgrid, char *varName, int tStep);
