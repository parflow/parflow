#include <netcdf.h>
#include <netcdf_par.h>

static int ncID, xID, yID, zID, timID, varID;
static int time_step = 0;
typedef struct
{
	char *varName;
	int ncType;
	int dimSize;
	int *dimIDs;
} varNCData;

void CreateNCFile(char *file_name, Vector *v);
void CloseNC(int ncID);
int LookUpInventory(char * varName, varNCData **myVarNCData);
void PutDataInNC(int varID, Vector *v, double t, varNCData *myVarNCData);
void find_variable_length( int nid, int varid, long dim_lengths[MAX_NC_VARS] );
void CreateNCFileNode(char *file_name, Vector *v);
void PutDataInNCNode(int varID, double *data_nc_node, int *nodeXIndices, int *nodeYIndices, int *nodeZIndices,
    			int *nodeXCount, int *nodeYCount, int *nodeZCount, double t, varNCData *myVarNCData);
void ReadPFNC(char *fileName, Vector *v);
void OpenNCFile(char *file_name, int *ncID);
void ReadNCFile(int ncID, int varID, Subvector *subvector, Subgrid *subgrid);
