#include "melissa.h"

int MELISSA_ACTIVE;


#ifdef HAVE_MELISSA

#include <melissa_api.h>

static int melissa_simu_id;


int getVectSize(Vector const * const vec)
{
  Grid *grid = VectorGrid(vec);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subgrid *subgrid;
  int g;

  // get last subgrid...
  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
  }

  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);

  D("Vect Size: %d", nx*ny*nz);

  return nx * ny * nz;
}

void MelissaInit(Vector const * const pressure,
Vector const * const saturation,
Vector const * const evap_trans_sum)
{
  const int rank = amps_Rank(amps_CommWorld);
  const int num = amps_Size(amps_CommWorld);
  MPI_Comm comm = amps_CommWorld;
  int coupling = MELISSA_COUPLING_ZMQ;
  int local_vect_size;
  local_vect_size = getVectSize(pressure);
  melissa_init("pressure", &local_vect_size, &num, &rank, &melissa_simu_id, &comm,
               &coupling);
  local_vect_size = getVectSize(saturation);
  melissa_init("saturation", &local_vect_size, &num, &rank, &melissa_simu_id, &comm,
               &coupling);
  //const int local_vect_size2D = nx * ny;
  //melissa_init("evap_trans_sum", &local_vect_size2D, &num, &rank, &melissa_simu_id, &comm,
  //             &coupling);
  local_vect_size = getVectSize(evap_trans_sum);
  melissa_init("evap_trans_sum", &local_vect_size, &num, &rank, &melissa_simu_id, &comm, &coupling);

  local_vect_size = 1;
  melissa_init("subsurf_storage", &local_vect_size, &num, &rank, &melissa_simu_id, &comm, &coupling);

  D("melissa initialized.");
}

void send_vec(const char * name, Vector const * const vec)
{
  Grid *grid = VectorGrid(vec);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subvector *subvector;
  Subgrid *subgrid;
  int g;

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(vec, g);
  }

  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);
  int nx = SubgridNX(subgrid);
  int ny = SubgridNY(subgrid);
  int nz = SubgridNZ(subgrid);
  //if (0 == strcmp("evap_trans_sum", name))
  //{
    //nz = 1;
  //}


  int nx_v = SubvectorNX(subvector);
  int ny_v = SubvectorNY(subvector);

  // some iterators
  int i, j, k, ai = 0, d = 0;
  double buffer[nx * ny * nz];

  double *data;
  data = SubvectorElt(subvector, ix, iy, iz);

  //printf("shape: %d %d %d\n", nx, ny, nz);

  BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz, ai, nx_v, ny_v, nz_v, 1, 1, 1, {
    buffer[d] = data[ai]; d++;
  });
  // TODO: would be more performant if we could read the things not cell by cell I guess
  // REM: if plotting all the ai-s one sees that there are steps... ai does not increase
  // linear! ... but as there are maybe patches that mask out some cells... etc...

  melissa_send(name, (double*)buffer);
}

int MelissaSend(const SimulationSnapshot * snapshot)
{
  send_vec("pressure", snapshot->pressure_out);
  send_vec("saturation", snapshot->saturation_out);
  //sendIt("evap_trans", snapshot->evap_trans);
  send_vec("evap_trans_sum", snapshot->evap_trans_sum);

  melissa_send("subsurf_storage", &snapshot->subsurf_storage);
  return 1;
}

void FreeMelissa(void)
{
  if (MELISSA_ACTIVE)
  {
    melissa_finalize();
  }
}

#endif
void NewMelissa(void)
{
  MELISSA_ACTIVE = GetBooleanDefault("Melissa", 0);

  if (!MELISSA_ACTIVE)
  {
    return;
  }
#ifndef HAVE_MELISSA
  PARFLOW_ERROR("Parflow was not compiled with Melissa but Melissa in the input file was set to True");
  return;
#else
  if (strcmp(GetString("Solver"), "Richards") != 0)
  {
    PARFLOW_ERROR("To use as parflow with Melissa, the Richards solver must be chosen!");
    return;
  }

  melissa_simu_id = GetInt("Melissa.SimuID");
  D("Melissa running with simuid %d", melissa_simu_id);
#endif
}
