#include "melissa.h"

int MELISSA_ACTIVE;


#ifdef HAVE_MELISSA

#define BUILD_WITH_MPI
#include <melissa_api.h>

static int melissa_simu_id;

void MelissaInit(Vector const * const pressure)
{
  // TODO: only pressure for now.
  Grid *grid = VectorGrid(pressure);
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


  const int local_vect_size = nx * ny * nz;
  const int rank = amps_Rank(amps_CommWorld);
  const int num = amps_Size(amps_CommWorld);
  MPI_Comm comm = amps_CommWorld;
  int coupling = MELISSA_COUPLING_FLOWVR;
  melissa_init("pressure", &local_vect_size, &num, &rank, &melissa_simu_id, &comm,
    &coupling);


  D("melissa initialized.");
}

int MelissaSend(Vector const * const pressure)
{
  Grid *grid = VectorGrid(pressure);
  SubgridArray *subgrids = GridSubgrids(grid);
  Subvector *subvector;
  Subgrid *subgrid;
  int g;

  ForSubgridI(g, subgrids)
  {
    subgrid = SubgridArraySubgrid(subgrids, g);
    subvector = VectorSubvector(pressure, g);
  }

  int ix = SubgridIX(subgrid);
  int iy = SubgridIY(subgrid);
  int iz = SubgridIZ(subgrid);

  //double const * const data = SubvectorElt(subvector, ix, iy, iz);
  double * data = SubvectorData(subvector);

  // TODO: How to know later which part of the array we got at which place?
  // how is the order of the ranks?
  // TODO: FIXME: possibly that is not in the good order here!
  melissa_send("pressure", (double*) data);
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


void FreeMelissa(void)
{
  if (MELISSA_ACTIVE)
  {
    melissa_finalize();
  }
}
