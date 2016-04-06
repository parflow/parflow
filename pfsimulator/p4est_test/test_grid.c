#include <stdio.h>
#include <parflow.h>
#include <parflow_p4est.h>

int
main (int argc, char **argv)
{
  int                   nx,ny,nz;
  Grid                 *grid;
  parflow_p4est_grid_t *pfgrid;

  if (amps_Init (&argc, &argv)) {
    amps_Printf ("Error: amps_Init initalization failed\n");
    exit (1);
  }

  NewGlobals (argv[1]);

  amps_ThreadLocal (input_database) = IDB_NewDB (GlobalsInFileName);

  GlobalsNumProcsX = GetIntDefault ("Process.Topology.P", 1);
  GlobalsNumProcsY = GetIntDefault ("Process.Topology.Q", 1);
  GlobalsNumProcsZ = GetIntDefault ("Process.Topology.R", 1);

  GlobalsNumProcs = amps_Size (amps_CommWorld);

  GlobalsBackground = ReadBackground ();

  GlobalsUserGrid = ReadUserGrid ();

  SetBackgroundBounds (GlobalsBackground, GlobalsUserGrid);

  grid = CreateGrid (GlobalsUserGrid);

  nx = GetIntDefault ("ComputationalGrid.NX", 1);
  ny = GetIntDefault ("ComputationalGrid.NY", 1);
  nz = GetIntDefault ("ComputationalGrid.NZ", 1);

  /*Initialize sc and p{4,casc8}est library */
  sc_init (amps_CommWorld, 1, 1, NULL, SC_LP_DEFAULT);
  p4est_init (NULL, SC_LP_DEFAULT);

  //pfgrid = parflow_p4est_grid_new (nx, ny, nz);

  //parflow_p4est_grid_destroy (pfgrid);

  PrintGrid ("pfgrid", grid);
  FreeGrid (grid);

  sc_finalize ();

  IDB_FreeDB (amps_ThreadLocal (input_database));
  FreeGlobals ();
  amps_Finalize ();

  return 0;
}
