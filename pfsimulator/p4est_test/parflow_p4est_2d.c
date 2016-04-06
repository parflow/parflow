#include <parflow.h>
#include "parflow_p4est_math.h"
#ifndef P4_TO_P8
#include <parflow_p4est_2d.h>
#include <p4est_vtk.h>
#else
#include <parflow_p4est_3d.h>
#include <p8est_vtk.h>
#endif

static int
parflow_p4est_refine_fn (p4est_t * p4est, p4est_topidx_t which_tree,
                         p4est_quadrant_t * quadrant)
{
  return 1;
}

parflow_p4est_grid_t *
parflow_p4est_grid_2d_new (int NX, int NY
#ifdef P4_TO_P8
                           , int NZ
#endif
  )
{
  int                 g, gt;
  int                 tx, ty;

#ifdef P4_TO_P8
  int                 tz;
#endif
  int                 level, refine_level, balance;
  parflow_p4est_grid_t *pfgrid;

  pfgrid = P4EST_ALLOC_ZERO (parflow_p4est_grid_t, 1);
#ifdef P4_TO_P8
  pfgrid->dim = 3;
#else
  pfgrid->dim = 2;
#endif
  tx = pfmax (NX - 1, 1);
  ty = pfmax (NY - 1, 1);
  gt = gcd (tx, ty);
#ifdef P4_TO_P8
  tz = pfmax (NZ - 1, 1);
  gt = gcd (gt, tz);
#endif
  g = powtwo_div (gt);

  refine_level = (int) log2 ((double) g);
  balance = refine_level > 0;

  /*
   * Create connectivity structure
   */
  pfgrid->connect = p4est_connectivity_new_brick (tx / g, ty / g,
#ifdef P4_TO_P8
                                                  tz / g,
#endif
                                                  0, 0
#ifdef P4_TO_P8
                                                  , 0
#endif
    );

  pfgrid->forest = p4est_new (amps_CommWorld, pfgrid->connect, 0, NULL, NULL);

  /*
   * Refine to get a grid with same number of elements as parflow
   * old Grid structure and resdistribute quadrants among mpi_comm
   */
  for (level = 0; level < refine_level; ++level) {
    p4est_refine (pfgrid->forest, 0, parflow_p4est_refine_fn, NULL);
    p4est_partition (pfgrid->forest, 0, NULL);
  }

  /*
   * After refine, call 2:1 balance and redistribute new quadrants
   * among the mpi communicator
   */
  if (balance) {
    p4est_balance (pfgrid->forest, P4EST_CONNECT_FULL, NULL);
    p4est_partition (pfgrid->forest, 0, NULL);
  }

  p4est_vtk_write_file (pfgrid->forest, NULL, P4EST_STRING "_pfbrick");

  return pfgrid;
}

void
parflow_p4est_grid_2d_destroy (parflow_p4est_grid_t * pfgrid)
{
  p4est_destroy (pfgrid->forest);
  p4est_connectivity_destroy (pfgrid->connect);
  P4EST_FREE (pfgrid);
}

void
parflow_p4est_qcoord_to_vertex_2d (parflow_p4est_grid_t * pfgrid,
                                   p4est_topidx_t treeid,
                                   p4est_quadrant_t * quad, double v[3])
{

  p4est_qcoord_to_vertex (pfgrid->connect, treeid, quad->x, quad->y,
#ifdef P4_TO_P8
                          quad->z,
#endif
                          v);
}
