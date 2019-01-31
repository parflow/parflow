#include <parflow.h>


static void    myInitVector(Vector *v)
{
  Grid       *grid = VectorGrid(v);
  int rank = amps_Rank(amps_CommWorld);
  Subvector  *v_sub;
  double     *vp;

  Subgrid    *subgrid;

  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;

  int i_s;
  int i, j, k, iv;


  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    v_sub = VectorSubvector(v, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    nx_v = SubvectorNX(v_sub);
    ny_v = SubvectorNY(v_sub);
    nz_v = SubvectorNZ(v_sub);

    vp = SubvectorElt(v_sub, ix, iy, iz);

    iv = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      vp[iv] = (double)(SubgridLocIdx(subgrid));
      //vp[iv] = (double)(1 << rank);
    });
  }
}

void parflow_p4est_vector_test(Grid *grid)
{
  Vector      *vector;
  VectorUpdateCommHandle  *handle;

  /* Create a vector */
  vector = NewVectorType(grid, 1, 1, vector_cell_centered);

  /* Put some values on it: including ghost points */
  InitVectorAll(vector, -1);

  /* Change the values in the interior nodes */
  myInitVector(vector);

  /* Try to do a parallel update*/
  handle = InitVectorUpdate(vector, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  /* Finalize everything */
  FreeVector(vector);
}
