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
      vp[iv] = (double)(1 << SubgridLocIdx(subgrid));
      //vp[iv] = (double)(1 << rank);
    });
  }
}

void parflow_p4est_vector_test(Grid *grid)
{
  Vector      *vector;
  VectorUpdateCommHandle  *handle;
#if 0
  char file_prefix[2048], file_type[2048], file_postfix[2048];

  sprintf(file_prefix, "vecTest");
  sprintf(file_type, "press");
  WriteSiloInit(file_prefix);
#endif

  /* Create a vector, by default all its entries are zero */
  vector = NewVectorType(grid, 1, 1, vector_cell_centered);

  /* Change the values in the interior nodes */
  myInitVector(vector);

  PrintVector("vecTest_before", vector);

  /* Try to do a parallel update*/
  handle = InitVectorUpdate(vector, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

#if 0
  sprintf(file_postfix, "1");
  WriteSilo(file_prefix, file_type, file_postfix, vector, 0.0, 0, "justNumbers");
#endif

  PrintVector("vecTest_after", vector);

  /* Finalize everything */
  FreeVector(vector);

  /*Force to quit the program here*/
  amps_Printf("Dummy vector test complete\n");
  //exit(0);
}
