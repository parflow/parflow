/*****************************************************************************
*
* Lattice Darcy solver equation solver.
*
*
* (C) 1993 Regents of the University of California.
*
* $Revision: 1.1.1.1 $
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/


/*--------------------*
* Include files
*--------------------*/
#include "parflow.h"


/**************************************************************************
* Lattice Boltzmann diffusion function
**************************************************************************/
void DiffuseLB(
               Lattice *lattice,
               Problem *problem,
               int      max_iterations,
               char *   file_prefix)
{
  /*------------------------------------------------------------*
  * Local variables
  *------------------------------------------------------------*/

  /* Lattice variables */
  Grid  *grid = (lattice->grid);
  double **e = (lattice->e);
  double *c = (lattice->c);
  Vector *pressure = (lattice->pressure);
  Vector *pwork = (lattice->pwork);
  Vector *perm = (lattice->perm);
  Vector *phi = (lattice->phi);
  double beta_pore = (lattice->beta_pore);
  double beta_fluid = (lattice->beta_fluid);
  double viscosity = (lattice->viscosity);
  CharVector *cellType = (lattice->cellType);
  double tscale = (lattice->tscale);
  double start = (lattice->start);
  double stop = (lattice->stop);
  double dump = (lattice->dump);
  double    *bforce = (lattice->bforce);
  double local_tstep = (lattice->step);

  /* Grid parameters */
  Subgrid   *subgrid;
  int ix, iy, iz;
  int nx, ny, nz;

  /* Physical variables and coefficients */
  Vector    *tmpVector;
  Subvector *sub_p;
  Subcharvector *sub_cellType;
  Subvector *sub_perm, *sub_phi;
  Subvector *sub_pwork;
  double    *pp, *phip;
  char      *cellTypep;
  double    *permp;
  double    *pworkp;
  double D, D0, maxD = 0.0;
  double compressibility;

  /* Indices */
  int i, j, k;
  int ii, jj, kk;
  int a, n_iter;
  int index, index_update;
  int gridloop;
  int pressure_file_number;
  int dump_switch;
  double write_pressure_time;
  int write_pressure_iteration;
  int iter_flag;

  /* miscellaneous */
  char file_postfix[80];
  double epsilon = 1.0e-10;
  double next_stop;

  /* Communications */
  VectorUpdateCommHandle *handle;

  (void)problem;

  /*------------------------------------------------------------*
  * Initialize some control parameters
  *------------------------------------------------------------*/
  amps_Printf("\n\n ... Running Lattice Diffusion Solver, v1.0\n\n");
  pressure_file_number = 1;
  n_iter = 0;
  (lattice->t) = start;
  dump_switch = 0;
  if (dump < 0)
  {
    dump_switch = 1;
    dump = -dump;
  }
  write_pressure_time = dump;
  write_pressure_iteration = (int)dump;
  iter_flag = 1;

  compressibility = beta_pore + beta_fluid;

  /*------------------------------------------------------------*
  * Main time loop
  *------------------------------------------------------------*/

  amps_Printf("Time step per iteration = %e\n\n", (lattice->step));
  amps_Printf("iter=%d; t=%e; dump=%e; stop=%e\n",
              n_iter, (lattice->t), write_pressure_time, (lattice->stop));

  while ((n_iter < max_iterations) && iter_flag && (lattice->t) < stop)
  {
    /* Propagate pore pressure; collisions occur here also */
    for (gridloop = 0; gridloop < GridNumSubgrids(grid); gridloop++)
    {
      subgrid = GridSubgrid(grid, gridloop);

      sub_p = VectorSubvector(pressure, gridloop);
      sub_perm = VectorSubvector(perm, gridloop);
      sub_phi = VectorSubvector(phi, gridloop);
      sub_cellType = CharVectorSubcharvector(cellType, gridloop);
      sub_pwork = VectorSubvector(pwork, gridloop);
      pp = SubvectorData(sub_p);
      permp = SubvectorData(sub_perm);
      phip = SubvectorData(sub_phi);
      cellTypep = SubcharvectorData(sub_cellType);
      pworkp = SubvectorData(sub_pwork);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      /* Update the pressures */
      for (i = ix; i < ix + nx; i++)
        for (j = iy; j < iy + ny; j++)
          for (k = iz; k < iz + nz; k++)
          {
            index = SubvectorEltIndex(sub_p, i, j, k);

            pworkp[index] = pp[index];
            if ((cellTypep[index]))
            {
              pworkp[index] = 0.0;
              D0 = 0.0;
              for (a = 1; a < nDirections; a++)
              {
                ii = i - (int)e[a][0];
                jj = j - (int)e[a][1];
                kk = k - (int)e[a][2];
                index_update = SubvectorEltIndex(sub_p, ii, jj, kk);

                /* Update the pore pressure */
                D = c[a] * tscale * permp[index_update];
                D /= (viscosity * compressibility * phip[index]);
                if (D > maxD)
                  maxD = D;

                D0 += D;
                pworkp[index] += D * (pp[index_update] + bforce[a]);
              }
              D0 = 1.0 - D0;

              pworkp[index] += D0 * (pp[index]);
              /* Add pressure change due to irreversible
               * processes, such as porosity reduction. */
            }
          }
    }   /* gridloop */

    /* Switch 'em */
    tmpVector = pressure;
    pressure = pwork;
    pwork = tmpVector;

    /* Update the boundary layers */
    handle = InitVectorUpdate(pressure, VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    (lattice->t) += local_tstep;
    n_iter++;
    if (!(n_iter % 100))
      amps_Printf("   ... iter=%d; t=%e; dump=%e; stop=%e\n",
                  n_iter, (lattice->t), write_pressure_time, (lattice->stop));

    /* If dump_switch == 1, write out the pressure field
     * every so many iterations. */
    if (dump_switch)
    {
      if (n_iter >= write_pressure_iteration)
      {
        /* Write out the pressure */
        sprintf(file_postfix, "press.%05d", pressure_file_number++);
        amps_Printf("Writing %s.%s.pfb at t=%f\n", file_prefix, file_postfix, (lattice->t));
        WritePFBinary(file_prefix, file_postfix, pressure);
        write_pressure_iteration += (int)dump;
      }
    }

    /* Otherwise, we're writing out the pressure field
     * on a *time* schedule. */
    else
    {
      if ((lattice->t) >= write_pressure_time - epsilon)
      {
        /* Write out the pressure */
        sprintf(file_postfix, "press.%05d", pressure_file_number++);
        amps_Printf("Writing %s.%s.pfb at t=%f\n", file_prefix, file_postfix, (lattice->t));
        WritePFBinary(file_prefix, file_postfix, pressure);
        write_pressure_time += dump;
      }
    }

    /* Check to see if it's time to stop */
    if ((lattice->t) >= stop)
    {
      iter_flag = 0;
    }

    else
    {
      /* next_stop is the minimum of stop and write_pressure_time */
      next_stop = write_pressure_time;
      if (next_stop > stop)
      {
        next_stop = stop;
        write_pressure_time = stop;
      }

      /* Reset local_tstep and tscale to original values */
      local_tstep = (lattice->step);
      tscale = (lattice->tscale);

      if ((lattice->t) + local_tstep > next_stop)
      {
        local_tstep = next_stop - (lattice->t);
        tscale = (lattice->tscale) * (local_tstep / (lattice->step));
      }
    }
  }  /* while */

  amps_Printf("\n=====>  Lattice Solver Finished  <=====\n");
}

/**************************************************************************
* LatticeFlowInit function
**************************************************************************/
void LatticeFlowInit(
                     Lattice *lattice,
                     Problem *problem)
{
  /*------------------------------------------------------------*
  * Local variables
  *------------------------------------------------------------*/

  /* Lattice variables */
  Grid  *grid = (lattice->grid);
  Vector *pressure = (lattice->pressure);
  Vector *perm = (lattice->perm);
  Vector *phi = (lattice->phi);
  double beta_perm = (lattice->beta_perm);
  double cfl = (lattice->cfl);
  double **e;
  double *c;
  double **K;
  double *Kscale;

  /* Grid parameters */
  Subgrid   *subgrid;
  double ds[3];

  /* Indices and dummy constants */
  int i, j, a;

  /* Physical variables and coefficients */
  double maxPerm_Phi, maxPressure;
  double L2min, L2max;
  double    *l_scale;
  double gravity[3];
  double rho;
  double viscosity, compressibility;
  double mu_beta;
  double s, sum, length;

  /*------------------------------------------------------------*
  * Allocate memory and initialize arrays
  *------------------------------------------------------------*/
  l_scale = ctalloc(double, nDirections);
  Kscale = ctalloc(double, nDirections);
  (lattice->Kscale) = Kscale;
  (lattice->bforce) = ctalloc(double, nDirections);
  c = ctalloc(double, nDirections);
  (lattice->c) = c;

  e = ctalloc(double*, nDirections);
  for (a = 0; a < nDirections; a++)
  {
    e[a] = ctalloc(double, 3);
  }
  (lattice->e) = e;

  /* Assign values to the permeability tensor.
   * Eventually, these should be inputs.
   * Assume K is orthonormal. */
  K = ctalloc(double*, 3);
  for (i = 0; i < 3; i++)
  {
    K[i] = ctalloc(double, 3);
    for (j = 0; j < 3; j++)
      K[i][j] = 0.0;
    K[i][i] = 1.0;
  }
  K[2][2] = 1.0;
  (lattice->Ktensor) = K;

  /* Weights for 4-dimensional hypercube projected onto 3-d lattice */
  if (nDirections == 7)
  {
    c[0] = 1.0 / 3.0;
    for (i = 1; i <= 6; i++)
      c[i] = 1.0 / 9.0;
  }

  else if (nDirections == 19)
  {
    c[0] = 1.0 / 3.0;
    for (i = 1; i < nDirections; i++)
      c[i] = 1.0 / 18.0;
  }

  /* Direction vectors */
  e[0][0] = 0;         e[0][1] = 0;    e[0][2] = 0;

  e[1][0] = 1;        e[1][1] = 0;   e[1][2] = 0;
  e[2][0] = -1;        e[2][1] = 0;   e[2][2] = 0;
  e[3][0] = 0;        e[3][1] = 1;   e[3][2] = 0;
  e[4][0] = 0;        e[4][1] = -1;   e[4][2] = 0;
  e[5][0] = 0;        e[5][1] = 0;   e[5][2] = 1;
  e[6][0] = 0;        e[6][1] = 0;   e[6][2] = -1;

  if (nDirections == 19)
  {
    e[7][0] = 1;     e[7][1] = 1;   e[7][2] = 0;
    e[8][0] = -1;     e[8][1] = -1;   e[8][2] = 0;
    e[9][0] = 1;     e[9][1] = -1;   e[9][2] = 0;
    e[10][0] = -1;    e[10][1] = 1;  e[10][2] = 0;

    e[11][0] = 1;    e[11][1] = 0;  e[11][2] = 1;
    e[12][0] = -1;    e[12][1] = 0;  e[12][2] = -1;
    e[13][0] = 1;    e[13][1] = 0;  e[13][2] = -1;
    e[14][0] = -1;    e[14][1] = 0;  e[14][2] = 1;

    e[15][0] = 0;    e[15][1] = 1;  e[15][2] = 1;
    e[16][0] = 0;    e[16][1] = -1;  e[16][2] = -1;
    e[17][0] = 0;    e[17][1] = 1;  e[17][2] = -1;
    e[18][0] = 0;    e[18][1] = -1;  e[18][2] = 1;
  }

  /*-------------------------------
   *  Compute anisotropy scaling, including length
   *  anisotropy and permeability tensor anisotropy.
   *-------------------------------*/
  subgrid = GridSubgrid(grid, 0);
  ds[0] = SubgridDX(subgrid);
  ds[1] = SubgridDY(subgrid);
  ds[2] = SubgridDZ(subgrid);

  L2min = ds[0] * ds[0];
  if (ds[1] * ds[1] < L2min)
    L2min = ds[1] * ds[1];
  if (ds[2] * ds[2] < L2min)
    L2min = ds[2] * ds[2];

  L2max = ds[0] * ds[0];
  if (ds[1] * ds[1] > L2max)
    L2max = ds[1] * ds[1];
  if (ds[2] * ds[2] > L2max)
    L2max = ds[2] * ds[2];

  /* length anisotropy */
  sum = 0.0;
  for (a = 1; a < nDirections; a++)
  {
    l_scale[a] = 0.0;
    for (i = 0; i < 3; i++)
    {
      s = e[a][i] * e[a][i] * ds[i] * ds[i];
      l_scale[a] += s;
    }
    c[a] *= L2min / (l_scale[a]);
    sum += c[a];
  }
  c[0] = 1.0 - sum;

  /* Permeability anisotropy */
  Kscale[0] = 1.0;
  for (a = 1; a < nDirections; a++)
  {
    Kscale[a] = 0.0;
    length = 0.0;
    for (i = 0; i < 3; i++)
    {
      length = 0.0;
      for (j = 0; j < 3; j++)
      {
        Kscale[a] += K[i][j] * e[a][i] * e[a][j] * ds[i] * ds[j];
        length += ds[i] * ds[j];
      }
    }
    /* This is temporary */
    Kscale[a] /= length;
    Kscale[a] = 1.0;
  }

  /*-------------------------------
   *  Get viscosity, compressibilities
   *-------------------------------*/
  compressibility = (lattice->beta_pore) + (lattice->beta_fluid);
  viscosity = (lattice->viscosity);

  /*-------------------------------------
   *  Find Maximum permeability and pressure
   *-------------------------------------*/
  maxPerm_Phi = MaxVectorDividend(perm, phi);
  maxPressure = MaxVectorValue(pressure);

  /* Compute time scale */
  mu_beta = viscosity * compressibility;
  (lattice->tscale) = cfl * mu_beta / (maxPerm_Phi * exp(beta_perm * maxPressure));
  (lattice->step) = L2min * (lattice->tscale);
  if (nDirections == 7)
    (lattice->step) /= 10;
  else if (nDirections == 19)
    (lattice->step) /= 10;

  /* Gravity: hydrostatic or lithostatic */
  gravity[0] = 0.0;
  gravity[1] = 0.0;
  gravity[2] = -ProblemGravity(problem);
  rho = RHO;

  for (a = 0; a < nDirections; a++)
  {
    (lattice->bforce)[a] = 0.0;
    for (i = 0; i < 3; i++)
    {
      (lattice->bforce)[a] += gravity[i] * rho * e[a][i] * ds[i];
    }
  }

  free(l_scale);
} /* LatticeFlowInit() */


/*--------------------------------------------------------------------------
 * MaxVectorValue
 *--------------------------------------------------------------------------*/

double  MaxVectorValue(
                       Vector *field)
{
  /*-----------------------
   * Local variables
   *-----------------------*/

  /* Grid variables */
  Grid         *grid;
  Subgrid      *subgrid;
  Subvector    *f_sub;
  double       *fp;
  double max_vector_value, tmp;

  /* Communications */
  amps_Invoice result_invoice;

  /* Indices */
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_f, ny_f, nz_f;
  int i_s, i, j, k, fi;

  /*-----------------------
   * Search for the maximum absolute
   * value in the field vector.
   *-----------------------*/
  max_vector_value = 0.0;

  grid = VectorGrid(field);
  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    f_sub = VectorSubvector(field, i_s);

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    fp = SubvectorElt(f_sub, ix, iy, iz);

    fi = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              fi, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      tmp = fabs(fp[fi]);
      if (tmp > max_vector_value)
        max_vector_value = tmp;
    });
  }

  /*--------------------------------------
   * Communicate results from grids and
   * return the maximum from all grids.
   *--------------------------------------*/
  result_invoice = amps_NewInvoice("%d", &max_vector_value);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);
  return max_vector_value;
} /* MaxVectorValue() */


/*--------------------------------------------------------------------------
 * MaxVectorDividend
 *--------------------------------------------------------------------------*/

double  MaxVectorDividend(
                          Vector *field1,
                          Vector *field2)
{
  /*-----------------------
   * Local variables
   *-----------------------*/

  /* Grid variables */
  Grid         *grid;
  Subgrid      *subgrid;
  Subvector    *f1_sub, *f2_sub;
  double       *f1p, *f2p;
  double max_dividend, tmp;

  /* Communications */
  amps_Invoice result_invoice;

  /* Indices */
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_f, ny_f, nz_f;
  int i_s, i, j, k, fi;

  /*-----------------------
   * Search for the maximum absolute
   * value in the field vector.
   *-----------------------*/
  max_dividend = 0.0;

  grid = VectorGrid(field1);
  ForSubgridI(i_s, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, i_s);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    f1_sub = VectorSubvector(field1, i_s);
    f2_sub = VectorSubvector(field2, i_s);

    nx_f = SubvectorNX(f1_sub);
    ny_f = SubvectorNY(f1_sub);
    nz_f = SubvectorNZ(f1_sub);

    f1p = SubvectorElt(f1_sub, ix, iy, iz);
    f2p = SubvectorElt(f2_sub, ix, iy, iz);

    fi = 0;
    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              fi, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      if (f2p[fi] == 0.0)
        tmp = 0.0;
      else
        tmp = fabs(f1p[fi] / f2p[fi]);

      if (tmp > max_dividend)
        max_dividend = tmp;
    });
  }

  /*--------------------------------------
   * Communicate results from grids and
   * return the maximum from all grids.
   *--------------------------------------*/
  result_invoice = amps_NewInvoice("%d", &max_dividend);
  amps_AllReduce(amps_CommWorld, result_invoice, amps_Max);
  amps_FreeInvoice(result_invoice);
  return max_dividend;
} /* MaxVectorDividend() */










