/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/

#include "parflow.h"
#include "llnlmath.h"
#include "llnltyps.h"

/*---------------------------------------------------------------------
 * Define module structures
 *---------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  Problem      *problem;

  PFModule     *density_module;
  PFModule     *saturation_module;
  PFModule     *rel_perm_module;
  PFModule     *bc_pressure;
  PFModule     *bc_internal;

  Vector       *density_der;
  Vector       *saturation_der;

  Matrix       *J;

  Grid         *grid;
  double       *temp_data;
} InstanceXtra;

/*--------------------------------------------------------------------------
 * Static stencil-shape definition
 *--------------------------------------------------------------------------*/

int jacobian_stencil_shape[7][3] = { { 0, 0, 0 },
                                     { -1, 0, 0 },
                                     { 1, 0, 0 },
                                     { 0, -1, 0 },
                                     { 0, 1, 0 },
                                     { 0, 0, -1 },
                                     { 0, 0, 1 } };


/*---------------------------------------------------------------------
 * Define macros for jacobian evaluation
 *---------------------------------------------------------------------*/

#define PMean(a, b, c, d)    HarmonicMean(c, d)
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*  This routine provides the interface between KINSOL and ParFlow
 *  for richards' equation jacobian evaluations and matrix-vector multiplies.*/

int       KINSolMatVec(current_state, x, y, recompute, pressure)
void     *current_state;
N_Vector x;
N_Vector y;
int      *recompute;
N_Vector pressure;
{
  PFModule    *richards_jacobian_eval = StateJacEval(((State*)current_state));
  Matrix      *J = StateJac(((State*)current_state));
  Vector      *saturation = StateSaturation(((State*)current_state));
  Vector      *density = StateDensity(((State*)current_state));
  ProblemData *problem_data = StateProblemData(((State*)current_state));
  double dt = StateDt(((State*)current_state));
  double time = StateTime(((State*)current_state));

  if (*recompute)
  {
    PFModuleInvoke(void, richards_jacobian_eval,
                   (pressure, &J, saturation, density, problem_data,
                    dt, time, 0));
  }

  Matvec(1.0, J, x, 0.0, y);

  return(0);
}


/*  This routine evaluates the Richards jacobian based on the current
 *  pressure values.  */

void    RichardsJacobianEval(pressure, ptr_to_J, saturation, density,
                             problem_data, dt, time, symm_part)
Vector * pressure;            /* Current pressure values */
Matrix      **ptr_to_J;       /* Pointer to the J pointer - this will be set
                               * to instance_xtra pointer at end */
Vector       *saturation;     /* Saturation / work vector */
Vector       *density;        /* Density vector */
ProblemData  *problem_data;   /* Geometry data for problem */
double dt;                    /* Time step size */
double time;                  /* New time value */
int symm_part;                /* Specifies whether to compute just the
                               * symmetric part of the Jacobian (1), or the
                               * full Jacobian */
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  Problem     *problem = (instance_xtra->problem);

  PFModule    *density_module = (instance_xtra->density_module);
  PFModule    *saturation_module = (instance_xtra->saturation_module);
  PFModule    *rel_perm_module = (instance_xtra->rel_perm_module);
  PFModule    *bc_pressure = (instance_xtra->bc_pressure);
  PFModule    *bc_internal = (instance_xtra->bc_internal);

  Matrix      *J = (instance_xtra->J);

  Vector      *density_der = (instance_xtra->density_der);
  Vector      *saturation_der = (instance_xtra->saturation_der);

  /* Re-use vectors to save memory */
  Vector      *rel_perm = saturation;
  Vector      *rel_perm_der = saturation_der;

  /* NEW vectors for overland flow */
  Vector      *KWD, *KED, *KND, *KSD;
  Vector      *qxd, *qyd;
  Subvector   *kwd_sub, *ked_sub, *knd_sub, *ksd_sub, *qxd_sub, *qyd_sub;
  Subvector   *x_sl_sub, *y_sl_sub, *mann_sub;
  double      *kwd_, *ked_, *knd_, *ksd_, *qxd_, *qyd_;
  double dir_x, dir_y;
  double      *x_sl_dat, *y_sl_dat, *mann_dat;
  double cx, cy;

  double press[58][30][360], pressbc[58][30], xslope[58][30], yslope[58][30], mans[58][30];

  Vector      *porosity = ProblemDataPorosity(problem_data);
  Vector      *permeability_x = ProblemDataPermeabilityX(problem_data);
  Vector      *permeability_y = ProblemDataPermeabilityY(problem_data);
  Vector      *permeability_z = ProblemDataPermeabilityZ(problem_data);
  Vector      *sstorage = ProblemDataSpecificStorage(problem_data);           //sk
  Vector      *x_sl = ProblemDataTSlopeX(problem_data);               //sk
  Vector      *y_sl = ProblemDataTSlopeY(problem_data);               //sk
  Vector      *man = ProblemDataMannings(problem_data);                //sk

  double gravity = ProblemGravity(problem);
  double viscosity = ProblemPhaseViscosity(problem, 0);

  Subgrid     *subgrid;

  Subvector   *p_sub, *d_sub, *s_sub, *po_sub, *rp_sub, *ss_sub;
  Subvector   *permx_sub, *permy_sub, *permz_sub, *dd_sub, *sd_sub, *rpd_sub;
  Submatrix   *J_sub;

  Grid        *grid = VectorGrid(pressure);

  double      *pp, *sp, *sdp, *pop, *dp, *ddp, *rpp, *rpdp;
  double      *permxp, *permyp, *permzp;
  double      *cp, *wp, *ep, *sop, *np, *lp, *up, *op, *ss;

  int i, j, k, r, is;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_m, ny_m, nz_m;
  int nx_po, ny_po, nz_po;
  int sy_v, sz_v;
  int sy_m, sz_m;
  int ip, ipo, im, iv, io;

  double dtmp, dx, dy, dz, vol, ffx, ffy, ffz;
  double diff, coeff, x_coeff, y_coeff, z_coeff;
  double prod, prod_rt, prod_no, prod_up, prod_val, prod_lo;
  double prod_der, prod_rt_der, prod_no_der, prod_up_der;
  double west_temp, east_temp, north_temp, south_temp;
  double lower_temp, upper_temp, o_temp;
  double sym_west_temp, sym_east_temp, sym_south_temp, sym_north_temp;
  double sym_lower_temp, sym_upper_temp;
  double lower_cond, upper_cond;

  BCStruct    *bc_struct;
  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);
  double      *bc_patch_values;
  double value, den_d, dend_d;
  int         *fdir;
  int ipatch, ival;

  CommHandle  *handle;

  /* Pass pressure values to neighbors.  */
  handle = InitVectorUpdate(pressure, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  KWD = NewVector(grid, 1, 1);
  InitVector(KWD, 0.0);

  KED = NewVector(grid, 1, 1);
  InitVector(KED, 0.0);

  KND = NewVector(grid, 1, 1);
  InitVector(KND, 0.0);

  KSD = NewVector(grid, 1, 1);
  InitVector(KSD, 0.0);

  qxd = NewVector(grid, 1, 1);
  InitVector(qxd, 0.0);

  qyd = NewVector(grid, 1, 1);
  InitVector(qyd, 0.0);

  /* Pass permeability values */
  /*
   * handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
   * FinalizeVectorUpdate(handle);
   *
   * handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
   * FinalizeVectorUpdate(handle);
   *
   * handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
   * FinalizeVectorUpdate(handle);*/

#if 0
  printf("Check 1 - before accumulation term.\n");
  fflush(NULL);
  malloc_verify(NULL);
#endif

  /* Initialize matrix values to zero. */
  InitMatrix(J, 0.0);

  /* Calculate time term contributions. */

  PFModuleInvoke(void, density_module, (0, pressure, density, &dtmp, &dtmp,
                                        CALCFCN));
  PFModuleInvoke(void, density_module, (0, pressure, density_der, &dtmp,
                                        &dtmp, CALCDER));
  PFModuleInvoke(void, saturation_module, (saturation, pressure,
                                           density, gravity, problem_data,
                                           CALCFCN));
  PFModuleInvoke(void, saturation_module, (saturation_der, pressure,
                                           density, gravity, problem_data,
                                           CALCDER));

#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    J_sub = MatrixSubmatrix(J, is);
    cp = SubmatrixStencilData(J_sub, 0);

    p_sub = VectorSubvector(pressure, is);
    d_sub = VectorSubvector(density, is);
    s_sub = VectorSubvector(saturation, is);
    dd_sub = VectorSubvector(density_der, is);
    sd_sub = VectorSubvector(saturation_der, is);
    po_sub = VectorSubvector(porosity, is);
    ss_sub = VectorSubvector(sstorage, is);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    vol = dx * dy * dz;

    nx_v = SubvectorNX(d_sub);
    ny_v = SubvectorNY(d_sub);
    nz_v = SubvectorNZ(d_sub);

    nx_po = SubvectorNX(po_sub);
    ny_po = SubvectorNY(po_sub);
    nz_po = SubvectorNZ(po_sub);

    nx_m = SubmatrixNX(J_sub);
    ny_m = SubmatrixNY(J_sub);
    nz_m = SubmatrixNZ(J_sub);

    pp = SubvectorData(p_sub);
    dp = SubvectorData(d_sub);
    sp = SubvectorData(s_sub);
    ddp = SubvectorData(dd_sub);
    sdp = SubvectorData(sd_sub);
    pop = SubvectorData(po_sub);
    ss = SubvectorData(ss_sub);

    im = SubmatrixEltIndex(J_sub, ix, iy, iz);
    ipo = SubvectorEltIndex(po_sub, ix, iy, iz);
    iv = SubvectorEltIndex(d_sub, ix, iy, iz);

    BoxLoopI3(i, j, k, ix, iy, iz, nx, ny, nz,
              im, nx_m, ny_m, nz_m, 1, 1, 1,
              ipo, nx_po, ny_po, nz_po, 1, 1, 1,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      cp[im] += (sdp[iv] * dp[iv] + sp[iv] * ddp[iv])
                * pop[ipo] * vol + ss[iv] * vol * (sdp[iv] * dp[iv] * pp[iv] + sp[iv] * ddp[iv] * pp[iv] + sp[iv] * dp[iv]); //sk start
    });
  }       /* End subgrid loop */
#endif


  bc_struct = PFModuleInvoke(BCStruct *, bc_pressure,
                             (problem_data, grid, gr_domain, time));

  /* Get boundary pressure values for Dirichlet boundaries.   */
  /* These are needed for upstream weighting in mobilities - need boundary */
  /* values for rel perms and densities. */

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    p_sub = VectorSubvector(pressure, is);

    nx_v = SubvectorNX(p_sub);
    ny_v = SubvectorNY(p_sub);
    nz_v = SubvectorNZ(p_sub);

    sy_v = nx_v;
    sz_v = ny_v * nx_v;

    pp = SubvectorData(p_sub);

    for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
    {
      bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

      switch (BCStructBCType(bc_struct, ipatch))
      {
        case DirichletBC:
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            ip = SubvectorEltIndex(p_sub, i, j, k);
            value = bc_patch_values[ival];
            pp[ip + fdir[0] * 1 + fdir[1] * sy_v + fdir[2] * sz_v] = value;
          });
          break;
        }
      }        /* End switch BCtype */
    }          /* End ipatch loop */
  }            /* End subgrid loop */

  /* Calculate rel_perm and rel_perm_der */

  PFModuleInvoke(void, rel_perm_module,
                 (rel_perm, pressure, density, gravity, problem_data,
                  CALCFCN));

  PFModuleInvoke(void, rel_perm_module,
                 (rel_perm_der, pressure, density, gravity, problem_data,
                  CALCDER));

#if 1
  /* Calculate contributions from second order derivatives and gravity */
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    p_sub = VectorSubvector(pressure, is);
    d_sub = VectorSubvector(density, is);
    rp_sub = VectorSubvector(rel_perm, is);
    dd_sub = VectorSubvector(density_der, is);
    rpd_sub = VectorSubvector(rel_perm_der, is);
    permx_sub = VectorSubvector(permeability_x, is);
    permy_sub = VectorSubvector(permeability_y, is);
    permz_sub = VectorSubvector(permeability_z, is);
    J_sub = MatrixSubmatrix(J, is);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid) + 1;
    ny = SubgridNY(subgrid) + 1;
    nz = SubgridNZ(subgrid) + 1;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ffx = dy * dz;
    ffy = dx * dz;
    ffz = dx * dy;

    nx_v = SubvectorNX(p_sub);
    ny_v = SubvectorNY(p_sub);
    nz_v = SubvectorNZ(p_sub);

    nx_m = SubmatrixNX(J_sub);
    ny_m = SubmatrixNY(J_sub);
    nz_m = SubmatrixNZ(J_sub);

    sy_v = nx_v;
    sz_v = ny_v * nx_v;
    sy_m = nx_m;
    sz_m = ny_m * nx_m;

    cp = SubmatrixStencilData(J_sub, 0);
    wp = SubmatrixStencilData(J_sub, 1);
    ep = SubmatrixStencilData(J_sub, 2);
    sop = SubmatrixStencilData(J_sub, 3);
    np = SubmatrixStencilData(J_sub, 4);
    lp = SubmatrixStencilData(J_sub, 5);
    up = SubmatrixStencilData(J_sub, 6);

    pp = SubvectorData(p_sub);
    dp = SubvectorData(d_sub);
    rpp = SubvectorData(rp_sub);
    ddp = SubvectorData(dd_sub);
    rpdp = SubvectorData(rpd_sub);
    permxp = SubvectorData(permx_sub);
    permyp = SubvectorData(permy_sub);
    permzp = SubvectorData(permz_sub);

    ip = SubvectorEltIndex(p_sub, ix, iy, iz);
    im = SubmatrixEltIndex(J_sub, ix, iy, iz);

    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_v, ny_v, nz_v, 1, 1, 1,
              im, nx_m, ny_m, nz_m, 1, 1, 1,
    {
      prod = rpp[ip] * dp[ip];
      prod_der = rpdp[ip] * dp[ip] + rpp[ip] * ddp[ip];

      prod_rt = rpp[ip + 1] * dp[ip + 1];
      prod_rt_der = rpdp[ip + 1] * dp[ip + 1] + rpp[ip + 1] * ddp[ip + 1];

      prod_no = rpp[ip + sy_v] * dp[ip + sy_v];
      prod_no_der = rpdp[ip + sy_v] * dp[ip + sy_v]
                    + rpp[ip + sy_v] * ddp[ip + sy_v];

      prod_up = rpp[ip + sz_v] * dp[ip + sz_v];
      prod_up_der = rpdp[ip + sz_v] * dp[ip + sz_v]
                    + rpp[ip + sz_v] * ddp[ip + sz_v];

      /* diff >= 0 implies flow goes left to right */
      diff = pp[ip] - pp[ip + 1];

      x_coeff = dt * ffx * (1.0 / dx)
                * PMean(pp[ip], pp[ip + 1], permxp[ip], permxp[ip + 1])
                / viscosity;

      sym_west_temp = -x_coeff
                      * RPMean(pp[ip], pp[ip + 1], prod, prod_rt);

      west_temp = -x_coeff * diff
                  * RPMean(pp[ip], pp[ip + 1], prod_der, 0.0)
                  + sym_west_temp;

      sym_east_temp = x_coeff
                      * -RPMean(pp[ip], pp[ip + 1], prod, prod_rt);

      east_temp = x_coeff * diff
                  * RPMean(pp[ip], pp[ip + 1], 0.0, prod_rt_der)
                  + sym_east_temp;

      /* diff >= 0 implies flow goes south to north */
      diff = pp[ip] - pp[ip + sy_v];

      y_coeff = dt * ffy * (1.0 / dy)
                * PMean(pp[ip], pp[ip + sy_v], permyp[ip], permyp[ip + sy_v])
                / viscosity;

      sym_south_temp = -y_coeff
                       * RPMean(pp[ip], pp[ip + sy_v], prod, prod_no);

      south_temp = -y_coeff * diff
                   * RPMean(pp[ip], pp[ip + sy_v], prod_der, 0.0)
                   + sym_south_temp;

      sym_north_temp = y_coeff
                       * -RPMean(pp[ip], pp[ip + sy_v], prod, prod_no);

      north_temp = y_coeff * diff
                   * RPMean(pp[ip], pp[ip + sy_v], 0.0,
                            prod_no_der)
                   + sym_north_temp;

      /* diff >= 0 implies flow goes lower to upper */
      lower_cond = pp[ip] - 0.5 * dz * dp[ip] * gravity;
      upper_cond = pp[ip + sz_v] + 0.5 * dz * dp[ip + sz_v] * gravity;
      diff = lower_cond - upper_cond;

      z_coeff = dt * ffz * (1.0 / dz)
                * PMean(lower_cond, upper_cond,
                        permzp[ip], permzp[ip + sz_v])
                / viscosity;

      sym_lower_temp = -z_coeff
                       * RPMean(lower_cond, upper_cond, prod,
                                prod_up);

      lower_temp = -z_coeff
                   * (diff * RPMean(lower_cond, upper_cond, prod_der, 0.0)
                      + (-gravity * 0.5 * dz * ddp[ip]
                         * RPMean(lower_cond, upper_cond, prod,
                                  prod_up)))
                   + sym_lower_temp;

      sym_upper_temp = z_coeff
                       * -RPMean(lower_cond, upper_cond, prod,
                                 prod_up);

      upper_temp = z_coeff
                   * (diff * RPMean(lower_cond, upper_cond, 0.0,
                                    prod_up_der)
                      + (-gravity * 0.5 * dz * ddp[ip + sz_v]
                         * RPMean(lower_cond, upper_cond, prod,
                                  prod_up)))
                   + sym_upper_temp;

      cp[im] -= west_temp + south_temp + lower_temp;
      cp[im + 1] -= east_temp;
      cp[im + sy_m] -= north_temp;
      cp[im + sz_m] -= upper_temp;

      if (!symm_part)
      {
        ep[im] += east_temp;
        np[im] += north_temp;
        up[im] += upper_temp;

        wp[im + 1] += west_temp;
        sop[im + sy_m] += south_temp;
        lp[im + sz_m] += lower_temp;
      }
      else            /* Symmetric matrix: just update upper coeffs */
      {
        ep[im] += sym_east_temp;
        np[im] += sym_north_temp;
        up[im] += sym_upper_temp;

        /*
         * wp[im+1] += sym_west_temp;
         * sop[im+sy_m] += sym_south_temp;
         * lp[im+sz_m] += sym_lower_temp;
         */
      }

      /*
       * if ( (i == 0) && (j==0) && (k==0) )
       * printf("Stencil update: east_temp %f prod %f prod_rt %f\n"
       *     "   north_temp %f upper_temp %f\n"
       *     "   ep[im] %f i %i j %i k %i\n",
       *     east_temp, prod, prod_rt, north_temp, upper_temp,
       *     ep[im], i, j, k);
       */
    });
  }

#if 0
  PrintMatrix("matrix_dump1", J);
#endif
#endif

  /*  Calculate correction for boundary conditions */

  if (symm_part)
  {
    /*  For symmetric part only, we first adjust coefficients of normal */
    /*  direction boundary pressure by adding in the nonsymmetric part. */
    /*  The entire coefficicent will be subtracted from the diagonal    */
    /*  and set to zero in the subsequent section - no matter what type */
    /*  of BC is involved.  Without this correction, only the symmetric */
    /*  part would be removed, incorrectly leaving the nonsymmetric     */
    /*  contribution on the diagonal.                                   */

    ForSubgridI(is, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, is);

      p_sub = VectorSubvector(pressure, is);
      dd_sub = VectorSubvector(density_der, is);
      rpd_sub = VectorSubvector(rel_perm_der, is);
      d_sub = VectorSubvector(density, is);
      rp_sub = VectorSubvector(rel_perm, is);
      permx_sub = VectorSubvector(permeability_x, is);
      permy_sub = VectorSubvector(permeability_y, is);
      permz_sub = VectorSubvector(permeability_z, is);
      J_sub = MatrixSubmatrix(J, is);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);

      ffx = dy * dz;
      ffy = dx * dz;
      ffz = dx * dy;

      nx_v = SubvectorNX(p_sub);
      ny_v = SubvectorNY(p_sub);
      nz_v = SubvectorNZ(p_sub);

      sy_v = nx_v;
      sz_v = ny_v * nx_v;

      cp = SubmatrixStencilData(J_sub, 0);
      wp = SubmatrixStencilData(J_sub, 1);
      ep = SubmatrixStencilData(J_sub, 2);
      sop = SubmatrixStencilData(J_sub, 3);
      np = SubmatrixStencilData(J_sub, 4);
      lp = SubmatrixStencilData(J_sub, 5);
      up = SubmatrixStencilData(J_sub, 6);

      pp = SubvectorData(p_sub);
      ddp = SubvectorData(dd_sub);
      rpdp = SubvectorData(rpd_sub);
      dp = SubvectorData(d_sub);
      rpp = SubvectorData(rp_sub);
      permxp = SubvectorData(permx_sub);
      permyp = SubvectorData(permy_sub);
      permzp = SubvectorData(permz_sub);

      for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
      {
        BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
        {
          ip = SubvectorEltIndex(p_sub, i, j, k);
          im = SubmatrixEltIndex(J_sub, i, j, k);

          if (fdir[0])
          {
            switch (fdir[0])
            {
              case -1:
                {
                  diff = pp[ip - 1] - pp[ip];
                  prod_der = rpdp[ip - 1] * dp[ip - 1] + rpp[ip - 1] * ddp[ip - 1];
                  coeff = dt * ffx * (1.0 / dx)
                          * PMean(pp[ip - 1], pp[ip], permxp[ip - 1], permxp[ip])
                          / viscosity;
                  wp[im] = -coeff * diff
                           * RPMean(pp[ip - 1], pp[ip], prod_der, 0.0);
                  break;
                }

              case 1:
                {
                  diff = pp[ip] - pp[ip + 1];
                  prod_der = rpdp[ip + 1] * dp[ip + 1] + rpp[ip + 1] * ddp[ip + 1];
                  coeff = dt * ffx * (1.0 / dx)
                          * PMean(pp[ip], pp[ip + 1], permxp[ip], permxp[ip + 1])
                          / viscosity;
                  ep[im] = coeff * diff
                           * RPMean(pp[ip], pp[ip + 1], 0.0, prod_der);
                  break;
                }
            }            /* End switch on fdir[0] */
          }              /* End if (fdir[0]) */

          else if (fdir[1])
          {
            switch (fdir[1])
            {
              case -1:
                {
                  diff = pp[ip - sy_v] - pp[ip];
                  prod_der = rpdp[ip - sy_v] * dp[ip - sy_v]
                             + rpp[ip - sy_v] * ddp[ip - sy_v];
                  coeff = dt * ffy * (1.0 / dy)
                          * PMean(pp[ip - sy_v], pp[ip],
                                  permyp[ip - sy_v], permyp[ip])
                          / viscosity;
                  sop[im] = -coeff * diff
                            * RPMean(pp[ip - sy_v], pp[ip], prod_der, 0.0);
                  break;
                }

              case 1:
                {
                  diff = pp[ip] - pp[ip + sy_v];
                  prod_der = rpdp[ip + sy_v] * dp[ip + sy_v]
                             + rpp[ip + sy_v] * ddp[ip + sy_v];
                  coeff = dt * ffy * (1.0 / dy)
                          * PMean(pp[ip], pp[ip + sy_v],
                                  permyp[ip], permyp[ip + sy_v])
                          / viscosity;
                  np[im] = -coeff * diff
                           * RPMean(pp[ip], pp[ip + sy_v], 0.0, prod_der);
                  break;
                }
            }            /* End switch on fdir[1] */
          }              /* End if (fdir[1]) */

          else if (fdir[2])
          {
            switch (fdir[2])
            {
              case -1:
                {
                  lower_cond = (pp[ip - sz_v])
                               - 0.5 * dz * dp[ip - sz_v] * gravity;
                  upper_cond = (pp[ip]) + 0.5 * dz * dp[ip] * gravity;
                  diff = lower_cond - upper_cond;
                  prod_der = rpdp[ip - sz_v] * dp[ip - sz_v]
                             + rpp[ip - sz_v] * ddp[ip - sz_v];
                  prod_lo = rpp[ip - sz_v] * dp[ip - sz_v];
                  coeff = dt * ffz * (1.0 / dz)
                          * PMean(pp[ip - sz_v], pp[ip],
                                  permzp[ip - sz_v], permzp[ip])
                          / viscosity;
                  lp[im] = -coeff *
                           (diff * RPMean(lower_cond, upper_cond,
                                          prod_der, 0.0)
                            - gravity * 0.5 * dz * ddp[ip]
                            * RPMean(lower_cond, upper_cond, prod_lo, prod));
                  break;
                }

              case 1:
                {
                  lower_cond = (pp[ip]) - 0.5 * dz * dp[ip] * gravity;
                  upper_cond = (pp[ip + sz_v])
                               + 0.5 * dz * dp[ip + sz_v] * gravity;
                  diff = lower_cond - upper_cond;
                  prod_der = rpdp[ip + sz_v] * dp[ip + sz_v]
                             + rpp[ip + sz_v] * ddp[ip + sz_v];
                  prod_up = rpp[ip + sz_v] * dp[ip + sz_v];
                  coeff = dt * ffz * (1.0 / dz)
                          * PMean(lower_cond, upper_cond,
                                  permzp[ip], permzp[ip + sz_v])
                          / viscosity;
                  up[im] = -coeff *
                           (diff * RPMean(lower_cond, upper_cond,
                                          0.0, prod_der)
                            - gravity * 0.5 * dz * ddp[ip]
                            * RPMean(lower_cond, upper_cond, prod, prod_up));
                  break;
                }
            }            /* End switch on fdir[2] */
          }           /* End if (fdir[2]) */
        });          /* End Patch Loop */
      }              /* End ipatch loop */
    }                /* End subgrid loop */
  }                  /* End if symm_part */


#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    p_sub = VectorSubvector(pressure, is);
    s_sub = VectorSubvector(saturation, is);
    dd_sub = VectorSubvector(density_der, is);
    rpd_sub = VectorSubvector(rel_perm_der, is);
    d_sub = VectorSubvector(density, is);
    rp_sub = VectorSubvector(rel_perm, is);
    permx_sub = VectorSubvector(permeability_x, is);
    permy_sub = VectorSubvector(permeability_y, is);
    permz_sub = VectorSubvector(permeability_z, is);
    J_sub = MatrixSubmatrix(J, is);

    kwd_sub = VectorSubvector(KWD, is);
    ked_sub = VectorSubvector(KED, is);
    knd_sub = VectorSubvector(KND, is);
    ksd_sub = VectorSubvector(KSD, is);
    qxd_sub = VectorSubvector(qxd, is);
    qyd_sub = VectorSubvector(qyd, is);
    x_sl_sub = VectorSubvector(x_sl, is);
    y_sl_sub = VectorSubvector(y_sl, is);
    mann_sub = VectorSubvector(man, is);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);

    ffx = dy * dz;
    ffy = dx * dz;
    ffz = dx * dy;

    nx_v = SubvectorNX(p_sub);
    ny_v = SubvectorNY(p_sub);
    nz_v = SubvectorNZ(p_sub);

    sy_v = nx_v;
    sz_v = ny_v * nx_v;
    cx = (dt * vol / (dz * dx));   //sk
    cy = (dt * vol / (dz * dy));   //sk

    cp = SubmatrixStencilData(J_sub, 0);
    wp = SubmatrixStencilData(J_sub, 1);
    ep = SubmatrixStencilData(J_sub, 2);
    sop = SubmatrixStencilData(J_sub, 3);
    np = SubmatrixStencilData(J_sub, 4);
    lp = SubmatrixStencilData(J_sub, 5);
    up = SubmatrixStencilData(J_sub, 6);

    pp = SubvectorData(p_sub);
    sp = SubvectorData(s_sub);
    ddp = SubvectorData(dd_sub);
    rpdp = SubvectorData(rpd_sub);
    dp = SubvectorData(d_sub);
    rpp = SubvectorData(rp_sub);
    permxp = SubvectorData(permx_sub);
    permyp = SubvectorData(permy_sub);
    permzp = SubvectorData(permz_sub);


    kwd_ = SubvectorData(kwd_sub);
    ked_ = SubvectorData(ked_sub);
    knd_ = SubvectorData(knd_sub);
    ksd_ = SubvectorData(ksd_sub);
    qxd_ = SubvectorData(qxd_sub);
    qyd_ = SubvectorData(qyd_sub);
    x_sl_dat = SubvectorData(x_sl_sub);
    y_sl_dat = SubvectorData(y_sl_sub);
    mann_dat = SubvectorData(mann_sub);


    for (ipatch = 0; ipatch < BCStructNumPatches(bc_struct); ipatch++)
    {
      bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);

      switch (BCStructBCType(bc_struct, ipatch))
      {
        case DirichletBC:
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            ip = SubvectorEltIndex(p_sub, i, j, k);

            value = bc_patch_values[ival];

            PFModuleInvoke(void, density_module,
                           (0, NULL, NULL, &value, &den_d, CALCFCN));
            PFModuleInvoke(void, density_module,
                           (0, NULL, NULL, &value, &dend_d, CALCDER));

            ip = SubvectorEltIndex(p_sub, i, j, k);
            im = SubmatrixEltIndex(J_sub, i, j, k);

            prod = rpp[ip] * dp[ip];
            prod_der = rpdp[ip] * dp[ip] + rpp[ip] * ddp[ip];

            if (fdir[0])
            {
              coeff = dt * ffx * (2.0 / dx) * permxp[ip] / viscosity;

              switch (fdir[0])
              {
                case -1:
                  {
                    op = wp;
                    prod_val = rpp[ip - 1] * den_d;
                    diff = value - pp[ip];
                    o_temp = coeff
                             * (diff * RPMean(value, pp[ip], 0.0, prod_der)
                                - RPMean(value, pp[ip], prod_val, prod));
                    break;
                  }

                case 1:
                  {
                    op = ep;
                    prod_val = rpp[ip + 1] * den_d;
                    diff = pp[ip] - value;
                    o_temp = -coeff
                             * (diff * RPMean(pp[ip], value, prod_der, 0.0)
                                + RPMean(pp[ip], value, prod, prod_val));
                    break;
                  }
              }       /* End switch on fdir[0] */
            }         /* End if (fdir[0]) */

            else if (fdir[1])
            {
              coeff = dt * ffy * (2.0 / dy) * permyp[ip] / viscosity;

              switch (fdir[1])
              {
                case -1:
                  {
                    op = sop;
                    prod_val = rpp[ip - sy_v] * den_d;
                    diff = value - pp[ip];
                    o_temp = coeff
                             * (diff * RPMean(value, pp[ip], 0.0, prod_der)
                                - RPMean(value, pp[ip], prod_val, prod));
                    break;
                  }

                case 1:
                  {
                    op = np;
                    prod_val = rpp[ip + sy_v] * den_d;
                    diff = pp[ip] - value;
                    o_temp = -coeff
                             * (diff * RPMean(pp[ip], value, prod_der, 0.0)
                                + RPMean(pp[ip], value, prod, prod_val));
                    break;
                  }
              }       /* End switch on fdir[1] */
            }         /* End if (fdir[1]) */

            else if (fdir[2])
            {
              coeff = dt * ffz * (2.0 / dz) * permzp[ip] / viscosity;

              switch (fdir[2])
              {
                case -1:
                  {
                    op = lp;
                    prod_val = rpp[ip - sz_v] * den_d;

                    lower_cond = (value) - 0.5 * dz * den_d * gravity;
                    upper_cond = (pp[ip]) + 0.5 * dz * dp[ip] * gravity;
                    diff = lower_cond - upper_cond;

                    o_temp = coeff
                             * (diff * RPMean(lower_cond, upper_cond, 0.0, prod_der)
                                + ((-1.0 - gravity * 0.5 * dz * ddp[ip])
                                   * RPMean(lower_cond, upper_cond, prod_val, prod)));
                    break;
                  }

                case 1:
                  {
                    op = up;
                    prod_val = rpp[ip + sz_v] * den_d;

                    lower_cond = (pp[ip]) - 0.5 * dz * dp[ip] * gravity;
                    upper_cond = (value) + 0.5 * dz * den_d * gravity;
                    diff = lower_cond - upper_cond;

                    o_temp = -coeff
                             * (diff * RPMean(lower_cond, upper_cond, prod_der, 0.0)
                                + ((1.0 - gravity * 0.5 * dz * ddp[ip])
                                   * RPMean(lower_cond, upper_cond, prod, prod_val)));
                    break;
                  }
              }       /* End switch on fdir[2] */
            }         /* End if (fdir[2]) */

            cp[im] += op[im];
            cp[im] -= o_temp;
            op[im] = 0.0;
            /*
             * if ( (i==0) && (j==0) && (k==0) )
             * printf("fdir: %i %i %i o_temp %f\n",
             *        fdir[0], fdir[1], fdir[2], o_temp);
             */
          });

          break;
        }            /* End DirichletBC case */

        case FluxBC:
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            im = SubmatrixEltIndex(J_sub, i, j, k);

            if (fdir[0] == -1)
              op = wp;
            if (fdir[0] == 1)
              op = ep;
            if (fdir[1] == -1)
              op = sop;
            if (fdir[1] == 1)
              op = np;
            if (fdir[2] == -1)
              op = lp;
            if (fdir[2] == 1)
              op = up;

            cp[im] += op[im];
            op[im] = 0.0;
          });

          break;
        }      /* End fluxbc case */

        case OverlandBC:  //sk
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            im = SubmatrixEltIndex(J_sub, i, j, k);

            if (fdir[0] == -1)
              op = wp;
            if (fdir[0] == 1)
              op = ep;
            if (fdir[1] == -1)
              op = sop;
            if (fdir[1] == 1)
              op = np;
            if (fdir[2] == -1)
              op = lp;
            if (fdir[2] == 1)
              op = up;

            cp[im] += op[im];
            op[im] = 0.0;
          });


          break;
        }      /* End overland flow case */
      }        /* End switch BCtype */
    }          /* End ipatch loop */
  }            /* End subgrid loop */
#endif

#if 0
  PrintMatrix("matrix_dump2", J);
#endif

  FreeBCStruct(bc_struct);
#if 1
  PFModuleInvoke(void, bc_internal, (problem, problem_data, NULL, J, time,
                                     pressure, CALCDER));
#endif
#if 1
  /* Set pressures outside domain to zero.
   * Recall: equation to solve is f = 0, so components of f outside
   * domain are set to the respective pressure value.
   *
   * Should change this to set pressures to scaling value.
   * CSW: Should I set this to pressure * vol * dt ??? */

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    J_sub = MatrixSubmatrix(J, is);

    /* RDF: assumes resolutions are the same in all 3 directions */
    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    cp = SubmatrixStencilData(J_sub, 0);
    wp = SubmatrixStencilData(J_sub, 1);
    ep = SubmatrixStencilData(J_sub, 2);
    sop = SubmatrixStencilData(J_sub, 3);
    np = SubmatrixStencilData(J_sub, 4);
    lp = SubmatrixStencilData(J_sub, 5);
    up = SubmatrixStencilData(J_sub, 6);

    GrGeomOutLoop(i, j, k, gr_domain,
                  r, ix, iy, iz, nx, ny, nz,
    {
      im = SubmatrixEltIndex(J_sub, i, j, k);
      cp[im] = 1.0;
      wp[im] = 0.0;
      ep[im] = 0.0;
      sop[im] = 0.0;
      np[im] = 0.0;
      lp[im] = 0.0;
      up[im] = 0.0;
    });
  }
#endif

  /*-----------------------------------------------------------------------
   * Update matrix ghost points
   *-----------------------------------------------------------------------*/

  if (MatrixCommPkg(J))
  {
    handle = InitMatrixUpdate(J);
    FinalizeMatrixUpdate(handle);
  }

  *ptr_to_J = J;

#if 0
  PrintMatrix("matrix_dump3", J);
  exit(1);
#endif

  //FreeVector(psi); //sk start
  FreeVector(KWD);
  FreeVector(KED);
  FreeVector(KND);
  FreeVector(KSD);
  FreeVector(qxd);
  FreeVector(qyd);

  return;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *RichardsJacobianEvalInitInstanceXtra(problem, grid, temp_data,
                                                  symmetric_jac)
Problem * problem;
Grid        *grid;
double      *temp_data;
int symmetric_jac;
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  Stencil       *stencil;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeMatrix(instance_xtra->J);
      FreeTempVector(instance_xtra->density_der);
      FreeTempVector(instance_xtra->saturation_der);
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    /* set up jacobian matrix */
    stencil = NewStencil(jacobian_stencil_shape, 7);

    if (symmetric_jac)
      (instance_xtra->J) = NewMatrix(grid, NULL, stencil, ON, stencil);
    else
      (instance_xtra->J) = NewMatrix(grid, NULL, stencil, OFF, stencil);

    (instance_xtra->density_der) = NewTempVector(grid, 1, 1);
    (instance_xtra->saturation_der) = NewTempVector(grid, 1, 1);
  }

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
    SetTempVectorData((instance_xtra->density_der), temp_data);
    temp_data += SizeOfVector(instance_xtra->density_der);
    SetTempVectorData((instance_xtra->saturation_der), temp_data);
    temp_data += SizeOfVector(instance_xtra->saturation_der);
  }

  if (problem != NULL)
  {
    (instance_xtra->problem) = problem;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->density_module) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
    (instance_xtra->bc_pressure) =
      PFModuleNewInstance(ProblemBCPressure(problem), (problem));
    (instance_xtra->saturation_module) =
      PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL));
    (instance_xtra->rel_perm_module) =
      PFModuleNewInstance(ProblemPhaseRelPerm(problem), (NULL, NULL));
    (instance_xtra->bc_internal) =
      PFModuleNewInstance(ProblemBCInternal(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->density_module), ());
    PFModuleReNewInstance((instance_xtra->bc_pressure), (problem));
    PFModuleReNewInstance((instance_xtra->saturation_module),
                          (NULL, NULL));
    PFModuleReNewInstance((instance_xtra->rel_perm_module),
                          (NULL, NULL));
    PFModuleReNewInstance((instance_xtra->bc_internal), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  RichardsJacobianEvalFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->density_module);
    PFModuleFreeInstance(instance_xtra->bc_pressure);
    PFModuleFreeInstance(instance_xtra->saturation_module);
    PFModuleFreeInstance(instance_xtra->rel_perm_module);
    PFModuleFreeInstance(instance_xtra->bc_internal);

    FreeTempVector(instance_xtra->density_der);
    FreeTempVector(instance_xtra->saturation_der);

    FreeMatrix(instance_xtra->J);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *RichardsJacobianEvalNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;


  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  RichardsJacobianEvalFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * RichardsJacobianEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  RichardsJacobianEvalSizeOfTempData()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  int sz = 0;

  /* add local TempData size to `sz' */
  sz += SizeOfVector(instance_xtra->density_der);
  sz += SizeOfVector(instance_xtra->saturation_der);

  return sz;
}
