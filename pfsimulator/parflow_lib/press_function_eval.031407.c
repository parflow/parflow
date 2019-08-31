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



/*---------------------------------------------------------------------
 * Define module structures
 *---------------------------------------------------------------------*/

typedef struct {
  int time_index;
} PublicXtra;

typedef struct {
  Problem      *problem;

  PFModule     *density_module;
  PFModule     *viscosity_module;
  PFModule     *saturation_module;
  PFModule     *rel_perm_module;
  PFModule     *phase_source;
  PFModule     *bc_pressure;
  PFModule     *bc_internal;

  Vector       *rel_perm;
  Vector       *qsource;
  Vector       *tsource;

  Grid         *grid;
} InstanceXtra;

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/

#define PMean(a, b, c, d)    HarmonicMean(c, d)
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)
#define AMean(a, b)           ArithmeticMean(a, b)

/*  This routine evaluates the nonlinear function based on the current
 *  pressure values.  This evaluation is basically an application
 *  of the stencil to the pressure array. */

void    PressFunctionEval(pressure, fval, problem_data, temperature, saturation,
                          old_saturation, density, old_density, viscosity, dt,
                          time, old_pressure, outflow, evap_trans, ovrl_bc_flx,
                          x_velocity, y_velocity, z_velocity)

Vector * pressure;           /* Current pressure values */
Vector      *fval;           /* Return values of the nonlinear function */
ProblemData *problem_data;   /* Geometry data for problem */
Vector      *temperature;
Vector      *saturation;     /* Saturation / work vector */
Vector      *old_saturation; /* Saturation values at previous time step */
Vector      *density;        /* Density vector */
Vector      *old_density;    /* Density values at previous time step */
Vector      *viscosity;
double dt;                   /* Time step size */
double time;                 /* New time value */
Vector      *old_pressure;
double      *outflow;       /*sk Outflow due to overland flow*/
Vector      *evap_trans;     /*sk sink term from land surface model*/
Vector      *ovrl_bc_flx;     /*sk overland flow boundary fluxes*/
Vector      *x_velocity;
Vector      *y_velocity;
Vector      *z_velocity;

{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);
  PublicXtra    *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  Problem     *problem = (instance_xtra->problem);

  PFModule    *density_module = (instance_xtra->density_module);
  PFModule    *viscosity_module = (instance_xtra->viscosity_module);
  PFModule    *saturation_module = (instance_xtra->saturation_module);
  PFModule    *rel_perm_module = (instance_xtra->rel_perm_module);
  PFModule    *phase_source = (instance_xtra->phase_source);
  PFModule    *bc_pressure = (instance_xtra->bc_pressure);
  PFModule    *bc_internal = (instance_xtra->bc_internal);

  /* Re-use saturation vector to save memory */
  Vector      *rel_perm = (instance_xtra->rel_perm);
  Vector      *qsource = (instance_xtra->qsource);
  Vector      *tsource = (instance_xtra->tsource);

  //Vector      *rel_perm          = saturation;
  //Vector      *source            = saturation;

  /* Overland flow variables */  //sk
  Vector      *KW, *KE, *KN, *KS;
  Vector      *qx, *qy;
  Subvector   *kw_sub, *ke_sub, *kn_sub, *ks_sub, *qx_sub, *qy_sub;
  Subvector   *x_sl_sub, *y_sl_sub, *mann_sub;
  Subvector   *obf_sub;
  double      *kw_, *ke_, *kn_, *ks_, *qx_, *qy_;
  double      *x_sl_dat, *y_sl_dat, *mann_dat;
  double      *obf_dat;
  double dir_x, dir_y;
  int t;
  double q_overlnd;

  //  double      press[12][12][10],pressbc[12][12],xslope[12][12],yslope[12][12];
//   double      press[2][1][390],pressbc[400][1],xslope[58][30],yslope[58][30];

  Vector      *porosity = ProblemDataPorosity(problem_data);
  Vector      *permeability25_x = ProblemDataPermeabilityX(problem_data);
  Vector      *permeability25_y = ProblemDataPermeabilityY(problem_data);
  Vector      *permeability25_z = ProblemDataPermeabilityZ(problem_data);
  Vector      *sstorage = ProblemDataSpecificStorage(problem_data);            //sk
  Vector      *x_sl = ProblemDataTSlopeX(problem_data);                //sk
  Vector      *y_sl = ProblemDataTSlopeY(problem_data);                //sk
  Vector      *man = ProblemDataMannings(problem_data);                 //sk

  double gravity = ProblemGravity(problem);

  /*Temperature variables*/
  Vector      *permeability_x;
  Vector      *permeability_y;
  Vector      *permeability_z;

  Subgrid     *subgrid;

  Subvector   *p_sub, *d_sub, *v_sub, *od_sub, *s_sub, *os_sub, *po_sub, *op_sub, *ss_sub, *et_sub;
  Subvector   *f_sub, *rp_sub, *permx_sub, *permy_sub, *permz_sub, *permx25_sub, *permy25_sub, *permz25_sub;

  /* Linear flow velocities */
  Subvector   *xv_sub, *yv_sub, *zv_sub;
  double      *xvp, *yvp, *zvp;

  Grid        *grid = VectorGrid(pressure);
  Grid        *grid2d = VectorGrid(x_sl);

  double      *pp, *odp, *sp, *osp, *pop, *fp, *dp, *vp, *rpp, *opp, *ss, *et;
  double      *permxp, *permyp, *permzp, *permx25p, *permy25p, *permz25p;

  int i, j, k, r, is;
  int ix, iy, iz;
  int nx, ny, nz, gnx, gny;
  int nx_f, ny_f, nz_f;
  int nx_p, ny_p, nz_p;
  int nx_po, ny_po, nz_po;
  int sy_p, sz_p;
  int ip, ipo, io;

  double dtmp, dx, dy, dz, vol, ffx, ffy, ffz;
  double u_right, u_front, u_upper;
  double diff = 0.0e0;
  double lower_cond, upper_cond;

  BCStruct    *bc_struct;
  GrGeomSolid *gr_domain = ProblemDataGrDomain(problem_data);
  double      *bc_patch_values;
  double u_old = 0.0e0;
  double u_new = 0.0e0;
  double value;
  int         *fdir;
  int ipatch, ival;
  int dir = 0;

  CommHandle  *handle;

  BeginTiming(public_xtra->time_index);

  /* Initialize function values to zero. */
  PFVConstInit(0.0, fval);

  /* Pass pressure values to neighbors.  */
  handle = InitVectorUpdate(pressure, VectorUpdateAll);
  FinalizeVectorUpdate(handle);

  KW = NewVector(grid2d, 1, 1);
  InitVector(KW, 0.0);

  KE = NewVector(grid2d, 1, 1);
  InitVector(KE, 0.0);

  KN = NewVector(grid2d, 1, 1);
  InitVector(KN, 0.0);

  KS = NewVector(grid2d, 1, 1);
  InitVector(KS, 0.0);

  qx = NewVector(grid2d, 1, 1);
  InitVector(qx, 0.0);

  qy = NewVector(grid2d, 1, 1);
  InitVector(qy, 0.0);

  permeability_x = NewVector(grid, 1, 1);
  InitVectorAll(permeability_x, 0.0);

  permeability_y = NewVector(grid, 1, 1);
  InitVectorAll(permeability_y, 0.0);

  permeability_z = NewVector(grid, 1, 1);
  InitVectorAll(permeability_z, 0.0);

  /* Pass permeability values */
  /*handle = InitVectorUpdate(permeability_x, VectorUpdateAll);
   * FinalizeVectorUpdate(handle);
   *
   * handle = InitVectorUpdate(permeability_y, VectorUpdateAll);
   * FinalizeVectorUpdate(handle);
   *
   * handle = InitVectorUpdate(permeability_z, VectorUpdateAll);
   * FinalizeVectorUpdate(handle); */

  /* Calculate pressure dependent properties: density and saturation */
  PFModuleInvoke(void, density_module, (0, pressure, density, &dtmp, &dtmp,
                                        CALCFCN));

  PFModuleInvoke(void, viscosity_module, (0, pressure, temperature, viscosity,
                                          CALCFCN));

  PFModuleInvoke(void, saturation_module, (saturation, pressure, density,
                                           gravity, problem_data, CALCFCN));

  /* bc_struct = PFModuleInvoke(BCStruct *, bc_pressure,
   *                         (problem_data, grid, gr_domain, time));*/

  /*@ Why are the above things calculated here again; they were allready
   *  calculated in the driver solver_richards and passed further @*/

#if 0
  printf("Check 1 - before accumulation term.\n");
  fflush(NULL);
  malloc_verify(NULL);
#endif

#if 1
  /* Calculate accumulation terms for the function values */

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    d_sub = VectorSubvector(density, is);
    od_sub = VectorSubvector(old_density, is);
    p_sub = VectorSubvector(pressure, is);
    op_sub = VectorSubvector(old_pressure, is);
    s_sub = VectorSubvector(saturation, is);
    os_sub = VectorSubvector(old_saturation, is);
    po_sub = VectorSubvector(porosity, is);
    f_sub = VectorSubvector(fval, is);

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

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    nx_po = SubvectorNX(po_sub);
    ny_po = SubvectorNY(po_sub);
    nz_po = SubvectorNZ(po_sub);

    dp = SubvectorData(d_sub);
    odp = SubvectorData(od_sub);
    sp = SubvectorData(s_sub);
    pp = SubvectorData(p_sub);
    opp = SubvectorData(op_sub);
    osp = SubvectorData(os_sub);
    pop = SubvectorData(po_sub);
    fp = SubvectorData(f_sub);

    ip = SubvectorEltIndex(f_sub, ix, iy, iz);
    ipo = SubvectorEltIndex(po_sub, ix, iy, iz);

    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_f, ny_f, nz_f, 1, 1, 1,
              ipo, nx_po, ny_po, nz_po, 1, 1, 1,
    {
      fp[ip] = (sp[ip] * dp[ip] - osp[ip] * odp[ip]) * pop[ipo] * vol;
    });
  }
#endif

  /*@ Add in contributions from compressible storage */

#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    ss_sub = VectorSubvector(sstorage, is);

    d_sub = VectorSubvector(density, is);
    od_sub = VectorSubvector(old_density, is);
    p_sub = VectorSubvector(pressure, is);
    op_sub = VectorSubvector(old_pressure, is);
    s_sub = VectorSubvector(saturation, is);
    os_sub = VectorSubvector(old_saturation, is);
    f_sub = VectorSubvector(fval, is);

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

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    ss = SubvectorData(ss_sub);

    dp = SubvectorData(d_sub);
    odp = SubvectorData(od_sub);
    sp = SubvectorData(s_sub);
    pp = SubvectorData(p_sub);
    opp = SubvectorData(op_sub);
    osp = SubvectorData(os_sub);
    fp = SubvectorData(f_sub);


    ip = SubvectorEltIndex(f_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      fp[ip] += ss[ip] * vol * (pp[ip] * sp[ip] * dp[ip] - opp[ip] * osp[ip] * odp[ip]);
      //press[i][j][k]=pp[ip];
    });
  }
#endif


  /* Add in contributions from source terms - user specified sources and
   * flux wells.  Calculate phase source values overwriting current
   * saturation vector */

  PFModuleInvoke(void, phase_source, (qsource, tsource, problem, problem_data,
                                      time));

#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    s_sub = VectorSubvector(qsource, is);
    f_sub = VectorSubvector(fval, is);
    et_sub = VectorSubvector(evap_trans, is);

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

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    sp = SubvectorData(s_sub);
    fp = SubvectorData(f_sub);
    et = SubvectorData(et_sub);

    ip = SubvectorEltIndex(f_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      fp[ip] -= vol * dt * (sp[ip] + et[ip]);
    });
  }
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

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    sy_p = nx_p;
    sz_p = ny_p * nx_p;

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
            pp[ip + fdir[0] * 1 + fdir[1] * sy_p + fdir[2] * sz_p] = value;
          });
          break;
        }
      }        /* End switch BCtype */
    }          /* End ipatch loop */
  }            /* End subgrid loop */

  /*Recalculate permeabilities based on new relative viscosities*/
#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    permx_sub = VectorSubvector(permeability_x, is);
    permy_sub = VectorSubvector(permeability_y, is);
    permz_sub = VectorSubvector(permeability_z, is);

    permx25_sub = VectorSubvector(permeability25_x, is);
    permy25_sub = VectorSubvector(permeability25_y, is);
    permz25_sub = VectorSubvector(permeability25_z, is);

    v_sub = VectorSubvector(viscosity, is);

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

    nx_f = SubvectorNX(f_sub);
    ny_f = SubvectorNY(f_sub);
    nz_f = SubvectorNZ(f_sub);

    permxp = SubvectorData(permx_sub);
    permyp = SubvectorData(permy_sub);
    permzp = SubvectorData(permz_sub);
    permx25p = SubvectorData(permx25_sub);
    permy25p = SubvectorData(permy25_sub);
    permz25p = SubvectorData(permz25_sub);
    vp = SubvectorData(v_sub);

    ip = SubvectorEltIndex(f_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_f, ny_f, nz_f, 1, 1, 1,
    {
      permxp[ip] = vp[ip] * permx25p[ip];
      permyp[ip] = vp[ip] * permy25p[ip];
      permzp[ip] = vp[ip] * permz25p[ip];
    });
  }
#endif

  /* Calculate relative permeability values overwriting current
   * phase source values */

  PFModuleInvoke(void, rel_perm_module,
                 (rel_perm, pressure, density, gravity, problem_data,
                  CALCFCN));

#if 1
  /* Calculate contributions from second order derivatives and gravity */
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    p_sub = VectorSubvector(pressure, is);
    s_sub = VectorSubvector(saturation, is);
    d_sub = VectorSubvector(density, is);
    rp_sub = VectorSubvector(rel_perm, is);
    f_sub = VectorSubvector(fval, is);
    permx_sub = VectorSubvector(permeability_x, is);
    permy_sub = VectorSubvector(permeability_y, is);
    permz_sub = VectorSubvector(permeability_z, is);
    po_sub = VectorSubvector(porosity, is);
    xv_sub = VectorSubvector(x_velocity, is);
    yv_sub = VectorSubvector(y_velocity, is);
    zv_sub = VectorSubvector(z_velocity, is);

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

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    sy_p = nx_p;
    sz_p = ny_p * nx_p;

    pp = SubvectorData(p_sub);
    sp = SubvectorData(s_sub);
    dp = SubvectorData(d_sub);
    rpp = SubvectorData(rp_sub);
    fp = SubvectorData(f_sub);
    permxp = SubvectorData(permx_sub);
    permyp = SubvectorData(permy_sub);
    permzp = SubvectorData(permz_sub);
    pop = SubvectorData(po_sub);
    xvp = SubvectorData(xv_sub);
    yvp = SubvectorData(yv_sub);
    zvp = SubvectorData(zv_sub);

    ip = SubvectorEltIndex(p_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              ip, nx_p, ny_p, nz_p, 1, 1, 1,
    {
//         if(k==0) printf("i, j, t, io, qy %d %d %d %d %e\n",i,j,t,io,pp[ip]);
      /* Calculate right face velocity.
       * diff >= 0 implies flow goes left to right */
      diff = pp[ip] - pp[ip + 1];
      u_right = ffx * PMean(pp[ip], pp[ip + 1],
                            permxp[ip], permxp[ip + 1])
                * (diff / dx)
                * RPMean(pp[ip], pp[ip + 1], rpp[ip] * dp[ip],
                         rpp[ip + 1] * dp[ip + 1]);
      // xvp[ip] = 1.0/ffx * u_right / ( AMean(sp[ip], sp[ip+1]) * AMean(pop[ip],pop[ip+1]) );
      xvp[ip] = 1.0 / ffx * u_right;
      //if (i>=0 && j>=0 && k>=0) printf("Vel %d %d %d %e %e %e \n",i,j,k,xvp[ip],diff, RPMean(pp[ip], pp[ip+1], rpp[ip]*dp[ip],
      //                        rpp[ip+1]*dp[ip+1]));

      /* Calculate front face velocity.
       * diff >= 0 implies flow goes back to front */
      diff = pp[ip] - pp[ip + sy_p];
      u_front = ffy * PMean(pp[ip], pp[ip + sy_p], permyp[ip],
                            permyp[ip + sy_p])
                * (diff / dy)
                * RPMean(pp[ip], pp[ip + sy_p], rpp[ip] * dp[ip],
                         rpp[ip + sy_p] * dp[ip + sy_p]);
      yvp[ip] = 1.0 / ffy * u_front / (AMean(sp[ip], sp[ip + sy_p]) * AMean(pop[ip], pop[ip + sy_p]));

      /* Calculate upper face velocity.
       * diff >= 0 implies flow goes lower to upper */
      lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
      upper_cond = (pp[ip + sz_p] / dz) + 0.5 * dp[ip + sz_p] * gravity;
      diff = lower_cond - upper_cond;
      u_upper = ffz * PMean(pp[ip], pp[ip + sz_p],
                            permzp[ip], permzp[ip + sz_p])
                * diff
                * RPMean(lower_cond, upper_cond, rpp[ip] * dp[ip],
                         rpp[ip + sz_p] * dp[ip + sz_p]);
      zvp[ip] = 1.0 / ffz * u_upper / (AMean(sp[ip], sp[ip + sz_p]) * AMean(pop[ip], pop[ip + sz_p]));
      /*if (i>=0.0&&j>=0.0&&k>=0.0) printf ("%e %e %e\n",rpp[ip],dp[ip],
       *                                                 RPMean(lower_cond, upper_cond, rpp[ip]*dp[ip],
       *                                                  rpp[ip+sz_p]*dp[ip+sz_p]));
       * if (i>=0.0&&j>=0.0&&k>=0.0) printf ("zvp %e %e %e %e\n",zvp[ip],u_upper,lower_cond,upper_cond);*/
      //zvp[ip] = 1/ffz * u_upper / ( AMean(sp[ip],sp[ip+sz_p]) );

      fp[ip] += dt * (u_right + u_front + u_upper);
      fp[ip + 1] -= dt * u_right;
      fp[ip + sy_p] -= dt * u_front;
      fp[ip + sz_p] -= dt * u_upper;
      /*
       * if ((k == 0) && (i == 7) && (j == 0))
       * printf("Update stencil contribution: fp[ip] %12.8f \n"
       * "  u_upper %14.10f u_right %14.10f u_front %14.10f\n"
       * "  pp[ip] %12.8f \n"
       * "  pp[ip+1] %12.8f pp[ip+sy_p] %12.8f pp[ip+sz_p] %12.8f\n"
       * "  pp[ip-1] %12.8f pp[ip-sy_p] %12.8f pp[ip-sz_p] %12.8f\n"
       * "   Rel perms:  ip  %f ip+1 %f ip+sy_p %f ip+sz_p %f \n"
       * "   Densities:  ip  %f ip+1 %f ip+sy_p %f ip+sz_p %f \n",
       * fp[ip], u_upper, u_right, u_front,
       * pp[ip], pp[ip+1], pp[ip+sy_p], pp[ip+sz_p],
       * pp[ip-1], pp[ip-sy_p], pp[ip-sz_p],
       * rpp[ip], rpp[ip+1], rpp[ip+sy_p], rpp[ip+sz_p],
       * dp[ip], dp[ip+1], dp[ip+sy_p], dp[ip+sz_p]);
       */
    });
  }
#endif

  /*  Calculate correction for boundary conditions */
#if 1
  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    d_sub = VectorSubvector(density, is);
    rp_sub = VectorSubvector(rel_perm, is);
    f_sub = VectorSubvector(fval, is);
    permx_sub = VectorSubvector(permeability_x, is);
    permy_sub = VectorSubvector(permeability_y, is);
    permz_sub = VectorSubvector(permeability_z, is);

    p_sub = VectorSubvector(pressure, is);
    op_sub = VectorSubvector(old_pressure, is);
    s_sub = VectorSubvector(saturation, is);
    os_sub = VectorSubvector(old_saturation, is);
    po_sub = VectorSubvector(porosity, is);

    xv_sub = VectorSubvector(x_velocity, is);
    yv_sub = VectorSubvector(y_velocity, is);
    zv_sub = VectorSubvector(z_velocity, is);

    // sk Overland flow
    kw_sub = VectorSubvector(KW, is);
    ke_sub = VectorSubvector(KE, is);
    kn_sub = VectorSubvector(KN, is);
    ks_sub = VectorSubvector(KS, is);
    qx_sub = VectorSubvector(qx, is);
    qy_sub = VectorSubvector(qy, is);
    x_sl_sub = VectorSubvector(x_sl, is);
    y_sl_sub = VectorSubvector(y_sl, is);
    mann_sub = VectorSubvector(man, is);
    gnx = GetInt("ComputationalGrid.NX");
    gny = GetInt("ComputationalGrid.NY");
    obf_sub = VectorSubvector(ovrl_bc_flx, is);


    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);

    ffx = dy * dz;
    ffy = dx * dz;
    ffz = dx * dy;

    vol = dx * dy * dz;

    nx_p = SubvectorNX(p_sub);
    ny_p = SubvectorNY(p_sub);
    nz_p = SubvectorNZ(p_sub);

    sy_p = nx_p;
    sz_p = ny_p * nx_p;

    dp = SubvectorData(d_sub);
    rpp = SubvectorData(rp_sub);
    fp = SubvectorData(f_sub);
    permxp = SubvectorData(permx_sub);
    permyp = SubvectorData(permy_sub);
    permzp = SubvectorData(permz_sub);

    kw_ = SubvectorData(kw_sub);
    ke_ = SubvectorData(ke_sub);
    kn_ = SubvectorData(kn_sub);
    ks_ = SubvectorData(ks_sub);
    qx_ = SubvectorData(qx_sub);
    qy_ = SubvectorData(qy_sub);
    x_sl_dat = SubvectorData(x_sl_sub);
    y_sl_dat = SubvectorData(y_sl_sub);
    mann_dat = SubvectorData(mann_sub);
    obf_dat = SubvectorData(obf_sub);

    pp = SubvectorData(p_sub);
    opp = SubvectorData(op_sub);
    sp = SubvectorData(s_sub);
    osp = SubvectorData(os_sub);
    pop = SubvectorData(po_sub);

    xvp = SubvectorData(xv_sub);
    yvp = SubvectorData(yv_sub);
    zvp = SubvectorData(zv_sub);

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

            /* Don't currently do upstream weighting on boundaries */

            if (fdir[0])
            {
              switch (fdir[0])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - 1] - pp[ip];
                  u_old = ffx
                          * PMean(pp[ip - 1], pp[ip], permxp[ip - 1], permxp[ip])
                          * (diff / dx)
                          * RPMean(pp[ip - 1], pp[ip],
                                   rpp[ip - 1] * dp[ip - 1], rpp[ip] * dp[ip]);
                  diff = value - pp[ip];
                  u_new = RPMean(value, pp[ip],
                                 rpp[ip - 1] * dp[ip - 1], rpp[ip] * dp[ip]);
                  xvp[ip - 1] = 2.0 * u_new * permxp[ip] * diff / dx;
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + 1];
                  u_old = ffx
                          * PMean(pp[ip], pp[ip + 1], permxp[ip], permxp[ip + 1])
                          * (diff / dx)
                          * RPMean(pp[ip], pp[ip + 1],
                                   rpp[ip] * dp[ip], rpp[ip + 1] * dp[ip + 1]);
                  diff = pp[ip] - value;
                  u_new = RPMean(pp[ip], value,
                                 rpp[ip] * dp[ip], rpp[ip + 1] * dp[ip + 1]);
                  xvp[ip] = 2.0 * u_new * permxp[ip] * diff / dx;
                  break;
              }
              u_new = u_new * ffx * permxp[ip] * 2.0 * (diff / dx);
              /*
               * if ((k == 0) && (i == 7) && (j == 0))
               * printf("Right BC u_new %12.8f u_old %12.8f value %12.8f "
               *    "dir %i RPVal %f\n",
               *    u_new, u_old, value, dir,(RPMean(pp[ip], value,
               *                rpp[ip]*dp[ip], rpp[ip+1]*dp[ip+1])));
               */
            }
            else if (fdir[1])
            {
              switch (fdir[1])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - sy_p] - pp[ip];
                  u_old = ffy
                          * PMean(pp[ip - sy_p], pp[ip],
                                  permyp[ip - sy_p], permyp[ip])
                          * (diff / dy)
                          * RPMean(pp[ip - sy_p], pp[ip],
                                   rpp[ip - sy_p] * dp[ip - sy_p], rpp[ip] * dp[ip]);
                  diff = value - pp[ip];
                  u_new = RPMean(value, pp[ip],
                                 rpp[ip - sy_p] * dp[ip - sy_p], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + sy_p];
                  u_old = ffy
                          * PMean(pp[ip], pp[ip + sy_p],
                                  permyp[ip], permyp[ip + sy_p])
                          * (diff / dy)
                          * RPMean(pp[ip], pp[ip + sy_p],
                                   rpp[ip] * dp[ip], rpp[ip + sy_p] * dp[ip + sy_p]);
                  diff = pp[ip] - value;
                  u_new = RPMean(pp[ip], value,
                                 rpp[ip] * dp[ip], rpp[ip + sy_p] * dp[ip + sy_p]);
                  break;
              }
              u_new = u_new * ffy * (permyp[ip])
                      * 2.0 * (diff / dy);
              /*
               * if ((k == 0) && (i == 0) && (j == 0))
               * printf("Front BC u_new %12.8f u_old %12.8f value %12.8f "
               *    "dir %i\n",
               *    u_new, u_old, value, dir);
               */
            }
            else if (fdir[2])
            {
              switch (fdir[2])
              {
                case -1:
                  {
                    dir = -1;
                    lower_cond = (pp[ip - sz_p] / dz)
                                 - 0.5 * dp[ip - sz_p] * gravity;
                    upper_cond = (pp[ip] / dz) + 0.5 * dp[ip] * gravity;
                    diff = lower_cond - upper_cond;

                    u_old = ffz
                            * PMean(pp[ip - sz_p], pp[ip],
                                    permzp[ip - sz_p], permzp[ip])
                            * diff
                            * RPMean(lower_cond, upper_cond,
                                     rpp[ip - sz_p] * dp[ip - sz_p], rpp[ip] * dp[ip]);

                    lower_cond = (value / dz) - 0.25 * dp[ip] * gravity;
                    upper_cond = (pp[ip] / dz) + 0.25 * dp[ip] * gravity;
                    diff = lower_cond - upper_cond;
                    u_new = RPMean(lower_cond, upper_cond,
                                   rpp[ip - sz_p] * dp[ip - sz_p], rpp[ip] * dp[ip]);

                    zvp[ip] = permzp[ip] / (sp[ip] * pop[ip]) * 2.0 * diff * u_new;
                    break;
                  }      /* End case -1 */

                case  1:
                  {
                    dir = 1;
                    lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
                    upper_cond = (pp[ip + sz_p] / dz)
                                 - 0.5 * dp[ip + sz_p] * gravity;
                    diff = lower_cond - upper_cond;
                    u_old = ffz
                            * PMean(pp[ip], pp[ip + sz_p],
                                    permzp[ip], permzp[ip + sz_p])
                            * diff
                            * RPMean(lower_cond, upper_cond,
                                     rpp[ip] * dp[ip], rpp[ip + sz_p] * dp[ip + sz_p]);
                    lower_cond = (pp[ip] / dz) - 0.25 * dp[ip] * gravity;
                    upper_cond = (value / dz) + 0.25 * dp[ip] * gravity;
                    diff = lower_cond - upper_cond;
                    u_new = RPMean(lower_cond, upper_cond,
                                   rpp[ip] * dp[ip], rpp[ip + sz_p] * dp[ip + sz_p]);

                    zvp[ip] = permzp[ip] / (sp[ip] * pop[ip]) * 2.0 * diff * u_new;
                    break;
                  }      /* End case 1 */
              }
              u_new = u_new * ffz * (permzp[ip])
                      * 2.0 * diff;
              /*
               * if ((k == 25) && (i==1) && (j == 0) )
               * printf("Upper BC u_new %12.8f u_old %12.8f value %12.8f\n"
               *    "   rpp[ip] %12.8f rpp[ip+sz_p] %12.8f "
               *    "dp[ip] %12.8f\n"
               *    "   dp[ip+sz_p] %12.8f diff %12.8f permp[ip] %12.8f\n",
               *    u_new, u_old, value, rpp[ip], rpp[ip+sz_p],
               *    dp[ip], dp[ip+sz_p], diff, permp[ip]);
               */
            }
            /*
             * if ( (k == 25) && (i == 1) && (j == 0))
             * printf("f before BC additions: %12.8f \n", fp[ip]);
             */

            /* Remove the boundary term computed above */
            fp[ip] -= dt * dir * u_old;

            /* Add the correct boundary term */
            fp[ip] += dt * dir * u_new;
            /*
             * if ( (k == 25) && (i == 1) && (j == 0))
             * printf("f after BC additions: %12.8f \n\n",
             *        fp[ip]);
             */
          });

          break;
        }

        case FluxBC:
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            ip = SubvectorEltIndex(p_sub, i, j, k);

            if (fdir[0])
            {
              switch (fdir[0])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - 1] - pp[ip];
                  u_old = ffx * PMean(pp[ip - 1], pp[ip],
                                      permxp[ip - 1], permxp[ip])
                          * (diff / dx)
                          * RPMean(pp[ip - 1], pp[ip],
                                   rpp[ip - 1] * dp[ip - 1], rpp[ip] * dp[ip]);
                  xvp[ip - 1] = bc_patch_values[ival];
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + 1];
                  u_old = ffx * PMean(pp[ip], pp[ip + 1],
                                      permxp[ip], permxp[ip + 1])
                          * (diff / dx)
                          * RPMean(pp[ip], pp[ip + 1],
                                   rpp[ip] * dp[ip], rpp[ip + 1] * dp[ip + 1]);
                  xvp[ip] = bc_patch_values[ival];
                  break;
              }
              u_new = ffx;
            }
            else if (fdir[1])
            {
              switch (fdir[1])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - sy_p] - pp[ip];
                  u_old = ffy * PMean(pp[ip - sy_p], pp[ip],
                                      permyp[ip - sy_p], permyp[ip])
                          * (diff / dy)
                          * RPMean(pp[ip - sy_p], pp[ip],
                                   rpp[ip - sy_p] * dp[ip - sy_p], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + sy_p];
                  u_old = ffy * PMean(pp[ip], pp[ip + sy_p],
                                      permyp[ip], permyp[ip + sy_p])
                          * (diff / dy)
                          * RPMean(pp[ip], pp[ip + sy_p],
                                   rpp[ip] * dp[ip], rpp[ip + sy_p] * dp[ip + sy_p]);
                  break;
              }
              u_new = ffy;
            }
            else if (fdir[2])
            {
              switch (fdir[2])
              {
                case -1:
                  dir = -1;
                  lower_cond = (pp[ip - sz_p] / dz)
                               - 0.5 * dp[ip - sz_p] * gravity;
                  upper_cond = (pp[ip] / dz) + 0.5 * dp[ip] * gravity;
                  diff = lower_cond - upper_cond;
                  u_old = ffz * PMean(pp[ip - sz_p], pp[ip],
                                      permzp[ip - sz_p], permzp[ip])
                          * diff
                          * RPMean(lower_cond, upper_cond,
                                   rpp[ip - sz_p] * dp[ip - sz_p], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
                  upper_cond = (pp[ip + sz_p] / dz)
                               + 0.5 * dp[ip + sz_p] * gravity;
                  diff = lower_cond - upper_cond;
                  u_old = ffz * PMean(0, 0, permzp[ip], permzp[ip + sz_p])
                          * diff
                          * RPMean(lower_cond, upper_cond,
                                   rpp[ip] * dp[ip], rpp[ip + sz_p] * dp[ip + sz_p]);
                  break;
              }
              u_new = ffz;

              /* Velocity in z-direction */
              zvp[ip] = bc_patch_values[ival] / (sp[ip] * pop[ip]);
            }

            /* Remove the boundary term computed above */
            fp[ip] -= dt * dir * u_old;
            /*
             * if ((fdir[2] > 0) && (i == 0) && (j == 0))
             * printf("f before flux BC additions: %12.8f \n", fp[ip]);
             */
            /* Add the correct boundary term */
            u_new = u_new * bc_patch_values[ival];
            fp[ip] += dt * dir * u_new;
            /*
             * if ((fdir[2] < 0) && (i == 0) && (j == 0))
             * printf("f after flux BC additions: %12.8f \n\n",
             *        fp[ip]);
             */
          });

          break;
        }      /* End fluxbc case */

        case OverlandBC:
        {
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            ip = SubvectorEltIndex(p_sub, i, j, k);

            if (fdir[0])
            {
              switch (fdir[0])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - 1] - pp[ip];
                  u_old = ffx * PMean(pp[ip - 1], pp[ip],
                                      permxp[ip - 1], permxp[ip])
                          * (diff / dx)
                          * RPMean(pp[ip - 1], pp[ip],
                                   rpp[ip - 1] * dp[ip - 1], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + 1];
                  u_old = ffx * PMean(pp[ip], pp[ip + 1],
                                      permxp[ip], permxp[ip + 1])
                          * (diff / dx)
                          * RPMean(pp[ip], pp[ip + 1],
                                   rpp[ip] * dp[ip], rpp[ip + 1] * dp[ip + 1]);
                  break;
              }
              u_new = ffx;
            }
            else if (fdir[1])
            {
              switch (fdir[1])
              {
                case -1:
                  dir = -1;
                  diff = pp[ip - sy_p] - pp[ip];
                  u_old = ffy * PMean(pp[ip - sy_p], pp[ip],
                                      permyp[ip - sy_p], permyp[ip])
                          * (diff / dy)
                          * RPMean(pp[ip - sy_p], pp[ip],
                                   rpp[ip - sy_p] * dp[ip - sy_p], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  diff = pp[ip] - pp[ip + sy_p];
                  u_old = ffy * PMean(pp[ip], pp[ip + sy_p],
                                      permyp[ip], permyp[ip + sy_p])
                          * (diff / dy)
                          * RPMean(pp[ip], pp[ip + sy_p],
                                   rpp[ip] * dp[ip], rpp[ip + sy_p] * dp[ip + sy_p]);
                  break;
              }
              u_new = ffy;
            }
            else if (fdir[2])
            {
              switch (fdir[2])
              {
                case -1:
                  dir = -1;
                  lower_cond = (pp[ip - sz_p] / dz)
                               - 0.5 * dp[ip - sz_p] * gravity;
                  upper_cond = (pp[ip] / dz) + 0.5 * dp[ip] * gravity;
                  diff = lower_cond - upper_cond;
                  u_old = ffz * PMean(pp[ip - sz_p], pp[ip],
                                      permzp[ip - sz_p], permzp[ip])
                          * diff
                          * RPMean(lower_cond, upper_cond,
                                   rpp[ip - sz_p] * dp[ip - sz_p], rpp[ip] * dp[ip]);
                  break;

                case  1:
                  dir = 1;
                  lower_cond = (pp[ip] / dz) - 0.5 * dp[ip] * gravity;
                  upper_cond = (pp[ip + sz_p] / dz)
                               + 0.5 * dp[ip + sz_p] * gravity;
                  diff = lower_cond - upper_cond;
                  u_old = ffz * PMean(0, 0, permzp[ip], permzp[ip + sz_p])
                          * diff
                          * RPMean(lower_cond, upper_cond,
                                   rpp[ip] * dp[ip], rpp[ip + sz_p] * dp[ip + sz_p]);
                  break;
              }
              u_new = ffz;
            }

            /* Remove the boundary term computed above */
            fp[ip] -= dt * dir * u_old;

            /*
             * if ((fdir[2] > 0) && (i == 0) && (j == 0))
             * printf("f before flux BC additions: %12.8f \n", fp[ip]);
             */
            //printf("u_new %e %e %e %e\n",u_new,dx,dy,dz);
            u_new = u_new * bc_patch_values[ival];        //sk: here we go in and implement surface routing!

            fp[ip] += dt * dir * u_new;
            /*
             * if ((fdir[2] < 0) && (i == 0) && (j == 0))
             * printf("f after flux BC additions: %12.8f \n\n",
             *        fp[ip]);
             */
          });


          BCStructPatchLoopOvrlnd(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            if (fdir[2])
            {
              switch (fdir[2])
              {
                case 1:
                  io = SubvectorEltIndex(p_sub, i, j, 0);
                  ip = SubvectorEltIndex(p_sub, i, j, k);
                  io = SubvectorEltIndex(p_sub, i, j, 0);

                  dir_x = 0.0;
                  dir_y = 0.0;
                  if (x_sl_dat[io] > 0.0)
                    dir_x = -1.0;
                  if (y_sl_dat[io] > 0.0)
                    dir_y = -1.0;
                  if (x_sl_dat[io] < 0.0)
                    dir_x = 1.0;
                  if (y_sl_dat[io] < 0.0)
                    dir_y = 1.0;

                  qx_[io] = dir_x * (RPowerR(fabs(x_sl_dat[io]), 0.5) / mann_dat[io]) * RPowerR(max((pp[ip]), 0.0), (5.0 / 3.0));
                  qy_[io] = dir_y * (RPowerR(fabs(y_sl_dat[io]), 0.5) / mann_dat[io]) * RPowerR(max((pp[ip]), 0.0), (5.0 / 3.0));
                  break;
              }
            }
          });

          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            if (fdir[2])
            {
              switch (fdir[2])
              {
                case 1:
                  io = SubvectorEltIndex(p_sub, i, j, 0);

                  ke_[io] = max(qx_[io], 0.0) - max(-qx_[io + 1], 0.0);
                  kw_[io] = max(qx_[io - 1], 0.0) - max(-qx_[io], 0.0);

                  kn_[io] = max(qy_[io], 0.0) - max(-qy_[io + sy_p], 0.0);
                  ks_[io] = max(qy_[io - sy_p], 0.0) - max(-qy_[io], 0.0);

                  break;
              }
            }
          });


          *outflow = 0.0;
          BCStructPatchLoop(i, j, k, fdir, ival, bc_struct, ipatch, is,
          {
            if (fdir[2])
            {
              switch (fdir[2])
              {
                case 1:
                  dir = 1;
                  ip = SubvectorEltIndex(p_sub, i, j, k);
                  io = SubvectorEltIndex(p_sub, i, j, 0);

                  q_overlnd = 0.0;
                  q_overlnd = vol * (max(pp[ip], 0.0) - max(opp[ip], 0.0)) / dz +
                              dt * vol * ((ke_[io] - kw_[io]) / dx + (kn_[io] - ks_[io]) / dy) / dz;

                  /*if ( i == 0 && j == 0){ //left-lower corner
                   * obf_dat[io]= fabs(kw_[io]) + fabs(ks_[io]);
                   * } else if (i == 0 && j > 0 && j < (gny-1)) { // west face
                   * obf_dat[io]= fabs(kw_[io]);
                   * } else if (i == (gnx-1) && j == (gny-1)) { //right-upper corner
                   * obf_dat[io]= fabs(ke_[io]) + fabs(kn_[io]);
                   * } else if (j == 0 && i > 0 && i < (gnx-1)) { //south face
                   * obf_dat[io]= fabs(ks_[io]);
                   * } else if ( i == (gnx-1) && j == 0 ) { //right-lower corner
                   * obf_dat[io] = fabs(ks_[io]) + fabs(ke_[io]);
                   * } else if (i == (gnx-1) && j > 0 && j < (gny-1)) { //east face
                   * obf_dat[io]= fabs(ke_[io]);
                   * } else if (i > 0 && i < (gnx-1) && j == (gny-1)) { //north face
                   * obf_dat[io] = kn_[io];
                   * } else if (i == 0 && j == (gny-1)) { //left-upper corner
                   * obf_dat[io] = fabs(kw_[io]) + fabs(kn_[io]);
                   * } else { //interior
                   * obf_dat[io] = qx_[io];
                   * }*/

                  obf_dat[io] = 0.0;
                  if (i >= 0 && i <= (gnx - 1) && j == 0 && qy_[io] < 0.0)   //south face
                  {
                    obf_dat[io] += fabs(qy_[io]);
                  }
                  else if (i == 0 && j >= 0 && j <= (gny - 1) && qx_[io] < 0.0)    // west face
                  {
                    obf_dat[io] += fabs(qx_[io]);
                  }
                  else if (i >= 0 && i <= (gnx - 1) && j == (gny - 1) && qy_[io] > 0.0)  //north face
                  {
                    obf_dat[io] += fabs(qy_[io]);
                  }
                  else if (i == (gnx - 1) && j >= 0 && j <= (gny - 1) && qx_[io] > 0.0)  //east face
                  {
                    obf_dat[io] += fabs(qx_[io]);
                  }
                  else if (i > 0 && i < (gnx - 1) && j > 0 && j < (gny - 1))  //interior
                  {
                    obf_dat[io] = qx_[io];
                  }


                  if (j == 0 && i == 0)
                  {
                    *outflow = fabs(ks_[io]) + fabs(kw_[io]);
                  }

                  fp[ip] += q_overlnd;

                  break;
              }
            }
          });

          break;
        }      /* End OverlandBC case */
      }        /* End switch BCtype */
    }          /* End ipatch loop */
  }            /* End subgrid loop */
#endif
  FreeBCStruct(bc_struct);
#if 1
  PFModuleInvoke(void, bc_internal, (problem, problem_data, fval, NULL,
                                     time, pressure, CALCFCN));
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

    p_sub = VectorSubvector(pressure, is);
    f_sub = VectorSubvector(fval, is);

    /* RDF: assumes resolutions are the same in all 3 directions */
    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    pp = SubvectorData(p_sub);
    fp = SubvectorData(f_sub);

    GrGeomOutLoop(i, j, k, gr_domain,
                  r, ix, iy, iz, nx, ny, nz,
    {
      ip = SubvectorEltIndex(f_sub, i, j, k);
      fp[ip] = pp[ip];
    });
  }
#endif

  EndTiming(public_xtra->time_index);

  FreeVector(KW);
  FreeVector(KE);
  FreeVector(KN);
  FreeVector(KS);
  FreeVector(qx);
  FreeVector(qy);
  FreeVector(permeability_x);
  FreeVector(permeability_y);
  FreeVector(permeability_z);

  return;
}


/*--------------------------------------------------------------------------
 * PressFunctionEvalInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *PressFunctionEvalInitInstanceXtra(problem, grid, temp_data)
Problem * problem;
Grid        *grid;
double      *temp_data;
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (grid != NULL)
  {
    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeVector(instance_xtra->rel_perm);
      FreeVector(instance_xtra->qsource);
      FreeVector(instance_xtra->tsource);
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    (instance_xtra->rel_perm) = NewVector(grid, 1, 1);
    (instance_xtra->qsource) = NewVector(grid, 1, 1);
    (instance_xtra->tsource) = NewVector(grid, 1, 1);
  }

  if (problem != NULL)
  {
    (instance_xtra->problem) = problem;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->density_module) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
    (instance_xtra->viscosity_module) =
      PFModuleNewInstance(ProblemPhaseViscosity(problem), ());
    (instance_xtra->saturation_module) =
      PFModuleNewInstance(ProblemSaturation(problem), (NULL, NULL));
    (instance_xtra->rel_perm_module) =
      PFModuleNewInstance(ProblemPhaseRelPerm(problem), (NULL, NULL));
    (instance_xtra->phase_source) =
      PFModuleNewInstance(ProblemPhaseSource(problem), (grid));
    (instance_xtra->bc_pressure) =
      PFModuleNewInstance(ProblemBCPressure(problem), (problem));
    (instance_xtra->bc_internal) =
      PFModuleNewInstance(ProblemBCInternal(problem), ());
  }
  else
  {
    PFModuleReNewInstance((instance_xtra->density_module), ());
    PFModuleReNewInstance((instance_xtra->viscosity_module), ());
    PFModuleReNewInstance((instance_xtra->saturation_module),
                          (NULL, NULL));
    PFModuleReNewInstance((instance_xtra->rel_perm_module),
                          (NULL, NULL));
    PFModuleReNewInstance((instance_xtra->phase_source), (NULL));
    PFModuleReNewInstance((instance_xtra->bc_pressure), (problem));
    PFModuleReNewInstance((instance_xtra->bc_internal), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * PressFunctionEvalFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  PressFunctionEvalFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->density_module);
    PFModuleFreeInstance(instance_xtra->viscosity_module);
    PFModuleFreeInstance(instance_xtra->saturation_module);
    PFModuleFreeInstance(instance_xtra->rel_perm_module);
    PFModuleFreeInstance(instance_xtra->phase_source);
    PFModuleFreeInstance(instance_xtra->bc_pressure);
    PFModuleFreeInstance(instance_xtra->bc_internal);

    FreeVector(instance_xtra->rel_perm);
    FreeVector(instance_xtra->qsource);
    FreeVector(instance_xtra->tsource);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PressFunctionEvalNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *PressFunctionEvalNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;


  public_xtra = ctalloc(PublicXtra, 1);

  (public_xtra->time_index) = RegisterTiming("NL_F_Eval");

  PFModulePublicXtra(this_module) = public_xtra;

  return this_module;
}


/*--------------------------------------------------------------------------
 * PressFunctionEvalFreePublicXtra
 *--------------------------------------------------------------------------*/

void  PressFunctionEvalFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * PressFunctionEvalSizeOfTempData
 *--------------------------------------------------------------------------*/

int  PressFunctionEvalSizeOfTempData()
{
  return 0;
}



