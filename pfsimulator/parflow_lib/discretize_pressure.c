/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
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

/*****************************************************************************
*
* Discretize the pressure equation using 7-point finite volumes.
*
*-----------------------------------------------------------------------------
*
*****************************************************************************/

#include "parflow.h"


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef struct {
  PFModule          *bc_pressure;
  PFModule          *phase_mobility;
  PFModule          *phase_density_module;
  PFModule          *capillary_pressure;
  PFModule          *phase_source;

  /* InitInstanceXtra arguments */
  Problem  *problem;
  Grid     *grid;
  double   *temp_data;

  /* instance data */
  Matrix   *A;
  Vector   *f;
} InstanceXtra;


/*--------------------------------------------------------------------------
 * Static stencil-shape definition
 *--------------------------------------------------------------------------*/

int seven_pt_shape[7][3] = { { 0, 0, 0 },
                             { -1, 0, 0 },
                             { 1, 0, 0 },
                             { 0, -1, 0 },
                             { 0, 1, 0 },
                             { 0, 0, -1 },
                             { 0, 0, 1 } };


/*--------------------------------------------------------------------------
 * Define macros for the DiscretizePressure routine
 *--------------------------------------------------------------------------*/

#define Mean(a, b)  CellFaceConductivity(a, b)

#define Coeff(Ta, Tb, Pa, Pb) \
        (((Ta) + (Tb)) ? ((Ta) * (Pb) + (Tb) * (Pa)) / ((Ta) + (Tb)) : 0)


/*--------------------------------------------------------------------------
 * DiscretizePressure
 *--------------------------------------------------------------------------*/

void          DiscretizePressure(
                                 Matrix **    ptr_to_A,
                                 Vector **    ptr_to_f,
                                 ProblemData *problem_data,
                                 double       time,
                                 Vector *     total_mobility_x,
                                 Vector *     total_mobility_y,
                                 Vector *     total_mobility_z,
                                 Vector **    phase_saturations)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  PFModule       *bc_pressure = (instance_xtra->bc_pressure);
  PFModule       *phase_mobility = (instance_xtra->phase_mobility);
  PFModule       *phase_density_module =
    (instance_xtra->phase_density_module);
  PFModule       *capillary_pressure =
    (instance_xtra->capillary_pressure);
  PFModule       *phase_source = (instance_xtra->phase_source);

  Problem        *problem = (instance_xtra->problem);

  Grid           *grid = (instance_xtra->grid);

  Matrix         *A = (instance_xtra->A);
  Vector         *f = (instance_xtra->f);

  Vector        **tmobility_x = NULL;
  Vector        **tmobility_y = NULL;
  Vector        **tmobility_z = NULL;
  Vector        **tcapillary = NULL;
  Vector         *tvector = NULL;

  int num_phases = ProblemNumPhases(problem);

  GrGeomSolid    *gr_domain = ProblemDataGrDomain(problem_data);

  double gravity, dtmp, *phase_density;

  VectorUpdateCommHandle     *handle;

  BCStruct       *bc_struct;
  double         *bc_patch_values;
  int            *fdir;

  Subgrid        *subgrid;

  Submatrix      *A_sub;
  Subvector      *f_sub;

  Subvector      *ttx_sub, *tty_sub, *ttz_sub;
  Subvector     **tmx_sub, **tmy_sub, **tmz_sub;
  Subvector     **tc_sub;
  Subvector      *tv_sub;

  double dx, dy, dz, d = 0.0;

  double         *cp, *wp, *ep, *sp, *np, *lp, *up, *op = NULL;
  double         *fp;
  double         *ttx_p, *tty_p, *ttz_p;
  double         *tmx_p, *tmy_p, *tmz_p;
  double         *tm_p, *tt_p = NULL;
  double        **tmx_pvec, **tmy_pvec, **tmz_pvec;
  double         *tc_p, **tc_pvec;
  double         *tv_p;

  double scale, ffx, ffy, ffz, ff = 0.0, vf;

  double e_temp, n_temp, u_temp, f_temp;
  double o_temp;

  int r;
  int ix, iy, iz;
  int nx, ny, nz;
  int nx_v, ny_v, nz_v;
  int nx_m, ny_m, nz_m;

  int iv, im, ival;
  int sy_v, sz_v, sv = 0;
  int sy_m, sz_m;

  int phase, ipatch, is, i, j, k;

#ifdef SHMEM_OBJECTS
  int p;
#endif

  /*-----------------------------------------------------------------------
   * Allocate temp vectors
   *-----------------------------------------------------------------------*/
  tmobility_x = ctalloc(Vector *, ProblemNumPhases(problem));
  tmobility_y = ctalloc(Vector *, ProblemNumPhases(problem));
  tmobility_z = ctalloc(Vector *, ProblemNumPhases(problem));
  tcapillary = ctalloc(Vector *, ProblemNumPhases(problem));

  for (phase = 0; phase < num_phases; phase++)
  {
    tmobility_x[phase] = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
    tmobility_y[phase] = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
    tmobility_z[phase] = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
    tcapillary[phase] = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);
  }
  tvector = NewVectorType(instance_xtra->grid, 1, 1, vector_cell_centered);

  /*-----------------------------------------------------------------------
   * Initialize and set some things
   *-----------------------------------------------------------------------*/

  /* Allocate several miscellaneous arrays */
  phase_density = talloc(double, num_phases);
  tmx_sub = talloc(Subvector *, num_phases);
  tmy_sub = talloc(Subvector *, num_phases);
  tmz_sub = talloc(Subvector *, num_phases);
  tc_sub = talloc(Subvector *, num_phases);
  tmx_pvec = talloc(double *, num_phases);
  tmy_pvec = talloc(double *, num_phases);
  tmz_pvec = talloc(double *, num_phases);
  tc_pvec = talloc(double *, num_phases);

  /* Initialize A and f */
  InitMatrix(A, 0.0);
  InitVector(f, 0.0);

  /* Compute scaling factor */
  scale = pfmin(RealSpaceDX(0), pfmin(RealSpaceDY(0), RealSpaceDZ(0)));
  scale = 1 / scale;

  gravity = ProblemGravity(problem);

  for (phase = 0; phase < num_phases; phase++)
  {
    /* Assume constant density here.  Use dtmp as dummy argument. */
    PFModuleInvokeType(PhaseDensityInvoke, phase_density_module,
                       (phase, NULL, NULL, &dtmp, &phase_density[phase],
                        CALCFCN));

    PFModuleInvokeType(PhaseMobilityInvoke, phase_mobility,
                       (tmobility_x[phase], tmobility_y[phase],
                        tmobility_z[phase],
                        ProblemDataPermeabilityX(problem_data),
                        ProblemDataPermeabilityY(problem_data),
                        ProblemDataPermeabilityZ(problem_data),
                        phase, phase_saturations[phase],
                        ProblemPhaseViscosity(problem, phase)));

    PFModuleInvokeType(CapillaryPressureInvoke, capillary_pressure,
                       (tcapillary[phase], phase, 0,
                        problem_data, phase_saturations[0]));

    /* update ghost-point values */
    handle = InitVectorUpdate(tmobility_x[phase], VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    handle = InitVectorUpdate(tmobility_y[phase], VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    handle = InitVectorUpdate(tmobility_z[phase], VectorUpdateAll);
    FinalizeVectorUpdate(handle);

    handle = InitVectorUpdate(tcapillary[phase], VectorUpdateAll);
    FinalizeVectorUpdate(handle);
  }

  /*-----------------------------------------------------------------------
   * Add phase_source contributions
   *-----------------------------------------------------------------------*/


  for (phase = 0; phase < num_phases; phase++)
  {
    /* get phase_source */
    PFModuleInvokeType(PhaseSourceInvoke, phase_source, (tvector, phase, problem,
                                                         problem_data, time));

    ForSubgridI(is, GridSubgrids(grid))
    {
      subgrid = GridSubgrid(grid, is);

      f_sub = VectorSubvector(f, is);
      tv_sub = VectorSubvector(tvector, is);

      ix = SubgridIX(subgrid);
      iy = SubgridIY(subgrid);
      iz = SubgridIZ(subgrid);

      nx = SubgridNX(subgrid);
      ny = SubgridNY(subgrid);
      nz = SubgridNZ(subgrid);

      dx = SubgridDX(subgrid);
      dy = SubgridDY(subgrid);
      dz = SubgridDZ(subgrid);

      vf = dx * dy * dz * scale;

      nx_v = SubvectorNX(f_sub);
      ny_v = SubvectorNY(f_sub);
      nz_v = SubvectorNZ(f_sub);

      fp = SubvectorData(f_sub);
      tv_p = SubvectorData(tv_sub);

      iv = SubvectorEltIndex(f_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        fp[iv] += vf * tv_p[iv];
      });
    }
  }

  /*-----------------------------------------------------------------------
   * Discretize pressure_0 derivative terms
   *   Note - To simplify the loop, we compute more coefficients than
   *   are needed on this process.  We can do this only if A has a
   *   ghost layer.  This is insured in the InitInstanceXtra routine.
   *-----------------------------------------------------------------------*/

  ForSubgridI(is, GridSubgrids(grid))
  {
    f_sub = VectorSubvector(f, is);

    subgrid = GridSubgrid(grid, is);

    A_sub = MatrixSubmatrix(A, is);
    ttx_sub = VectorSubvector(total_mobility_x, is);
    tty_sub = VectorSubvector(total_mobility_y, is);
    ttz_sub = VectorSubvector(total_mobility_z, is);

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid) + 1;
    ny = SubgridNY(subgrid) + 1;
    nz = SubgridNZ(subgrid) + 1;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ffx = dy * dz * scale;
    ffy = dx * dz * scale;
    ffz = dx * dy * scale;

    nx_m = SubmatrixNX(A_sub);
    ny_m = SubmatrixNY(A_sub);
    nz_m = SubmatrixNZ(A_sub);

    nx_v = SubvectorNX(f_sub);
    ny_v = SubvectorNY(f_sub);
    nz_v = SubvectorNZ(f_sub);

    sy_v = nx_v;
    sy_m = nx_m;
    sz_v = ny_v * nx_v;
    sz_m = ny_m * nx_m;

    cp = SubmatrixStencilData(A_sub, 0);
    wp = SubmatrixStencilData(A_sub, 1);
    ep = SubmatrixStencilData(A_sub, 2);
    sp = SubmatrixStencilData(A_sub, 3);
    np = SubmatrixStencilData(A_sub, 4);
    lp = SubmatrixStencilData(A_sub, 5);
    up = SubmatrixStencilData(A_sub, 6);
    ttx_p = SubvectorData(ttx_sub);
    tty_p = SubvectorData(tty_sub);
    ttz_p = SubvectorData(ttz_sub);

    iv = SubvectorEltIndex(ttx_sub, ix, iy, iz);
    im = SubmatrixEltIndex(A_sub, ix, iy, iz);

    BoxLoopI2(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
              im, nx_m, ny_m, nz_m, 1, 1, 1,
    {
      e_temp = -ffx * Mean(ttx_p[iv], ttx_p[iv + 1]) / dx;
      n_temp = -ffy * Mean(tty_p[iv], tty_p[iv + sy_v]) / dy;
      u_temp = -ffz * Mean(ttz_p[iv], ttz_p[iv + sz_v]) / dz;

      ep[im] += e_temp;
      np[im] += n_temp;
      up[im] += u_temp;
      cp[im] -= e_temp + n_temp + u_temp;

      cp[im + 1] -= e_temp;
      cp[im + sy_m] -= n_temp;
      cp[im + sz_m] -= u_temp;
    });
  }

  /*-----------------------------------------------------------------------
   * Discretize capillary pressure and gravity derivative terms
   *   See note above for pressure_0 derivative term
   *-----------------------------------------------------------------------*/

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    f_sub = VectorSubvector(f, is);
    ttx_sub = VectorSubvector(total_mobility_x, is);
    tty_sub = VectorSubvector(total_mobility_y, is);
    ttz_sub = VectorSubvector(total_mobility_z, is);
    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_sub[phase] = VectorSubvector(tmobility_x[phase], is);
      tmy_sub[phase] = VectorSubvector(tmobility_y[phase], is);
      tmz_sub[phase] = VectorSubvector(tmobility_z[phase], is);
      tc_sub[phase] = VectorSubvector(tcapillary[phase], is);
    }

    ix = SubgridIX(subgrid) - 1;
    iy = SubgridIY(subgrid) - 1;
    iz = SubgridIZ(subgrid) - 1;

    nx = SubgridNX(subgrid) + 1;
    ny = SubgridNY(subgrid) + 1;
    nz = SubgridNZ(subgrid) + 1;

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ffx = dy * dz * scale;
    ffy = dx * dz * scale;
    ffz = dx * dy * scale;

    nx_v = SubvectorNX(f_sub);
    ny_v = SubvectorNY(f_sub);
    nz_v = SubvectorNZ(f_sub);

    sy_v = nx_v;
    sz_v = ny_v * nx_v;

    fp = SubvectorData(f_sub);
    ttx_p = SubvectorData(ttx_sub);
    tty_p = SubvectorData(tty_sub);
    ttz_p = SubvectorData(ttz_sub);


#ifdef SHMEM_OBJECTS
    {
      double *tmp_ptr;
      int index;

      tmp_ptr = malloc(nx_v * ny_v * nz_v * sizeof(double));

      for (index = 0; index < nx_v * ny_v * nz_v; index++)
      {
        tmp_ptr[index] = fp[index];
      }

      fp = tmp_ptr;
    }
#endif


    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_p = SubvectorData(tmx_sub[phase]);
      tmy_p = SubvectorData(tmy_sub[phase]);
      tmz_p = SubvectorData(tmz_sub[phase]);
      tc_p = SubvectorData(tc_sub[phase]);

      iv = SubvectorEltIndex(f_sub, ix, iy, iz);

      BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
                iv, nx_v, ny_v, nz_v, 1, 1, 1,
      {
        e_temp = -ffx * Coeff(ttx_p[iv], ttx_p[iv + 1],
                              tmx_p[iv], tmx_p[iv + 1])
                 / dx;
        n_temp = -ffy * Coeff(tty_p[iv], tty_p[iv + sy_v],
                              tmy_p[iv], tmy_p[iv + sy_v])
                 / dy;
        u_temp = -ffz * Coeff(ttz_p[iv], ttz_p[iv + sz_v],
                              tmz_p[iv], tmz_p[iv + sz_v])
                 / dz;

        f_temp = ffz * Coeff(ttz_p[iv], ttz_p[iv + sz_v],
                             tmz_p[iv], tmz_p[iv + sz_v]) *
                 (phase_density[phase] * gravity);

        fp[iv] += f_temp;

        fp[iv + sz_v] -= f_temp;

        /* capillary pressure contribution */
        fp[iv] -= (e_temp * (tc_p[iv + 1] - tc_p[iv]) +
                   n_temp * (tc_p[iv + sy_v] - tc_p[iv]) +
                   u_temp * (tc_p[iv + sz_v] - tc_p[iv]));
        fp[iv + 1] += e_temp * (tc_p[iv + 1] - tc_p[iv]);
        fp[iv + sy_v] += n_temp * (tc_p[iv + sy_v] - tc_p[iv]);
        fp[iv + sz_v] += u_temp * (tc_p[iv + sz_v] - tc_p[iv]);
      });
    }
  }

#ifdef SHMEM_OBJECTS
  {
    double *tmp_ptr;

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    vf = dx * dy * dz * scale;

    nx_v = SubvectorNX(f_sub);
    ny_v = SubvectorNY(f_sub);
    nz_v = SubvectorNZ(f_sub);

    tmp_ptr = SubvectorData(f_sub);

    iv = SubvectorEltIndex(f_sub, ix, iy, iz);

    BoxLoopI1(i, j, k, ix, iy, iz, nx, ny, nz,
              iv, nx_v, ny_v, nz_v, 1, 1, 1,
    {
      tmp_ptr[iv] = fp[iv];
    });

    free(fp);
  }
#endif

  /*-----------------------------------------------------------------------
   * Fix up boundaries and impose default no-flow condition
   *-----------------------------------------------------------------------*/

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    A_sub = MatrixSubmatrix(A, is);
    f_sub = VectorSubvector(f, is);
    ttx_sub = VectorSubvector(total_mobility_x, is);
    tty_sub = VectorSubvector(total_mobility_y, is);
    ttz_sub = VectorSubvector(total_mobility_z, is);
    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_sub[phase] = VectorSubvector(tmobility_x[phase], is);
      tmy_sub[phase] = VectorSubvector(tmobility_y[phase], is);
      tmz_sub[phase] = VectorSubvector(tmobility_z[phase], is);
      tc_sub[phase] = VectorSubvector(tcapillary[phase], is);
    }

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ffx = dy * dz * scale;
    ffy = dx * dz * scale;
    ffz = dx * dy * scale;

    nx_v = SubvectorNX(f_sub);
    ny_v = SubvectorNY(f_sub);
    nz_v = SubvectorNZ(f_sub);

    sy_v = nx_v;
    sz_v = ny_v * nx_v;

    cp = SubmatrixStencilData(A_sub, 0);
    wp = SubmatrixStencilData(A_sub, 1);
    ep = SubmatrixStencilData(A_sub, 2);
    sp = SubmatrixStencilData(A_sub, 3);
    np = SubmatrixStencilData(A_sub, 4);
    lp = SubmatrixStencilData(A_sub, 5);
    up = SubmatrixStencilData(A_sub, 6);
    fp = SubvectorData(f_sub);
    ttx_p = SubvectorData(ttx_sub);
    tty_p = SubvectorData(tty_sub);
    ttz_p = SubvectorData(ttz_sub);
    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_pvec[phase] = SubvectorData(tmx_sub[phase]);
      tmy_pvec[phase] = SubvectorData(tmy_sub[phase]);
      tmz_pvec[phase] = SubvectorData(tmz_sub[phase]);
      tc_pvec[phase] = SubvectorData(tc_sub[phase]);
    }

    GrGeomSurfLoop(i, j, k, fdir, gr_domain, 0, ix, iy, iz, nx, ny, nz,
    {
      if (fdir[0])
      {
        ff = ffx;
        d = dx;
        switch (fdir[0])
        {
            case -1:
              sv = -1;
              op = wp;
              break;

            case  1:
              sv = 1;
              op = ep;
              break;
        }
      }
      else if (fdir[1])
      {
        ff = ffy;
        d = dy;
        switch (fdir[1])
        {
            case -1:
              sv = -sy_v;
              op = sp;
              break;

            case  1:
              sv = sy_v;
              op = np;
              break;
        }
      }
      else if (fdir[2])
      {
        ff = ffz;
        d = dz;
        switch (fdir[2])
        {
            case -1:
              sv = -sz_v;
              op = lp;
              break;

            case  1:
              sv = sz_v;
              op = up;
              break;
        }
      }

      iv = SubvectorEltIndex(f_sub, i, j, k);
      im = SubmatrixEltIndex(A_sub, i, j, k);

      /* fix pressure_0 part */
      cp[im] += op[im];
      op[im] = 0.0;

      for (phase = 0; phase < num_phases; phase++)
      {
        if (fdir[0])
        {
          tm_p = tmx_pvec[phase];
          tt_p = ttx_p;
        }
        else if (fdir[1])
        {
          tm_p = tmy_pvec[phase];
          tt_p = tty_p;
        }
        else if (fdir[2])
        {
          tm_p = tmz_pvec[phase];
          tt_p = ttz_p;
        }
        else
        {
          tm_p = 0;
          tt_p = 0;
        }


        tc_p = tc_pvec[phase];

        /* fix capillary pressure part */
        o_temp = -ff * Coeff(tt_p[iv], tt_p[iv + sv],
                             tm_p[iv], tm_p[iv + sv]) / d;
        fp[iv] += o_temp * (tc_p[iv + sv] - tc_p[iv]);

        /* fix gravity part */
        if (fdir[2])
        {
          f_temp =
            ff * Coeff(tt_p[iv], tt_p[iv + sv],
                       tm_p[iv], tm_p[iv + sv]) *
            (phase_density[phase] * gravity);
          fp[iv] -= fdir[2] * f_temp;
        }
      }
    });
  }

  /*-----------------------------------------------------------------------
   * Impose boundary conditions:
   *-----------------------------------------------------------------------*/

  bc_struct = PFModuleInvokeType(BCPressureInvoke, bc_pressure, (problem_data, grid,
                                                                 gr_domain, time));

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    A_sub = MatrixSubmatrix(A, is);
    f_sub = VectorSubvector(f, is);
    ttx_sub = VectorSubvector(total_mobility_x, is);
    tty_sub = VectorSubvector(total_mobility_y, is);
    ttz_sub = VectorSubvector(total_mobility_z, is);
    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_sub[phase] = VectorSubvector(tmobility_x[phase], is);
      tmy_sub[phase] = VectorSubvector(tmobility_y[phase], is);
      tmz_sub[phase] = VectorSubvector(tmobility_z[phase], is);
    }

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    ffx = dy * dz * scale;
    ffy = dx * dz * scale;
    ffz = dx * dy * scale;

    cp = SubmatrixStencilData(A_sub, 0);
    fp = SubvectorData(f_sub);
    ttx_p = SubvectorData(ttx_sub);
    tty_p = SubvectorData(tty_sub);
    ttz_p = SubvectorData(ttz_sub);
    for (phase = 0; phase < num_phases; phase++)
    {
      tmx_pvec[phase] = SubvectorData(tmx_sub[phase]);
      tmy_pvec[phase] = SubvectorData(tmy_sub[phase]);
      tmz_pvec[phase] = SubvectorData(tmz_sub[phase]);
    }

    ForBCStructNumPatches(ipatch, bc_struct)
    {
      bc_patch_values = BCStructPatchValues(bc_struct, ipatch, is);
      ForPatchCellsPerFace(DirichletBC,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                           Locals(int iv, im, phase;
                                  double ff, d, o_temp, f_temp;
                                  double *tm_p; ),
                           CellSetup({
        iv = SubvectorEltIndex(f_sub, i, j, k);
        im = SubmatrixEltIndex(A_sub, i, j, k);
      }),
                           FACE(LeftFace, {
        ff = ffx;
        d = dx;
        tt_p = ttx_p;
      }),
                           FACE(RightFace, {
        ff = ffx;
        d = dx;
        tt_p = ttx_p;
      }),
                           FACE(DownFace, {
        ff = ffy;
        d = dy;
        tt_p = tty_p;
      }),
                           FACE(UpFace, {
        ff = ffy;
        d = dy;
        tt_p = tty_p;
      }),
                           FACE(BackFace, {
        ff = ffz;
        d = dz;
        tt_p = ttz_p;

        for (phase = 0; phase < num_phases; phase++)
        {
          tm_p = tmz_pvec[phase];
          f_temp = ff * tm_p[iv] *
                   (phase_density[phase] * gravity);
          fp[iv] += (-f_temp);
        }
      }),
                           FACE(FrontFace, {
        ff = ffz;
        d = dz;
        tt_p = ttz_p;

        for (phase = 0; phase < num_phases; phase++)
        {
          tm_p = tmz_pvec[phase];
          f_temp = ff * tm_p[iv] *
                   (phase_density[phase] * gravity);
          fp[iv] += f_temp;
        }
      }),
                           CellFinalize({
        o_temp = -ff * 2.0 * tt_p[iv] / d;
        cp[im] -= o_temp;
        fp[iv] -= o_temp * bc_patch_values[ival];
      }),
                           AfterAllCells(DoNothing)
                           );

      ForPatchCellsPerFace(FluxBC,
                           BeforeAllCells(DoNothing),
                           LoopVars(i, j, k, ival, bc_struct, ipatch, is),
                           Locals(int iv, dir; double ff; ),
                           CellSetup({ iv = SubvectorEltIndex(f_sub, i, j, k); }),
                           FACE(LeftFace, { ff = ffx; dir = -1; }),
                           FACE(RightFace, { ff = ffx; dir = 1; }),
                           FACE(DownFace, { ff = ffy; dir = -1; }),
                           FACE(UpFace, { ff = ffy; dir = 1; }),
                           FACE(BackFace, { ff = ffz; dir = -1; }),
                           FACE(FrontFace, { ff = ffz; dir = 1; }),
                           CellFinalize({ fp[iv] -= ff * dir * bc_patch_values[ival]; }),
                           AfterAllCells(DoNothing)
                           );
    }
  }

  FreeBCStruct(bc_struct);

  /*-----------------------------------------------------------------------
   * Impose internal boundary conditions
   *-----------------------------------------------------------------------*/

  PFModuleInvokeType(BCInternalInvoke, ProblemBCInternal(problem),
                     (problem, problem_data, A, f, time));

  /*-----------------------------------------------------------------------
   * Set system values outside of the domain
   *-----------------------------------------------------------------------*/

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    A_sub = MatrixSubmatrix(A, is);
    f_sub = VectorSubvector(f, is);

    /* RDF: assumes resolutions are the same in all 3 directions */
    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    cp = SubmatrixStencilData(A_sub, 0);
    wp = SubmatrixStencilData(A_sub, 1);
    ep = SubmatrixStencilData(A_sub, 2);
    sp = SubmatrixStencilData(A_sub, 3);
    np = SubmatrixStencilData(A_sub, 4);
    lp = SubmatrixStencilData(A_sub, 5);
    up = SubmatrixStencilData(A_sub, 6);
    fp = SubvectorData(f_sub);

    GrGeomOutLoop(i, j, k, gr_domain,
                  r, ix, iy, iz, nx, ny, nz,
    {
      iv = SubvectorEltIndex(f_sub, i, j, k);
      im = SubmatrixEltIndex(A_sub, i, j, k);

      cp[im] = 1.0;
      wp[im] = 0.0;
      ep[im] = 0.0;
      sp[im] = 0.0;
      np[im] = 0.0;
      lp[im] = 0.0;
      up[im] = 0.0;
      fp[iv] = 0.0;
    });
  }

  /*-----------------------------------------------------------------------
   * Update matrix ghost points
   *-----------------------------------------------------------------------*/

  if (MatrixCommPkg(A))
  {
    CommHandle *matrix_handle = InitMatrixUpdate(A);
    FinalizeMatrixUpdate(matrix_handle);
  }

  /*-----------------------------------------------------------------------
   * Clean up and exit
   *-----------------------------------------------------------------------*/

  tfree(phase_density);
  tfree(tmx_sub);
  tfree(tmy_sub);
  tfree(tmz_sub);
  tfree(tc_sub);
  tfree(tmx_pvec);
  tfree(tmy_pvec);
  tfree(tmz_pvec);
  tfree(tc_pvec);

  *ptr_to_A = A;
  *ptr_to_f = f;


  /*-----------------------------------------------------------------------
   * Free temp vectors
   *-----------------------------------------------------------------------*/
  FreeVector(tvector);
  for (phase = 0; phase < num_phases; phase++)
  {
    FreeVector(tcapillary[phase]);
    FreeVector(tmobility_x[phase]);
    FreeVector(tmobility_y[phase]);
    FreeVector(tmobility_z[phase]);
  }
  tfree(tcapillary);
  tfree(tmobility_x);
  tfree(tmobility_y);
  tfree(tmobility_z);
}


/*--------------------------------------------------------------------------
 * DiscretizePressureInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule    *DiscretizePressureInitInstanceXtra(
                                                Problem *problem,
                                                Grid *   grid,
                                                double * temp_data)
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  if (PFModuleInstanceXtra(this_module) == NULL)
    instance_xtra = ctalloc(InstanceXtra, 1);
  else
    instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `problem'
   *-----------------------------------------------------------------------*/

  if (problem != NULL)
  {
    (instance_xtra->problem) = problem;
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `grid'
   *-----------------------------------------------------------------------*/

  if (grid != NULL)
  {
    Stencil   *stencil;


    /* free old data */
    if ((instance_xtra->grid) != NULL)
    {
      FreeVector(instance_xtra->f);
      FreeMatrix(instance_xtra->A);
    }

    /* set new data */
    (instance_xtra->grid) = grid;

    stencil = NewStencil(seven_pt_shape, 7);

    (instance_xtra->A) = NewMatrixType(grid, NULL, stencil, ON, stencil,
                                       matrix_cell_centered);
    (instance_xtra->f) = NewVectorType(grid, 1, 1, vector_cell_centered);
  }

  /*-----------------------------------------------------------------------
   * Initialize data associated with argument `temp_data'
   *-----------------------------------------------------------------------*/

  if (temp_data != NULL)
  {
    (instance_xtra->temp_data) = temp_data;
  }

  if (PFModuleInstanceXtra(this_module) == NULL)
  {
    (instance_xtra->bc_pressure) =
      PFModuleNewInstanceType(BCPressurePackageInitInstanceXtraInvoke, ProblemBCPressure(problem), (problem));
    (instance_xtra->phase_mobility) =
      PFModuleNewInstance(ProblemPhaseMobility(problem), ());
    (instance_xtra->phase_density_module) =
      PFModuleNewInstance(ProblemPhaseDensity(problem), ());
    (instance_xtra->capillary_pressure) =
      PFModuleNewInstance(ProblemCapillaryPressure(problem), ());
    (instance_xtra->phase_source) =
      PFModuleNewInstance(ProblemPhaseSource(problem), ());
  }
  else
  {
    PFModuleReNewInstanceType(BCPressurePackageInitInstanceXtraInvoke, (instance_xtra->bc_pressure), (problem));
    PFModuleReNewInstance((instance_xtra->phase_mobility), ());
    PFModuleReNewInstance((instance_xtra->phase_density_module), ());
    PFModuleReNewInstance((instance_xtra->capillary_pressure), ());
    PFModuleReNewInstance((instance_xtra->phase_source), ());
  }

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * DiscretizePressureFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  DiscretizePressureFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    PFModuleFreeInstance(instance_xtra->bc_pressure);
    PFModuleFreeInstance(instance_xtra->phase_mobility);
    PFModuleFreeInstance(instance_xtra->phase_density_module);
    PFModuleFreeInstance(instance_xtra->capillary_pressure);
    PFModuleFreeInstance(instance_xtra->phase_source);

    FreeVector(instance_xtra->f);
    FreeMatrix(instance_xtra->A);

    tfree(instance_xtra);
  }
}


/*--------------------------------------------------------------------------
 * DiscretizePressureNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule   *DiscretizePressureNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * DiscretizePressureFreePublicXtra
 *--------------------------------------------------------------------------*/

void  DiscretizePressureFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);


  if (public_xtra)
  {
    tfree(public_xtra);
  }
}


/*--------------------------------------------------------------------------
 * DiscretizePressureSizeOfTempData
 *--------------------------------------------------------------------------*/

int  DiscretizePressureSizeOfTempData()
{
  int sz = 0;

  return sz;
}
