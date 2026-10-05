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
*  This module computes the contributions for the spatial discretization of the
*  kinematic wave approximation for the overland flow boundary condition:KE,KW,KN,KS.
*
*  It also computes the derivatives of these terms for inclusion in the Jacobian.
*
* @LEC, @RMM
*****************************************************************************/

#include "parflow.h"

#if !defined(PARFLOW_HAVE_CUDA) && !defined(PARFLOW_HAVE_KOKKOS)
#include "llnlmath.h"
#endif
/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void PublicXtra;

typedef void InstanceXtra;

/*---------------------------------------------------------------------
 * Define macros for function evaluation
 *---------------------------------------------------------------------*/
#define RPMean(a, b, c, d)   UpstreamMean(a, b, c, d)

/*--------------------------------------------------------------------------
 * Diffusion correction helpers
 *--------------------------------------------------------------------------*/

/* Ponded depth of the surface cell at (ii, jj), or zero where there is none */
#define DCPondedDepth(pdat, ktop, ii, jj)                                      \
        (((ktop) >= 0) ?                                                       \
         pfmax((pdat)[SubvectorEltIndex(p_sub, (ii), (jj), (ktop))], 0.0) : 0.0)

/* Centered difference where both neighbors exist, one-sided where only one
 * does, zero where neither does */
#define DCCenteredGrad(hc, hm, has_m, hp, has_p, d)                            \
        (((has_m) && (has_p)) ? ((hp) - (hm)) / (2.0 * (d)) :                  \
         ((has_p) ? ((hp) - (hc)) / (d) :                                      \
          ((has_m) ? ((hc) - (hm)) / (d) : 0.0)))

/* Slope magnitude in the diffusion coefficient at the east face (D_denom_x)
 * and the north face (D_denom_y) of cell (i, j).  The gradient normal to a
 * face is the two-point difference across it.  The gradient along a face is
 * the average of the centered differences in the two cells that share it,
 * which keeps the scheme symmetric.  That average reads the diagonal
 * neighbors, so pressure and the top index need corner ghost cells.  pdat is
 * the pressure array to read: the current pressure, or the old-time pressure
 * for the lagged velocity correction. */
#define DCFaceDenominators(pdat, Pdown, Pup_x_dc, Pup_y_dc, D_denom_x, D_denom_y)          \
        {                                                                                  \
          if (diff_denom == 0)                                                             \
          {                                                                                \
            D_denom_x = Sf_mag;                                                            \
            D_denom_y = Sf_mag;                                                            \
          }                                                                                \
          else                                                                             \
          {                                                                                \
            int dc_kne = (int)top_dat[itop + 1 + sy_v];                                    \
            int dc_kse = (int)top_dat[itop + 1 - sy_v];                                    \
            int dc_knw = (int)top_dat[itop - 1 + sy_v];                                    \
            double dc_he = (k1x >= 0) ? (Pup_x_dc) : 0.0;                                  \
            double dc_hn = (k1y >= 0) ? (Pup_y_dc) : 0.0;                                  \
            double dc_hw = DCPondedDepth(pdat, k0x, i - 1, j);                             \
            double dc_hs = DCPondedDepth(pdat, k0y, i, j - 1);                             \
            double dc_hne = DCPondedDepth(pdat, dc_kne, i + 1, j + 1);                     \
            double dc_hse = DCPondedDepth(pdat, dc_kse, i + 1, j - 1);                     \
            double dc_hnw = DCPondedDepth(pdat, dc_knw, i - 1, j + 1);                     \
            double dc_gx_c = DCCenteredGrad((Pdown), dc_hw, k0x >= 0,                      \
                                            dc_he, k1x >= 0, dx);                          \
            double dc_gy_c = DCCenteredGrad((Pdown), dc_hs, k0y >= 0,                      \
                                            dc_hn, k1y >= 0, dy);                          \
            double dc_gy_e = (k1x >= 0) ?                                                  \
                             0.5 * (dc_gy_c                                                \
                                    + DCCenteredGrad(dc_he, dc_hse, dc_kse >= 0,           \
                                                     dc_hne, dc_kne >= 0, dy))             \
                             : dc_gy_c;                                                    \
            double dc_gx_n = (k1y >= 0) ?                                                  \
                             0.5 * (dc_gx_c                                                \
                                    + DCCenteredGrad(dc_hn, dc_hnw, dc_knw >= 0,           \
                                                     dc_hne, dc_kne >= 0, dx))             \
                             : dc_gx_c;                                                    \
            double dc_gx_e = diff_alpha * ((Pup_x_dc) - (Pdown)) / dx;                     \
            double dc_gy_n = diff_alpha * ((Pup_y_dc) - (Pdown)) / dy;                     \
            dc_gy_e *= diff_alpha;                                                         \
            dc_gx_n *= diff_alpha;                                                         \
            if (diff_denom == 1)                                                           \
            {                                                                              \
              D_denom_x = RPowerR((sx_dat[io] + dc_gx_e) * (sx_dat[io] + dc_gx_e)          \
                                  + (sy_dat[io] + dc_gy_e) * (sy_dat[io] + dc_gy_e), 0.5); \
              D_denom_y = RPowerR((sx_dat[io] + dc_gx_n) * (sx_dat[io] + dc_gx_n)          \
                                  + (sy_dat[io] + dc_gy_n) * (sy_dat[io] + dc_gy_n), 0.5); \
            }                                                                              \
            else                                                                           \
            {                                                                              \
              double dc_s02 = sx_dat[io] * sx_dat[io] + sy_dat[io] * sy_dat[io];           \
              D_denom_x = RPowerR(dc_s02 + dc_gx_e * dc_gx_e + dc_gy_e * dc_gy_e, 0.5);    \
              D_denom_y = RPowerR(dc_s02 + dc_gx_n * dc_gx_n + dc_gy_n * dc_gy_n, 0.5);    \
            }                                                                              \
            if (D_denom_x < ov_epsilon)                                                    \
            D_denom_x = ov_epsilon;                                                        \
            if (D_denom_y < ov_epsilon)                                                    \
            D_denom_y = ov_epsilon;                                                        \
          }                                                                                \
        }

/* Slope magnitude at an internal patch edge, where only the gradient normal to
 * the edge is known.  S_n and S_t are the bed slope normal to and along the
 * edge, grad_n the water-surface gradient across it. */
__host__ __device__ static inline double
DCEdgeDenominator(int diff_denom, double diff_alpha,
                  double S_n, double S_t, double grad_n,
                  double S0_mag, double ov_epsilon)
{
  double denom;

  if (diff_denom == 0)
    return S0_mag;

  if (diff_denom == 1)
    denom = RPowerR((S_n + diff_alpha * grad_n) * (S_n + diff_alpha * grad_n)
                    + S_t * S_t, 0.5);
  else
    denom = RPowerR(S_n * S_n + S_t * S_t
                    + diff_alpha * grad_n * diff_alpha * grad_n, 0.5);

  return (denom < ov_epsilon) ? ov_epsilon : denom;
}


/*--------------------------------------------------------------------------
 * OverlandKinDiffusionOptions
 *
 * Reads the Solver.OverlandKinematic.Diffusion keys once, checks them, and
 * returns them as indices.  The flux across a face is
 *
 *   q = -(h^{5/3} / n) (S_0 / |A|^{1/2} + alpha dh / |B|^{1/2}).
 *
 * slope_magnitude:  0 Kinematic (no water-surface term), 1 BedSlope,
 *                   2 FrictionSlope, 3 Pythagorean.  The magnitude used
 *                   for A and B.
 * bed_term:         what A is.  0 BedSlope, 1 Lagged (the magnitude from the
 *                   pressure at the previous time step), 2 Implicit (from
 *                   the current pressure).
 * surface_term:     the time level of B.  0 Lagged, 1 Implicit.
 * jacobian:         0 Picard, 1 FullNewton, 2 FullNewtonDdx.
 *
 * With BedSlope both A and B are |S_0| and the two term keys have no effect.
 *--------------------------------------------------------------------------*/

void OverlandKinDiffusionOptions(int *slope_magnitude, int *bed_term,
                                 int *surface_term, int *jacobian,
                                 double *alpha)
{
  static int options_read = 0;
  static int s_slope_magnitude, s_bed_term, s_surface_term, s_jacobian;
  static double s_alpha;

  if (!options_read)
  {
    const char *old_keys[] = { "Type", "Alpha", "Jacobian", "Denominator",
                               "VelocityCorrection", "DenominatorTimeLevel" };
    char key[IDB_MAX_KEY_LEN];
    NameArray na;
    int idx;

    /* The keys were renamed; a deck with the old names must not run as the
     * kinematic wave without saying so. */
    for (idx = 0; idx < 6; idx++)
    {
      sprintf(key, "Solver.OverlandKinematic.DiffusionCorrection.%s", old_keys[idx]);
      IDB_Entry lookup_entry;

      lookup_entry.key = key;
      if (HBT_lookup(amps_ThreadLocal(input_database), &lookup_entry) != NULL)
      {
        InputError("Error: key <%s> is no longer used.  Use the %s keys:\n"
                   "       SlopeMagnitude, BedTermMagnitude, SurfaceTermMagnitude, Jacobian, Alpha\n",
                   key, "Solver.OverlandKinematic.Diffusion");
      }
    }

    sprintf(key, "Solver.OverlandKinematic.Diffusion.SlopeMagnitude");
    na = NA_NewNameArray("Kinematic BedSlope FrictionSlope Pythagorean");
    s_slope_magnitude = NA_NameToIndexExitOnError(na, GetStringDefault(key, "Kinematic"), key);
    NA_FreeNameArray(na);

    sprintf(key, "Solver.OverlandKinematic.Diffusion.BedTermMagnitude");
    na = NA_NewNameArray("BedSlope Lagged Implicit");
    s_bed_term = NA_NameToIndexExitOnError(na, GetStringDefault(key, "Lagged"), key);
    NA_FreeNameArray(na);

    sprintf(key, "Solver.OverlandKinematic.Diffusion.SurfaceTermMagnitude");
    na = NA_NewNameArray("Lagged Implicit");
    s_surface_term = NA_NameToIndexExitOnError(na, GetStringDefault(key, "Lagged"), key);
    NA_FreeNameArray(na);

    sprintf(key, "Solver.OverlandKinematic.Diffusion.Jacobian");
    na = NA_NewNameArray("Picard FullNewton FullNewtonDdx");
    s_jacobian = NA_NameToIndexExitOnError(na, GetStringDefault(key, "FullNewton"), key);
    NA_FreeNameArray(na);

    s_alpha = GetDoubleDefault("Solver.OverlandKinematic.Diffusion.Alpha", 1.0);

    /* An implicit bed term with a lagged surface term is not implemented */
    if (s_slope_magnitude > 1 && s_bed_term == 2 && s_surface_term == 0)
    {
      InputError("Error: %s = Implicit needs %s = Implicit\n",
                 "Solver.OverlandKinematic.Diffusion.BedTermMagnitude",
                 "Solver.OverlandKinematic.Diffusion.SurfaceTermMagnitude");
    }

    options_read = 1;
  }

  *slope_magnitude = s_slope_magnitude;
  *bed_term = s_bed_term;
  *surface_term = s_surface_term;
  *jacobian = s_jacobian;
  *alpha = s_alpha;
}


/*-------------------------------------------------------------------------
 * OverlandFlowEval
 *-------------------------------------------------------------------------*/

void    OverlandFlowEvalKin(
                            Grid *       grid,  /* data struct for computational grid */
                            int          sg,  /* current subgrid */
                            BCStruct *   bc_struct,  /* data struct of boundary patch values */
                            int          ipatch,  /* current boundary patch */
                            ProblemData *problem_data,  /* Geometry data for problem */
                            Vector *     pressure,  /* Vector of phase pressures at each block */
                            Vector *     old_pressure,  /* Vector of phase pressures at previous time */
                            double *     ke_v,  /* return array corresponding to the east face KE  */
                            double *     kw_v,  /* return array corresponding to the west face KW */
                            double *     kn_v,  /* return array corresponding to the north face KN */
                            double *     ks_v,  /* return array corresponding to the south face KS */
                            double *     ke_vns,  /* return array corresponding to the nonsymetric east face KE derivative  */
                            double *     kw_vns,  /* return array corresponding to the nonsymetricwest face KW derivative */
                            double *     kn_vns,  /* return array corresponding to the nonsymetricnorth face KN derivative */
                            double *     ks_vns,  /* return array corresponding to the nonsymetricsouth face KS derivative*/
                            double *     qx_v,  /* return array corresponding to the flux in x-dir */
                            double *     qy_v,  /* return array corresponding to the flux in y-dir */
                            int          fcn)  /* Flag determining what to calculate
                                                * fcn = CALCFCN => calculate the function value
                                                * fcn = CALCDER => calculate the function
                                                *                  derivative */
{
  Vector      *slope_x = ProblemDataTSlopeX(problem_data);
  Vector      *slope_y = ProblemDataTSlopeY(problem_data);
  Vector      *mannings = ProblemDataMannings(problem_data);
  Vector      *top = ProblemDataIndexOfDomainTop(problem_data);
  Vector      *patch = ProblemDataPatchIndexOfDomainTop(problem_data);

  Subvector     *sx_sub, *sy_sub, *mann_sub, *top_sub, *patch_sub, *p_sub;

  double        *sx_dat, *sy_dat, *mann_dat, *top_dat, *patch_dat, *pp;
  double        *opp = NULL;

  double ov_epsilon;

  int diffusion_correction;
  int diff_jacobian;
  int diff_denom;        /* 0=BedSlope, 1=FrictionSlope, 2=Pythagorean */
  int vel_corr;          /* bed term: 0=bed slope, 1=old-time magnitude, 2=same magnitude as the surface term */
  int denom_old;         /* 1 if the slope magnitude in D uses the old-time pressure */
  double diff_alpha;
  double dx, dy;

  int i, j, k, ival = 0, sy_v;

  PF_UNUSED(ival);

  p_sub = VectorSubvector(pressure, sg);

  sx_sub = VectorSubvector(slope_x, sg);
  sy_sub = VectorSubvector(slope_y, sg);
  mann_sub = VectorSubvector(mannings, sg);
  top_sub = VectorSubvector(top, sg);
  patch_sub = VectorSubvector(patch, sg);

  pp = SubvectorData(p_sub);

  sx_dat = SubvectorData(sx_sub);
  sy_dat = SubvectorData(sy_sub);
  mann_dat = SubvectorData(mann_sub);
  top_dat = SubvectorData(top_sub);
  patch_dat = SubvectorData(patch_sub);

  sy_v = SubvectorNX(top_sub);

  //ov_epsilon= 1.0e-5;
  ov_epsilon = GetDoubleDefault("Solver.OverlandKinematic.Epsilon", 1.0e-5);

  /* Diffusion keys, mapped to the flags used below.  With the bed term on
   * the bed slope the flux is the kinematic flux plus a diffusive term.
   * Putting the slope magnitude under the bed term as well adds
   * S_0 h^{5/3}/n (1/|S_0|^{1/2} - 1/|S_denom|^{1/2}) to the flux, which is
   * what makes the flux vanish under hydrostatic conditions.  A lagged
   * magnitude comes from the old-time pressure, so it is fixed within a
   * time step.  With
   * FrictionSlope and both terms lagged this is the diffusive wave with a
   * lagged friction-slope magnitude: the BedSlope flux times
   * (|S_0| / |S_f,old|)^{1/2}. */
  {
    int slope_magnitude, bed_term, surface_term;

    OverlandKinDiffusionOptions(&slope_magnitude, &bed_term, &surface_term,
                                &diff_jacobian, &diff_alpha);

    diffusion_correction = (slope_magnitude > 0) ? 1 : 0;
    diff_denom = (slope_magnitude > 0) ? slope_magnitude - 1 : 0;
    vel_corr = 0;
    denom_old = 0;
    if (diff_denom > 0)
    {
      denom_old = (surface_term == 0) ? 1 : 0;
      /* With a lagged surface term, vel_corr 2 shares its old-time magnitude */
      if (bed_term == 1)
        vel_corr = denom_old ? 2 : 1;
      else if (bed_term == 2)
        vel_corr = 2;
    }
    if (vel_corr == 1 || denom_old)
      opp = SubvectorData(VectorSubvector(old_pressure, sg));
  }

  {
    Subgrid *subgrid = GridSubgrid(grid, sg);
    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
  }

  if (fcn == CALCFCN)
  {
    ForPatchCellsPerFaceWithGhost(BC_ALL,
                                  BeforeAllCells(DoNothing),
                                  LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                                  Locals(int io, itop, ip, ipp1, ipm1, ipmsy, ippsy, ipat;
                                         int k1, k0x, k0y, k1x, k1y;
                                         int p1, p0x, p0y;
                                         double Sf_x, Sf_y, Sf_mag;
                                         double Press_x, Press_y;
                                         double PP_ipp1, PP_ippsy, PP_ip; ),
                                  CellSetup(DoNothing),
                                  FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                                  FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                                  FACE(BackFace, DoNothing),
                                  FACE(FrontFace,
    {
      io = SubvectorEltIndex(sx_sub, i, j, 0);
      itop = SubvectorEltIndex(top_sub, i, j, 0);
      ipat = SubvectorEltIndex(patch_sub, i, j, 0);

      k1 = (int)top_dat[itop];
      k0x = (int)top_dat[itop - 1];
      k0y = (int)top_dat[itop - sy_v];
      k1x = (int)top_dat[itop + 1];
      k1y = (int)top_dat[itop + sy_v];
      //RMM added patches to check for internal bc edges
      p1 = (int)patch_dat[ipat];
      p0x = (int)patch_dat[ipat - 1];
      p0y = (int)patch_dat[ipat - sy_v];

      if (k1 >= 0)
      {
        ip = SubvectorEltIndex(p_sub, i, j, k1);
        Sf_x = sx_dat[io];
        Sf_y = sy_dat[io];
        ipp1 = (int)SubvectorEltIndex(p_sub, i + 1, j, k1x);
        ippsy = (int)SubvectorEltIndex(p_sub, i, j + 1, k1y);

        Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
        if (Sf_mag < ov_epsilon)
          Sf_mag = ov_epsilon;
        PP_ipp1 = 0.0;
        PP_ippsy = 0.0;
        PP_ip = pp[ip];
        if (k1x >= 0)
          PP_ipp1 = pp[ipp1];
        if (k1y >= 0)
          PP_ippsy = pp[ippsy];

        /* Upwind selection: use friction slope Sf* when diffusion correction is on,
         * bed slope S_0 otherwise. Sf* = S_0 + alpha*grad(psi).
         * At domain boundaries (k1x/k1y < 0), revert to bed slope — no
         * physical neighbor for the gradient. */
        if (diffusion_correction)
        {
          double Pdown = pfmax(PP_ip, 0.0);
          double Pup_x = pfmax(PP_ipp1, 0.0);
          double Pup_y = pfmax(PP_ippsy, 0.0);
          double Sf_star_x = (k1x >= 0) ? Sf_x + diff_alpha * (Pup_x - Pdown) / dx : Sf_x;
          double Sf_star_y = (k1y >= 0) ? Sf_y + diff_alpha * (Pup_y - Pdown) / dy : Sf_y;

          Press_x = RPMean(-Sf_star_x, 0.0, Pdown, Pup_x);
          Press_y = RPMean(-Sf_star_y, 0.0, Pdown, Pup_y);
        }
        else
        {
          Press_x = RPMean(-Sf_x, 0.0,
                           pfmax((PP_ip), 0.0),
                           pfmax((PP_ipp1), 0.0));
          Press_y = RPMean(-Sf_y, 0.0,
                           pfmax((PP_ip), 0.0),
                           pfmax((PP_ippsy), 0.0));
        }

        /* Slope magnitude under the kinematic term at the east and north
         * faces: |S_0|, or with the velocity correction the slope magnitude of
         * the diffusion coefficient.  Faces at a domain boundary keep |S_0|. */
        double K_denom_x = Sf_mag;
        double K_denom_y = Sf_mag;
        double D_denom_x = Sf_mag;
        double D_denom_y = Sf_mag;
        if (diffusion_correction)
        {
          double Pdown = pfmax(PP_ip, 0.0);
          double Pup_x = (k1x >= 0) ? pfmax(PP_ipp1, 0.0) : Pdown;
          double Pup_y = (k1y >= 0) ? pfmax(PP_ippsy, 0.0) : Pdown;

          /* Slope magnitude in D at the east and north faces, from the
           * current pressure or the old-time pressure */
          double D_old_x = Sf_mag;
          double D_old_y = Sf_mag;
          if (vel_corr == 1 || denom_old)
          {
            double Pdown_o = pfmax(opp[ip], 0.0);
            double Pup_x_o = (k1x >= 0) ? pfmax(opp[ipp1], 0.0) : Pdown_o;
            double Pup_y_o = (k1y >= 0) ? pfmax(opp[ippsy], 0.0) : Pdown_o;
            DCFaceDenominators(opp, Pdown_o, Pup_x_o, Pup_y_o, D_old_x, D_old_y);
          }
          if (denom_old)
          {
            D_denom_x = D_old_x;
            D_denom_y = D_old_y;
          }
          else
          {
            DCFaceDenominators(pp, Pdown, Pup_x, Pup_y, D_denom_x, D_denom_y);
          }

          if (vel_corr == 2)
          {
            if (k1x >= 0)
              K_denom_x = D_denom_x;
            if (k1y >= 0)
              K_denom_y = D_denom_y;
          }
          else if (vel_corr == 1)
          {
            if (k1x >= 0)
              K_denom_x = D_old_x;
            if (k1y >= 0)
              K_denom_y = D_old_y;
          }
        }

        /* Kinematic flux: S_0 in numerator, Sf*-upwinded depth */
        qx_v[io] = -(Sf_x / (RPowerR(fabs(K_denom_x), 0.5) * mann_dat[io]))
                   * RPowerR(Press_x, (5.0 / 3.0));
        qy_v[io] = -(Sf_y / (RPowerR(fabs(K_denom_y), 0.5)
                             * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));

        /* Diffusion correction: -D * grad(psi), D = alpha * Press^{5/3}/(n * |S_denom|^{1/2})
         * Skip correction at domain boundaries (k1x/k1y < 0) — no physical neighbor. */
        if (diffusion_correction)
        {
          double Pdown = pfmax(PP_ip, 0.0);
          double Pup_x = (k1x >= 0) ? pfmax(PP_ipp1, 0.0) : Pdown;
          double Pup_y = (k1y >= 0) ? pfmax(PP_ippsy, 0.0) : Pdown;

          if (ipp1 >= 0 && k1x >= 0)
          {
            double D_x = diff_alpha * RPowerR(Press_x, 5.0 / 3.0)
                         / (RPowerR(fabs(D_denom_x), 0.5) * mann_dat[io]);
            qx_v[io] += -D_x * (Pup_x - Pdown) / dx;
          }
          if (ippsy >= 0 && k1y >= 0)
          {
            double D_y = diff_alpha * RPowerR(Press_y, 5.0 / 3.0)
                         / (RPowerR(fabs(D_denom_y), 0.5) * mann_dat[io]);
            qy_v[io] += -D_y * (Pup_y - Pdown) / dy;
          }
        }
      }
      // fix for internal patch edges in x direction
      if (p1 >= 0 && p0x >= 0 && p1 != p0x)
      {
        if (k1 >= 0)
        {
          ip = SubvectorEltIndex(p_sub, i, j, k1);
          Sf_x = sx_dat[io - 1];
          Sf_y = sy_dat[io - 1];
          ipm1 = (int)SubvectorEltIndex(p_sub, i - 1, j, k0x);

          Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          double Pdown_w = pfmax(pp[ipm1], 0.0);
          double Pup_w = pfmax(pp[ip], 0.0);

          if (diffusion_correction)
          {
            double Sf_star_w = Sf_x + diff_alpha * (Pup_w - Pdown_w) / dx;
            Press_x = RPMean(-Sf_star_w, 0.0, Pdown_w, Pup_w);
          }
          else
          {
            Press_x = RPMean(-Sf_x, 0.0, Pdown_w, Pup_w);
          }

          double K_denom_w = Sf_mag;
          if (vel_corr)
          {
            double grad_w = (vel_corr == 1 || denom_old) ?
                            (pfmax(opp[ip], 0.0) - pfmax(opp[ipm1], 0.0)) / dx :
                            (Pup_w - Pdown_w) / dx;
            K_denom_w = DCEdgeDenominator(diff_denom, diff_alpha, Sf_x, Sf_y, grad_w,
                                          Sf_mag, ov_epsilon);
          }

          qx_v[io - 1] = -(Sf_x / (RPowerR(fabs(K_denom_w), 0.5) * mann_dat[io - 1]))
                         * RPowerR(Press_x, (5.0 / 3.0));

          if (diffusion_correction)
          {
            double D_denom_mag;
            if (diff_denom == 0)
            {
              D_denom_mag = Sf_mag;
            }
            else if (diff_denom == 1)
            {
              double Sf_star_w = Sf_x + diff_alpha * (Pup_w - Pdown_w) / dx;
              double Sf_star_wy = Sf_y + 0.0;  /* no y-gradient info at patch edge */
              D_denom_mag = RPowerR(Sf_star_w * Sf_star_w + Sf_star_wy * Sf_star_wy, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            else
            {
              double dhdx_w = diff_alpha * (Pup_w - Pdown_w) / dx;
              D_denom_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y
                                    + dhdx_w * dhdx_w, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            if (denom_old)
              D_denom_mag = DCEdgeDenominator(diff_denom, diff_alpha, Sf_x, Sf_y,
                                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipm1], 0.0)) / dx,
                                              Sf_mag, ov_epsilon);
            double D_coeff = diff_alpha
                             / (RPowerR(fabs(D_denom_mag), 0.5) * mann_dat[io - 1]);
            double D_x = D_coeff * RPowerR(Press_x, 5.0 / 3.0);
            qx_v[io - 1] += -D_x * (Pup_w - Pdown_w) / dx;
          }
        }
      }

      // fix for internal patch edges in y direction
      if (p1 >= 0 && p0y >= 0 && p1 != p0y)
      {
        if (k1 >= 0)
        {
          ip = SubvectorEltIndex(p_sub, i, j, k1);
          Sf_x = sx_dat[io - sy_v];
          Sf_y = sy_dat[io - sy_v];
          ipmsy = (int)SubvectorEltIndex(p_sub, i, j - 1, k0y);

          Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          double Pdown_s = pfmax(pp[ipmsy], 0.0);
          double Pup_s = pfmax(pp[ip], 0.0);

          if (diffusion_correction)
          {
            double Sf_star_s = Sf_y + diff_alpha * (Pup_s - Pdown_s) / dy;
            Press_y = RPMean(-Sf_star_s, 0.0, Pdown_s, Pup_s);
          }
          else
          {
            Press_y = RPMean(-Sf_y, 0.0, Pdown_s, Pup_s);
          }

          double K_denom_s = Sf_mag;
          if (vel_corr)
          {
            double grad_s = (vel_corr == 1 || denom_old) ?
                            (pfmax(opp[ip], 0.0) - pfmax(opp[ipmsy], 0.0)) / dy :
                            (Pup_s - Pdown_s) / dy;
            K_denom_s = DCEdgeDenominator(diff_denom, diff_alpha, Sf_y, Sf_x, grad_s,
                                          Sf_mag, ov_epsilon);
          }

          qy_v[io - sy_v] = -(Sf_y / (RPowerR(fabs(K_denom_s), 0.5)
                                      * mann_dat[io - sy_v])) * RPowerR(Press_y, (5.0 / 3.0));

          if (diffusion_correction)
          {
            double D_denom_mag;
            if (diff_denom == 0)
            {
              D_denom_mag = Sf_mag;
            }
            else if (diff_denom == 1)
            {
              double Sf_star_sx = Sf_x + 0.0;  /* no x-gradient info at patch edge */
              double Sf_star_s = Sf_y + diff_alpha * (Pup_s - Pdown_s) / dy;
              D_denom_mag = RPowerR(Sf_star_sx * Sf_star_sx + Sf_star_s * Sf_star_s, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            else
            {
              double dhdy_s = diff_alpha * (Pup_s - Pdown_s) / dy;
              D_denom_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y
                                    + dhdy_s * dhdy_s, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            if (denom_old)
              D_denom_mag = DCEdgeDenominator(diff_denom, diff_alpha, Sf_y, Sf_x,
                                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipmsy], 0.0)) / dy,
                                              Sf_mag, ov_epsilon);
            double D_coeff = diff_alpha
                             / (RPowerR(fabs(D_denom_mag), 0.5) * mann_dat[io - sy_v]);
            double D_y = D_coeff * RPowerR(Press_y, 5.0 / 3.0);
            qy_v[io - sy_v] += -D_y * (Pup_s - Pdown_s) / dy;
          }
        }
      }

      //fix for lower x boundary
      if (k0x < 0.0)
      {
        if (k1 >= 0.0)
        {
          Sf_x = sx_dat[io];
          Sf_y = sy_dat[io];

          double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          if (Sf_x > 0.0)
          {
            ip = SubvectorEltIndex(p_sub, i, j, k1);
            Press_x = pfmax((pp[ip]), 0.0);
            qx_v[io - 1] = -(Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (5.0 / 3.0));
            /* No diffusion correction at lower boundaries — no physical neighbor */
          }
        }
      }

      //fix for lower y boundary
      if (k0y < 0.0)
      {
        if (k1 >= 0.0)
        {
          Sf_x = sx_dat[io];
          Sf_y = sy_dat[io];

          double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          if (Sf_y > 0.0)
          {
            ip = SubvectorEltIndex(p_sub, i, j, k1);
            Press_y = pfmax((pp[ip]), 0.0);
            qy_v[io - sy_v] = -(Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (5.0 / 3.0));
            /* No diffusion correction at lower boundaries — no physical neighbor */
          }
        }
      }
    }),
                                  CellFinalize(DoNothing),
                                  AfterAllCells(DoNothing)
                                  );

    ForPatchCellsPerFace(BC_ALL,
                         BeforeAllCells(DoNothing),
                         LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                         Locals(int io; ),
                         CellSetup(DoNothing),
                         FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                         FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                         FACE(BackFace, DoNothing),
                         FACE(FrontFace,
    {
      io = SubvectorEltIndex(sx_sub, i, j, 0);
      ke_v[io] = qx_v[io];
      kw_v[io] = qx_v[io - 1];
      kn_v[io] = qy_v[io];
      ks_v[io] = qy_v[io - sy_v];
    }),
                         CellFinalize(DoNothing),
                         AfterAllCells(DoNothing)
                         );
  }
  else          //fcn = CALCDER calculates the derivs
  {
    ForPatchCellsPerFaceWithGhost(BC_ALL,
                                  BeforeAllCells(DoNothing),
                                  LoopVars(i, j, k, ival, bc_struct, ipatch, sg),
                                  Locals(int io, itop, ipat, ip, ipp1, ippsy, ipm1, ipmsy;
                                         int k1, k0x, k0y, k1x, k1y;
                                         int p1, p0x, p0y;
                                         double Sf_x, Sf_y, Sf_mag;
                                         double Press_x, Press_y, qx_temp, qy_temp;
                                         double PP_ipp1, PP_ippsy; ),
                                  CellSetup(DoNothing),
                                  FACE(LeftFace, DoNothing), FACE(RightFace, DoNothing),
                                  FACE(DownFace, DoNothing), FACE(UpFace, DoNothing),
                                  FACE(BackFace, DoNothing),
                                  FACE(FrontFace,
    {
      io = SubvectorEltIndex(sx_sub, i, j, 0);
      itop = SubvectorEltIndex(top_sub, i, j, 0);
      ipat = SubvectorEltIndex(patch_sub, i, j, 0);

      k1 = (int)top_dat[itop];
      k0x = (int)top_dat[itop - 1];
      k0y = (int)top_dat[itop - sy_v];
      k1x = (int)top_dat[itop + 1];
      k1y = (int)top_dat[itop + sy_v];
      //RMM added patches to check for internal bc edges
      p1 = (int)patch_dat[ipat];
      p0x = (int)patch_dat[ipat - 1];
      p0y = (int)patch_dat[ipat - sy_v];

      if (k1 >= 0)
      {
        ip = SubvectorEltIndex(p_sub, i, j, k1);
        ipp1 = (int)SubvectorEltIndex(p_sub, i + 1, j, k1x);
        ippsy = (int)SubvectorEltIndex(p_sub, i, j + 1, k1y);

        Sf_x = sx_dat[io];
        Sf_y = sy_dat[io];

        Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
        if (Sf_mag < ov_epsilon)
          Sf_mag = ov_epsilon;

        /* Guard the neighbor reads on the top value: the computed index can
        * land in an allocated ghost layer when there is no surface cell. */
        PP_ipp1 = 0.0;
        PP_ippsy = 0.0;
        if (k1x >= 0)
          PP_ipp1 = pp[ipp1];
        if (k1y >= 0)
          PP_ippsy = pp[ippsy];

        /* Derivative of q = -(Sf_star / (|S_0|^{1/2}*n)) * Press^{5/3}
         * h-derivative uses Sf_star (combined kin+diff), pfmax routes to upwind cell.
         * Sf_star-derivative gives +/-D/dx Picard terms with ponding guards. */
        /* At domain boundaries (k1x/k1y < 0), revert to bed slope — no
         * physical neighbor for the gradient. */
        if (diffusion_correction)
        {
          double Pdown = pfmax(pp[ip], 0.0);
          double Pup_x = pfmax(PP_ipp1, 0.0);
          double Pup_y = pfmax(PP_ippsy, 0.0);
          double Sf_star_x = (k1x >= 0) ? Sf_x + diff_alpha * (Pup_x - Pdown) / dx : Sf_x;
          double Sf_star_y = (k1y >= 0) ? Sf_y + diff_alpha * (Pup_y - Pdown) / dy : Sf_y;
          /* For D_denom and flux, zero gradient at domain boundaries */
          double Pup_x_dc = (k1x >= 0) ? Pup_x : Pdown;
          double Pup_y_dc = (k1y >= 0) ? Pup_y : Pdown;

          Press_x = RPMean(-Sf_star_x, 0.0, Pdown, Pup_x);
          Press_y = RPMean(-Sf_star_y, 0.0, Pdown, Pup_y);

          /* Slope magnitude in D at the east face and at the north face, from
           * the current pressure or the old-time pressure */
          double D_denom_x;
          double D_denom_y;
          double D_old_x = Sf_mag;
          double D_old_y = Sf_mag;
          if (vel_corr == 1 || denom_old)
          {
            double Pdown_o = pfmax(opp[ip], 0.0);
            double Pup_x_o = (k1x >= 0) ? pfmax(opp[ipp1], 0.0) : Pdown_o;
            double Pup_y_o = (k1y >= 0) ? pfmax(opp[ippsy], 0.0) : Pdown_o;
            DCFaceDenominators(opp, Pdown_o, Pup_x_o, Pup_y_o, D_old_x, D_old_y);
          }
          if (denom_old)
          {
            D_denom_x = D_old_x;
            D_denom_y = D_old_y;
          }
          else
          {
            DCFaceDenominators(pp, Pdown, Pup_x_dc, Pup_y_dc, D_denom_x, D_denom_y);
          }

          /* h-derivative: Picard uses S_0 (freezes D), FullNewton uses Sf*
           * (adds dD/dh * grad(h) cross-term to upwind diagonal) */
          double slope_jac_x = diff_jacobian ? Sf_star_x : Sf_x;
          double slope_jac_y = diff_jacobian ? Sf_star_y : Sf_y;
          if (diff_denom == 0)
          {
            qx_temp = -(5.0 / 3.0) * (slope_jac_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]))
                      * RPowerR(Press_x, (2.0 / 3.0));
            qy_temp = -(5.0 / 3.0) * (slope_jac_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]))
                      * RPowerR(Press_y, (2.0 / 3.0));
          }
          else
          {
            /* The flux is -(S_0 / |K_denom|^{1/2} + alpha grad(h) / |D_denom|^{1/2})
             * h^{5/3} / n, so each term carries its own slope magnitude in the
             * h-derivative.  K_denom is |S_0|, or with the velocity correction
             * the slope magnitude of D, held fixed in the derivative.  For
             * Lagged that is exact.  For Implicit it leaves out the derivative
             * of the denominator, as FullNewton does for D. */
            double K_denom_x = Sf_mag;
            double K_denom_y = Sf_mag;
            if (vel_corr == 2)
            {
              if (k1x >= 0)
                K_denom_x = D_denom_x;
              if (k1y >= 0)
                K_denom_y = D_denom_y;
            }
            else if (vel_corr == 1)
            {
              if (k1x >= 0)
                K_denom_x = D_old_x;
              if (k1y >= 0)
                K_denom_y = D_old_y;
            }
            double grad_jac_x = diff_jacobian ? (Sf_star_x - Sf_x) : 0.0;
            double grad_jac_y = diff_jacobian ? (Sf_star_y - Sf_y) : 0.0;
            qx_temp = -(5.0 / 3.0) * ((Sf_x / RPowerR(fabs(K_denom_x), 0.5)
                                       + grad_jac_x / RPowerR(fabs(D_denom_x), 0.5)) / mann_dat[io])
                      * RPowerR(Press_x, (2.0 / 3.0));
            qy_temp = -(5.0 / 3.0) * ((Sf_y / RPowerR(fabs(K_denom_y), 0.5)
                                       + grad_jac_y / RPowerR(fabs(D_denom_y), 0.5)) / mann_dat[io])
                      * RPowerR(Press_y, (2.0 / 3.0));
          }

          if (diff_denom == 0)
          {
            ke_v[io] = pfmax(qx_temp, 0);
            kw_v[io + 1] = -pfmax(-qx_temp, 0);
            kn_v[io] = pfmax(qy_temp, 0);
            ks_v[io + sy_v] = -pfmax(-qy_temp, 0);
          }
          else
          {
            /* The depth is upwinded by the sign of Sf*, so the h-derivative
             * belongs to the cell that sign selects.  With two different
             * slope magnitudes the flux can have the other sign, so do not
             * route by the sign of the derivative. */
            ke_v[io] = (Sf_star_x < 0.0) ? qx_temp : 0.0;
            kw_v[io + 1] = (Sf_star_x < 0.0) ? 0.0 : qx_temp;
            kn_v[io] = (Sf_star_y < 0.0) ? qy_temp : 0.0;
            ks_v[io + sy_v] = (Sf_star_y < 0.0) ? 0.0 : qy_temp;
          }

          /* Sf*-derivative: ±D/dx with ponding guards.  D uses the slope
           * magnitude at the east face for D_x and at the north face for D_y. */

          double D_x = diff_alpha * RPowerR(Press_x, 5.0 / 3.0)
                       / (RPowerR(fabs(D_denom_x), 0.5) * mann_dat[io]);
          double D_y = diff_alpha * RPowerR(Press_y, 5.0 / 3.0)
                       / (RPowerR(fabs(D_denom_y), 0.5) * mann_dat[io]);

          if (k1x >= 0)
          {
            if (Pdown > 0.0)
              ke_v[io] += D_x / dx;
            if (Pup_x > 0.0)
              kw_v[io + 1] += -D_x / dx;
          }
          if (k1y >= 0)
          {
            if (Pdown > 0.0)
              kn_v[io] += D_y / dy;
            if (Pup_y > 0.0)
              ks_v[io + sy_v] += -D_y / dy;
          }

          /* Level 2: derivative of D through its slope magnitude.  With
           * D ~ |Seff|^{-1/2}, d(D hx)/dhx = D (1 - f), where f = hx^2/(2|Seff|^2)
           * for Pythagorean and f = hx (S_0 + hx)/(2|Seff|^2) for FrictionSlope,
           * taking the normal component only.  f is 1/2 on flat ground, which
           * is the nonlinear diffusion h_x/|h_x|^{1/2}.  It is kept within
           * [-1/2, 1/2].  Only active for FrictionSlope/Pythagorean. */
          if (diff_jacobian >= 2 && diff_denom > 0 && !denom_old)
          {
            double dhdx_val = diff_alpha * (Pup_x_dc - Pdown) / dx;
            double dhdy_val = diff_alpha * (Pup_y_dc - Pdown) / dy;
            double Seff2_x = D_denom_x * D_denom_x;
            double Seff2_y = D_denom_y * D_denom_y;
            double numx = (diff_denom == 1) ? dhdx_val * (sx_dat[io] + dhdx_val) : dhdx_val * dhdx_val;
            double numy = (diff_denom == 1) ? dhdy_val * (sy_dat[io] + dhdy_val) : dhdy_val * dhdy_val;
            double fx = (Seff2_x > 0) ? pfmax(-0.5, pfmin(0.5, numx / (2.0 * Seff2_x))) : 0.0;
            double fy = (Seff2_y > 0) ? pfmax(-0.5, pfmin(0.5, numy / (2.0 * Seff2_y))) : 0.0;

            if (k1x >= 0)
            {
              if (Pdown > 0.0)
                ke_v[io] += -fx * D_x / dx;
              if (Pup_x > 0.0)
                kw_v[io + 1] += fx * D_x / dx;
            }
            if (k1y >= 0)
            {
              if (Pdown > 0.0)
                kn_v[io] += -fy * D_y / dy;
              if (Pup_y > 0.0)
                ks_v[io + sy_v] += fy * D_y / dy;
            }
          }
        }
        else
        {
          Press_x = RPMean(-Sf_x, 0.0,
                           pfmax((pp[ip]), 0.0),
                           pfmax((PP_ipp1), 0.0));
          Press_y = RPMean(-Sf_y, 0.0,
                           pfmax((pp[ip]), 0.0),
                           pfmax((PP_ippsy), 0.0));

          qx_temp = -(5.0 / 3.0) * (Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]))
                    * RPowerR(Press_x, (2.0 / 3.0));
          qy_temp = -(5.0 / 3.0) * (Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io]))
                    * RPowerR(Press_y, (2.0 / 3.0));

          ke_v[io] = pfmax(qx_temp, 0);
          kw_v[io + 1] = -pfmax(-qx_temp, 0);
          kn_v[io] = pfmax(qy_temp, 0);
          ks_v[io + sy_v] = -pfmax(-qy_temp, 0);
        }
      }

      // fix for internal patch edges in x direction
      if (p1 >= 0 && p0x >= 0 && p1 != p0x)
      {
        if (k1 >= 0)
        {
          ip = SubvectorEltIndex(p_sub, i, j, k1);
          Sf_x = sx_dat[io - 1];
          Sf_y = sy_dat[io - 1];
          ipm1 = (int)SubvectorEltIndex(p_sub, i - 1, j, k0x);

          Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          double Pdown_w = pfmax(pp[ipm1], 0.0);
          double Pup_w = pfmax(pp[ip], 0.0);

          if (diffusion_correction)
          {
            double Sf_star_w = Sf_x + diff_alpha * (Pup_w - Pdown_w) / dx;
            Press_x = RPMean(-Sf_star_w, 0.0, Pdown_w, Pup_w);

            double slope_jac_w = diff_jacobian ? Sf_star_w : Sf_x;
            if (diff_denom == 0)
            {
              qx_temp = -(5.0 / 3.0) * (slope_jac_w / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io - 1]))
                        * RPowerR(Press_x, (2.0 / 3.0));
            }
            else
            {
              double grad_now_w = (Pup_w - Pdown_w) / dx;
              double grad_w = (vel_corr == 1 || denom_old) ?
                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipm1], 0.0)) / dx : grad_now_w;
              double K_denom_w = (vel_corr == 0) ? Sf_mag :
                                 DCEdgeDenominator(diff_denom, diff_alpha, Sf_x, Sf_y, grad_w,
                                                   Sf_mag, ov_epsilon);
              double D_now_w = DCEdgeDenominator(diff_denom, diff_alpha, Sf_x, Sf_y,
                                                 denom_old ? grad_w : grad_now_w,
                                                 Sf_mag, ov_epsilon);
              double grad_jac_w = diff_jacobian ? (Sf_star_w - Sf_x) : 0.0;
              qx_temp = -(5.0 / 3.0) * ((Sf_x / RPowerR(fabs(K_denom_w), 0.5)
                                         + grad_jac_w / RPowerR(fabs(D_now_w), 0.5)) / mann_dat[io - 1])
                        * RPowerR(Press_x, (2.0 / 3.0));
            }
            if (diff_denom == 0)
            {
              kw_v[io] = -pfmax(-qx_temp, 0);
              ke_v[io - 1] = pfmax(qx_temp, 0);
            }
            else
            {
              kw_v[io] = (Sf_star_w < 0.0) ? 0.0 : qx_temp;
              ke_v[io - 1] = (Sf_star_w < 0.0) ? qx_temp : 0.0;
            }

            double D_denom_mag;
            if (diff_denom == 0)
            {
              D_denom_mag = Sf_mag;
            }
            else if (diff_denom == 1)
            {
              double Sf_star_wy = Sf_y + 0.0;
              D_denom_mag = RPowerR(Sf_star_w * Sf_star_w + Sf_star_wy * Sf_star_wy, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            else
            {
              double dhdx_w = diff_alpha * (Pup_w - Pdown_w) / dx;
              D_denom_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y
                                    + dhdx_w * dhdx_w, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            if (denom_old)
              D_denom_mag = DCEdgeDenominator(diff_denom, diff_alpha, Sf_x, Sf_y,
                                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipm1], 0.0)) / dx,
                                              Sf_mag, ov_epsilon);
            double D_coeff = diff_alpha
                             / (RPowerR(fabs(D_denom_mag), 0.5) * mann_dat[io - 1]);
            double D_x = D_coeff * RPowerR(Press_x, 5.0 / 3.0);

            if (Pdown_w > 0.0)
              ke_v[io - 1] += D_x / dx;
            if (Pup_w > 0.0)
              kw_v[io] += -D_x / dx;

            if (diff_jacobian >= 2 && diff_denom > 0 && !denom_old)
            {
              double dhdx_w = diff_alpha * (Pup_w - Pdown_w) / dx;
              double Seff2 = D_denom_mag * D_denom_mag;
              double numx = (diff_denom == 1) ? dhdx_w * (Sf_x + dhdx_w) : dhdx_w * dhdx_w;
              double fx = (Seff2 > 0) ? pfmax(-0.5, pfmin(0.5, numx / (2.0 * Seff2))) : 0.0;
              if (Pdown_w > 0.0)
                ke_v[io - 1] += -fx * D_x / dx;
              if (Pup_w > 0.0)
                kw_v[io] += fx * D_x / dx;
            }
          }
          else
          {
            Press_x = RPMean(-Sf_x, 0.0, Pdown_w, Pup_w);

            qx_temp = -(5.0 / 3.0) * (Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io - 1]))
                      * RPowerR(Press_x, (2.0 / 3.0));
            kw_v[io] = -pfmax(-qx_temp, 0);
            ke_v[io - 1] = pfmax(qx_temp, 0);
          }
        }
      }

      // fix for internal patch edges in y direction
      if (p1 >= 0 && p0y >= 0 && p1 != p0y)
      {
        if (k1 >= 0)
        {
          ip = SubvectorEltIndex(p_sub, i, j, k1);
          Sf_x = sx_dat[io - sy_v];
          Sf_y = sy_dat[io - sy_v];
          ipmsy = (int)SubvectorEltIndex(p_sub, i, j - 1, k0y);

          Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          double Pdown_s = pfmax(pp[ipmsy], 0.0);
          double Pup_s = pfmax(pp[ip], 0.0);

          if (diffusion_correction)
          {
            double Sf_star_s = Sf_y + diff_alpha * (Pup_s - Pdown_s) / dy;
            Press_y = RPMean(-Sf_star_s, 0.0, Pdown_s, Pup_s);

            double slope_jac_s = diff_jacobian ? Sf_star_s : Sf_y;
            if (diff_denom == 0)
            {
              qy_temp = -(5.0 / 3.0) * (slope_jac_s / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io - sy_v]))
                        * RPowerR(Press_y, (2.0 / 3.0));
            }
            else
            {
              double grad_now_s = (Pup_s - Pdown_s) / dy;
              double grad_s = (vel_corr == 1 || denom_old) ?
                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipmsy], 0.0)) / dy : grad_now_s;
              double K_denom_s = (vel_corr == 0) ? Sf_mag :
                                 DCEdgeDenominator(diff_denom, diff_alpha, Sf_y, Sf_x, grad_s,
                                                   Sf_mag, ov_epsilon);
              double D_now_s = DCEdgeDenominator(diff_denom, diff_alpha, Sf_y, Sf_x,
                                                 denom_old ? grad_s : grad_now_s,
                                                 Sf_mag, ov_epsilon);
              double grad_jac_s = diff_jacobian ? (Sf_star_s - Sf_y) : 0.0;
              qy_temp = -(5.0 / 3.0) * ((Sf_y / RPowerR(fabs(K_denom_s), 0.5)
                                         + grad_jac_s / RPowerR(fabs(D_now_s), 0.5)) / mann_dat[io - sy_v])
                        * RPowerR(Press_y, (2.0 / 3.0));
            }
            if (diff_denom == 0)
            {
              ks_v[io] = -pfmax(-qy_temp, 0);
              kn_v[io - sy_v] = pfmax(qy_temp, 0);
            }
            else
            {
              ks_v[io] = (Sf_star_s < 0.0) ? 0.0 : qy_temp;
              kn_v[io - sy_v] = (Sf_star_s < 0.0) ? qy_temp : 0.0;
            }

            double D_denom_mag;
            if (diff_denom == 0)
            {
              D_denom_mag = Sf_mag;
            }
            else if (diff_denom == 1)
            {
              double Sf_star_sx = Sf_x + 0.0;
              D_denom_mag = RPowerR(Sf_star_sx * Sf_star_sx + Sf_star_s * Sf_star_s, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            else
            {
              double dhdy_s = diff_alpha * (Pup_s - Pdown_s) / dy;
              D_denom_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y
                                    + dhdy_s * dhdy_s, 0.5);
              if (D_denom_mag < ov_epsilon)
                D_denom_mag = ov_epsilon;
            }
            if (denom_old)
              D_denom_mag = DCEdgeDenominator(diff_denom, diff_alpha, Sf_y, Sf_x,
                                              (pfmax(opp[ip], 0.0) - pfmax(opp[ipmsy], 0.0)) / dy,
                                              Sf_mag, ov_epsilon);
            double D_coeff = diff_alpha
                             / (RPowerR(fabs(D_denom_mag), 0.5) * mann_dat[io - sy_v]);
            double D_y = D_coeff * RPowerR(Press_y, 5.0 / 3.0);

            if (Pdown_s > 0.0)
              kn_v[io - sy_v] += D_y / dy;
            if (Pup_s > 0.0)
              ks_v[io] += -D_y / dy;

            if (diff_jacobian >= 2 && diff_denom > 0 && !denom_old)
            {
              double dhdy_s = diff_alpha * (Pup_s - Pdown_s) / dy;
              double Seff2 = D_denom_mag * D_denom_mag;
              double numy = (diff_denom == 1) ? dhdy_s * (Sf_y + dhdy_s) : dhdy_s * dhdy_s;
              double fy = (Seff2 > 0) ? pfmax(-0.5, pfmin(0.5, numy / (2.0 * Seff2))) : 0.0;
              if (Pdown_s > 0.0)
                kn_v[io - sy_v] += -fy * D_y / dy;
              if (Pup_s > 0.0)
                ks_v[io] += fy * D_y / dy;
            }
          }
          else
          {
            Press_y = RPMean(-Sf_y, 0.0, Pdown_s, Pup_s);

            qy_temp = -(5.0 / 3.0) * (Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io - sy_v]))
                      * RPowerR(Press_y, (2.0 / 3.0));
            ks_v[io] = -pfmax(-qy_temp, 0);
            kn_v[io - sy_v] = pfmax(qy_temp, 0);
          }
        }
      }

      //fix for lower x boundary
      if (k0x < 0.0)
      {
        if (k1 >= 0.0)
        {
          Sf_x = sx_dat[io];
          Sf_y = sy_dat[io];

          double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          if (Sf_x > 0.0)
          {
            ip = SubvectorEltIndex(p_sub, i, j, k1);
            Press_x = pfmax((pp[ip]), 0.0);
            qx_temp = -(5.0 / 3.0) * (Sf_x / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_x, (2.0 / 3.0));

            kw_v[io] = -pfmax(-qx_temp, 0);
            ke_v[io - 1] = pfmax(qx_temp, 0);
            /* No diffusion correction at lower boundaries — no physical neighbor */
          }
        }
      }

      //fix for lower y boundary
      if (k0y < 0.0)
      {
        if (k1 >= 0.0)
        {
          Sf_x = sx_dat[io];
          Sf_y = sy_dat[io];

          double Sf_mag = RPowerR(Sf_x * Sf_x + Sf_y * Sf_y, 0.5);                                //+ov_epsilon;
          if (Sf_mag < ov_epsilon)
            Sf_mag = ov_epsilon;

          if (Sf_y > 0.0)
          {
            ip = SubvectorEltIndex(p_sub, i, j, k1);
            Press_y = pfmax((pp[ip]), 0.0);
            qy_temp = -(5.0 / 3.0) * (Sf_y / (RPowerR(fabs(Sf_mag), 0.5) * mann_dat[io])) * RPowerR(Press_y, (2.0 / 3.0));

            ks_v[io] = -pfmax(-qy_temp, 0);
            kn_v[io - sy_v] = pfmax(qy_temp, 0);
            /* No diffusion correction at lower boundaries — no physical neighbor */
          }
        }
      }
    }),
                                  CellFinalize(DoNothing),
                                  AfterAllCells(DoNothing)
                                  );
  }   // else calcder
}     // function


//*/
/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinInitInstanceXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalKinInitInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra;

  instance_xtra = NULL;

  PFModuleInstanceXtra(this_module) = instance_xtra;
  return this_module;
}


/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinFreeInstanceXtra
 *--------------------------------------------------------------------------*/

void  OverlandFlowEvalKinFreeInstanceXtra()
{
  PFModule      *this_module = ThisPFModule;
  InstanceXtra  *instance_xtra = (InstanceXtra*)PFModuleInstanceXtra(this_module);

  if (instance_xtra)
  {
    tfree(instance_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinNewPublicXtra
 *--------------------------------------------------------------------------*/

PFModule  *OverlandFlowEvalKinNewPublicXtra()
{
  PFModule      *this_module = ThisPFModule;
  PublicXtra    *public_xtra;

  public_xtra = NULL;

  PFModulePublicXtra(this_module) = public_xtra;
  return this_module;
}

/*-------------------------------------------------------------------------
 * OverlandFlowEvalKinFreePublicXtra
 *-------------------------------------------------------------------------*/

void  OverlandFlowEvalKinFreePublicXtra()
{
  PFModule    *this_module = ThisPFModule;
  PublicXtra  *public_xtra = (PublicXtra*)PFModulePublicXtra(this_module);

  if (public_xtra)
  {
    tfree(public_xtra);
  }
}

/*--------------------------------------------------------------------------
 * OverlandFlowEvalKinSizeOfTempData
 *--------------------------------------------------------------------------*/

int  OverlandFlowEvalKinSizeOfTempData()
{
  return 0;
}
