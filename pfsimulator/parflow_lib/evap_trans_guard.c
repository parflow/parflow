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

/** @file
 * @brief Key parsing and per-step CSV reporting for the prescribed
 *        evap_trans sink guard (Solver.EvapTransGuard).  The guard math
 *        itself is inline in evap_trans_guard.h so the nonlinear residual
 *        and the analytic Jacobian share one definition.
 */

#include "parflow.h"
#include "evap_trans_guard.h"

#include <string.h>

/*--------------------------------------------------------------------------
 * EvapTransGuardReadKeys
 *--------------------------------------------------------------------------*/

void EvapTransGuardReadKeys(EvapTransGuard *guard, int verbose)
{
  NameArray switch_na = NA_NewNameArray("False True");
  char      *switch_name;
  char key[IDB_MAX_KEY_LEN];

  sprintf(key, "Solver.EvapTransGuard");
  switch_name = GetStringDefault(key, "False");
  guard->active = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  guard->margin = GetDoubleDefault("Solver.EvapTransGuard.Margin", 0.02);
  guard->ramp_width = GetDoubleDefault("Solver.EvapTransGuard.RampWidth", 0.05);

  sprintf(key, "Solver.EvapTransGuard.PrintLog");
  switch_name = GetStringDefault(key, "False");
  guard->print_log = NA_NameToIndexExitOnError(switch_na, switch_name, key);

  NA_FreeNameArray(switch_na);

  guard->log_started = 0;
  guard->withheld_cum = 0.0;

  if (guard->active)
  {
    if (guard->margin < 0.0)
      InputError("Error: negative value for key <%s>%s\n",
                 "Solver.EvapTransGuard.Margin", "");
    if (guard->ramp_width <= 0.0)
      InputError("Error: key <%s> must be positive%s\n",
                 "Solver.EvapTransGuard.RampWidth", "");

    /* The guard limits PRESCRIBED fluxes only.  Under CLM the evap_trans
     * vector is CLM's own moisture-limited flux and must pass through
     * unchanged, so the guard deactivates itself. */
    if (strcmp(GetStringDefault("Solver.LSM", "none"), "CLM") == 0)
    {
      guard->active = 0;
      if (verbose && !amps_Rank(amps_CommWorld))
        amps_Printf("EvapTransGuard: Solver.LSM = CLM, guard deactivated "
                    "(CLM fluxes are already moisture-limited)\n");
    }
    else if (verbose && !amps_Rank(amps_CommWorld))
    {
      amps_Printf("EvapTransGuard active: Margin = %f, RampWidth = %f, "
                  "PrintLog = %d\n",
                  guard->margin, guard->ramp_width, guard->print_log);
    }
  }
}

/*--------------------------------------------------------------------------
 * EvapTransGuardLogStep
 *--------------------------------------------------------------------------*/

void EvapTransGuardLogStep(EvapTransGuard *guard,
                           Vector *        evap_trans,
                           Vector *        saturation,
                           Vector *        sres,
                           ProblemData *   problem_data,
                           double          time,
                           double          dt,
                           int             step)
{
  Grid          *grid = VectorGrid(evap_trans);
  GrGeomSolid   *gr_domain = ProblemDataGrDomain(problem_data);
  Vector        *z_mult = ProblemDataZmult(problem_data);

  Subgrid       *subgrid;
  Subvector     *et_sub, *s_sub, *srs_sub, *zm_sub;
  double        *et, *sp, *srs, *zm;

  int is, i, j, k, r, ix, iy, iz, nx, ny, nz;
  double dx, dy, dz;

  int n_neg = 0, n_limited = 0, n_shut = 0;
  double prescribed = 0.0, applied = 0.0;
  double min_beta = 1.0, min_sat = 1.0;

  double margin = guard->margin;
  double ramp_width = guard->ramp_width;

  ForSubgridI(is, GridSubgrids(grid))
  {
    subgrid = GridSubgrid(grid, is);

    et_sub = VectorSubvector(evap_trans, is);
    s_sub = VectorSubvector(saturation, is);
    srs_sub = VectorSubvector(sres, is);
    zm_sub = VectorSubvector(z_mult, is);

    r = SubgridRX(subgrid);

    ix = SubgridIX(subgrid);
    iy = SubgridIY(subgrid);
    iz = SubgridIZ(subgrid);

    nx = SubgridNX(subgrid);
    ny = SubgridNY(subgrid);
    nz = SubgridNZ(subgrid);

    dx = SubgridDX(subgrid);
    dy = SubgridDY(subgrid);
    dz = SubgridDZ(subgrid);

    double vol = dx * dy * dz;

    et = SubvectorData(et_sub);
    sp = SubvectorData(s_sub);
    srs = SubvectorData(srs_sub);
    zm = SubvectorData(zm_sub);

    GrGeomInLoop(i, j, k, gr_domain, r, ix, iy, iz, nx, ny, nz,
    {
      int ip = SubvectorEltIndex(et_sub, i, j, k);

      if (et[ip] < 0.0)
      {
        double beta = EvapTransGuardBeta(sp[ip], srs[ip], margin, ramp_width);
        double cell_sink = -et[ip] * vol * zm[ip] * dt;

        n_neg += 1;
        prescribed += cell_sink;
        applied += cell_sink * beta;
        if (beta < 1.0)
          n_limited += 1;
        if (beta <= 0.0)
          n_shut += 1;
        if (beta < min_beta)
          min_beta = beta;
        if (sp[ip] < min_sat)
          min_sat = sp[ip];
      }
    });
  }

  {
    amps_Invoice invoice =
      amps_NewInvoice("%i%i%i%d%d", &n_neg, &n_limited, &n_shut,
                      &prescribed, &applied);
    amps_AllReduce(amps_CommWorld, invoice, amps_Add);
    amps_FreeInvoice(invoice);

    invoice = amps_NewInvoice("%d%d", &min_beta, &min_sat);
    amps_AllReduce(amps_CommWorld, invoice, amps_Min);
    amps_FreeInvoice(invoice);
  }

  double withheld = prescribed - applied;
  guard->withheld_cum += withheld;

  if (!amps_Rank(amps_CommWorld))
  {
    FILE *log_file;
    char file_name[2048];

    sprintf(file_name, "%s.etguard.csv", GlobalsOutFileName);

    if (!guard->log_started)
    {
      log_file = fopen(file_name, "w");
      if (log_file)
        fprintf(log_file,
                "step,time,dt,n_neg,n_limited,n_shut,prescribed_sink,"
                "applied_sink,withheld_step,withheld_cum,min_beta,"
                "min_sat_guarded\n");
      guard->log_started = 1;
    }
    else
    {
      log_file = fopen(file_name, "a");
    }

    if (log_file)
    {
      fprintf(log_file,
              "%d,%.6e,%.6e,%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.6e,%.6e\n",
              step, time, dt, n_neg, n_limited, n_shut, prescribed, applied,
              withheld, guard->withheld_cum, min_beta, min_sat);
      fclose(log_file);
    }
  }
}
