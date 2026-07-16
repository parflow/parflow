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
 * @brief Moisture-limited guard on prescribed evap_trans sink cells
 *        (Solver.EvapTransGuard).
 *
 * Prescribed-flux (P-ET EvapTransFile) spin-up runs apply net-negative
 * (ET-demand) cells with no moisture limitation, which can drive cells far
 * below residual saturation into unphysical suction.  When the guard is
 * enabled, cells with negative evap_trans have the sink scaled by a C1
 * smoothstep factor beta(S) that ramps from 1 to 0 as the cell saturation
 * approaches its van Genuchten residual:
 *
 *   beta = 1                          S >= S_start
 *   beta = t*t*(3 - 2*t)              S_stop < S < S_start,
 *                                     t = (S - S_stop) / RampWidth
 *   beta = 0                          S <= S_stop
 *
 *   S_stop  = S_res(cell) + Margin
 *   S_start = S_stop + RampWidth
 *
 * The same inline beta / dbeta_dS below are used by both the nonlinear
 * residual (nl_function_eval.c) and the analytic Jacobian
 * (richards_jacobian_eval.c), so the two stay consistent by construction.
 * Positive (source) cells are never modified, and the guard is inert unless
 * Solver.EvapTransGuard is True.
 */

#ifndef _EVAP_TRANS_GUARD_H
#define _EVAP_TRANS_GUARD_H

#include "parflow.h"

typedef struct {
  int active;              /* master switch (forced off when Solver.LSM = CLM) */
  double margin;           /* S_stop = S_res + margin */
  double ramp_width;       /* S_start = S_stop + ramp_width */
  int print_log;           /* per-accepted-step CSV <runname>.etguard.csv */

  /* CSV logging state (used only by the solver-level reporter) */
  int log_started;         /* header written */
  double withheld_cum;     /* cumulative withheld sink volume (all ranks agree) */
} EvapTransGuard;

/* Parse the Solver.EvapTransGuard keys into *guard.  verbose != 0 prints the
 * one rank-0 activation line (pass verbose only from one module so the line
 * appears once). */
void EvapTransGuardReadKeys(EvapTransGuard *guard, int verbose);

/* Compute per-accepted-step guard statistics (reduced across ranks) and
 * append one CSV row.  Call only when guard->active && guard->print_log. */
void EvapTransGuardLogStep(EvapTransGuard *guard,
                           Vector *        evap_trans,
                           Vector *        saturation,
                           Vector *        sres,
                           ProblemData *   problem_data,
                           double          time,
                           double          dt,
                           int             step);

/* The guard math is expressed as macros (matching the RPMean/PMean in-loop
 * helper convention) so it is usable inside GrGeomInLoop bodies under every
 * backend, including device lambdas in accelerated builds. */

/* Ramp coordinate t = (S - S_stop) / RampWidth, unclamped. */
#define EvapTransGuardT(s, s_res, margin, ramp_width) \
        (((s) - ((s_res) + (margin))) / (ramp_width))

/* Sink scale factor beta(S) in [0, 1]; C1 cubic smoothstep. */
#define EvapTransGuardBeta(s, s_res, margin, ramp_width)                     \
        ((EvapTransGuardT(s, s_res, margin, ramp_width) <= 0.0) ? 0.0 :      \
         (EvapTransGuardT(s, s_res, margin, ramp_width) >= 1.0) ? 1.0 :      \
         (EvapTransGuardT(s, s_res, margin, ramp_width)                      \
          * EvapTransGuardT(s, s_res, margin, ramp_width)                    \
          * (3.0 - 2.0 * EvapTransGuardT(s, s_res, margin, ramp_width))))

/* dbeta/dS, analytic derivative of the smoothstep (zero outside the ramp). */
#define EvapTransGuardBetaDer(s, s_res, margin, ramp_width)                  \
        ((EvapTransGuardT(s, s_res, margin, ramp_width) <= 0.0) ? 0.0 :      \
         (EvapTransGuardT(s, s_res, margin, ramp_width) >= 1.0) ? 0.0 :      \
         (6.0 * EvapTransGuardT(s, s_res, margin, ramp_width)                \
          * (1.0 - EvapTransGuardT(s, s_res, margin, ramp_width))            \
          / (ramp_width)))

#endif /* _EVAP_TRANS_GUARD_H */
