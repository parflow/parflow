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
* Structures for the solver.
*
*****************************************************************************/

#ifndef _SOLVER_HEADER
#define _SOLVER_HEADER


/*--------------------------------------------------------------------------
 * Some global information
 *--------------------------------------------------------------------------*/

#define ArithmeticMean(a, b)  (0.5 * ((a) + (b)))
#define GeometricMean(a, b)   (sqrt((a) * (b)))
#define HarmonicMean(a, b)    (((a) + (b)) ? (2.0 * (a) * (b)) / ((a) + (b)) : 0)
//#define HarmonicMeanDZ(a, b, c, d)    ( ((a*c) + (b*d)) ? ( (c+d)/ ((c/a) + (d/b)) ) : 0 )
#define HarmonicMeanDZ(a, b, c, d) (((c * b) + (a * d)) ?  (((c + d) * a * b) / ((b * c) + (a * d))) : 0)
#define UpstreamMean(a, b, c, d) (((a - b) >= 0) ? c : d)

/* Smoothed upstream weight for TFG lateral fluxes (cubic smoothstep over a
 * band of half-width eps).  w -> 1 selects the (diff >= 0) upwind branch,
 * w -> 0 the other.  With eps <= 0 it reduces to UpstreamMean branch selection
 * exactly, including the diff == 0 tie (returns 1.0).  Pure arithmetic/ternary,
 * so it is device-callable by textual expansion like the other Mean macros. */
#define UpwindWeightSmooth(diff, eps)                            \
        (((diff) >= (eps)) ? 1.0 :                               \
         ((diff) <= -(eps)) ? 0.0 :                              \
         (0.5 * ((diff) / (eps) + 1.0)) *                        \
         (0.5 * ((diff) / (eps) + 1.0)) *                        \
         (3.0 - 2.0 * (0.5 * ((diff) / (eps) + 1.0))))

/* d(UpwindWeightSmooth)/d(diff): zero outside the band and for eps <= 0, so the
 * smoothing contributes nothing to the Jacobian when eps = 0. */
#define UpwindWeightSmoothDer(diff, eps)                                 \
        (((eps) <= 0.0 || (diff) >= (eps) || (diff) <= -(eps)) ? 0.0 :   \
         3.0 * (0.5 * ((diff) / (eps) + 1.0)) *                          \
         (1.0 - (0.5 * ((diff) / (eps) + 1.0))) / (eps))

#define CellFaceConductivity  HarmonicMean


#endif





