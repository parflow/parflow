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
* Header info for evaluation points,
* used with root-finding/maximization procedures.
*
*****************************************************************************/

#ifndef _PROBLEM_EVAL_HEADER
#define _PROBLEM_EVAL_HEADER

typedef struct {
  int num_f_points, num_h_points;
  double     *f_points, *f_values;
  double     *h_points, *h_values;
} EvalStruct;

/*--------------------------------------------------------------------------
 * Accessor macros: FHEvalStruct
 *--------------------------------------------------------------------------*/

#define EvalNumFPoints(eval_struct)      ((eval_struct)->num_f_points)
#define EvalFPoints(eval_struct)         ((eval_struct)->f_points)
#define EvalFPoint(eval_struct, i)       ((eval_struct)->f_points[i])
#define EvalFValues(eval_struct)         ((eval_struct)->f_values)
#define EvalFValue(eval_struct, i)       ((eval_struct)->f_values[i])

#define EvalNumHPoints(eval_struct)      ((eval_struct)->num_h_points)
#define EvalHPoints(eval_struct)         ((eval_struct)->h_points)
#define EvalHPoint(eval_struct, i)       ((eval_struct)->h_points[i])
#define EvalHValues(eval_struct)         ((eval_struct)->h_values)
#define EvalHValue(eval_struct, i)       ((eval_struct)->h_values[i])

/*--------------------------------------------------------------------------
 * The actual functions in use.
 *--------------------------------------------------------------------------*/

#define Fprime_OF_S(s, a, b)  (2.0 * (a) * (b) *                                             \
                               (-pow((s), 2.0) + (s))                                        \
                               /                                                             \
                               pow(((a) * pow((s), 2.0) + (b) * pow((1.0 - (s)), 2.0)), 2.0) \
                               )

#define Hprime_OF_S(s, a, b)  (2.0 * (a) * (b) *                                           \
                               (((a) + (b)) * pow((s), 5.0)                                \
                                - ((a) + 4.0 * (b)) * pow((s), 4.0)                        \
                                + 6 * (b) * pow((s), 3.0)                                  \
                                - 4 * (b) * pow((s), 2.0)                                  \
                                + (b) * (s))                                               \
                               /                                                           \
                               pow(((a) * pow((s), 2) + (b) * pow((1.0 - (s)), 2.0)), 2.0) \
                               )

/*--------------------------------------------------------------------------
 * Misc define's: FHEvalStruct
 *--------------------------------------------------------------------------*/

#define XSTART 1.0
#define MAXITERATIONS 500
#define EPSILON1 .0000000000001
#define EPSILON2 .00000000001

#endif
