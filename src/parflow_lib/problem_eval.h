/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
/******************************************************************************
 * Header info for evaluation points,
 * used with root-finding/maximization procedures.
 *
 *****************************************************************************/

#ifndef _PROBLEM_EVAL_HEADER
#define _PROBLEM_EVAL_HEADER

typedef struct
{
   int     num_f_points, num_h_points;
   double     *f_points,    *f_values;
   double     *h_points,    *h_values;
} EvalStruct;

/*--------------------------------------------------------------------------
 * Accessor macros: FHEvalStruct
 *--------------------------------------------------------------------------*/

#define EvalNumFPoints(eval_struct)      ((eval_struct) -> num_f_points)
#define EvalFPoints(eval_struct)         ((eval_struct) -> f_points)
#define EvalFPoint(eval_struct, i)       ((eval_struct) -> f_points[i])
#define EvalFValues(eval_struct)         ((eval_struct) -> f_values)
#define EvalFValue(eval_struct, i)       ((eval_struct) -> f_values[i])

#define EvalNumHPoints(eval_struct)      ((eval_struct) -> num_h_points)
#define EvalHPoints(eval_struct)         ((eval_struct) -> h_points)
#define EvalHPoint(eval_struct, i)       ((eval_struct) -> h_points[i])
#define EvalHValues(eval_struct)         ((eval_struct) -> h_values)
#define EvalHValue(eval_struct, i)       ((eval_struct) -> h_values[i])

/*--------------------------------------------------------------------------
 * The actual functions in use.
 *--------------------------------------------------------------------------*/

#define Fprime_OF_S(s,a,b)  ( 2.0 * (a) * (b) * \
                              (-pow((s),2.0) + (s)) \
                                   / \
           pow(((a) * pow((s),2.0) + (b) * pow((1.0-(s)),2.0)),2.0) \
                            )

#define Hprime_OF_S(s,a,b)  ( 2.0 * (a) * (b) *  \
                             (    ((a)+(b))*pow((s),5.0) \
                            - ((a)+4.0*(b))*pow((s),4.0) \
                            +       6*(b)*pow((s),3.0) \
                            -       4*(b)*pow((s),2.0) \
                            +         (b)*(s)) \
                                      / \
           pow(((a) * pow((s),2) + (b) * pow((1.0-(s)),2.0)),2.0) \
                            )

/*--------------------------------------------------------------------------
 * Misc define's: FHEvalStruct
 *--------------------------------------------------------------------------*/

#define XSTART 1.0
#define MAXITERATIONS 500
#define EPSILON1 .0000000000001
#define EPSILON2 .00000000001

#endif
