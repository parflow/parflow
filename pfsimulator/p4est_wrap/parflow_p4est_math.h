#ifndef PARLOW_P4EST_MATH_H
#define PARFLOW_P4EST_MATH_H

/** Greatest common divisor routine
 *  \param [in] a   integer
 *  \param [in] b   integer
 *  \return         Greatest common divisor of \a and \b
 */
int             parflow_p4est_gcd(int a, int b);

/** Biggest power of 2 divisor routine
 *  \param [in] a   integer
 *  \return         Greatest power of 2 that divides \a
 */
int             parflow_p4est_powtwo_div(int a);

/** Compare integers functions
 *  \param [in] a   integer
 *  \param [in] b   integer
 *  \return         0 if \a == \b, -1 if \a < \b and +1 if \a > \b
 */
int             parflow_p4est_int_compare(int64_t a, int64_t b);

#endif                          // !PARFLOW_P4EST_MATH_H
