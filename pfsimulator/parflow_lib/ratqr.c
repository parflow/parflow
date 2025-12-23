/*  -- translated by f2c (version 19940127).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

#include "f2c.h"

extern double epslon_(double *x);

/* Subroutine */ int ratqr_(
                            int *   n,
                            double *eps1,
                            double *d,
                            double *e,
                            double *e2,
                            int *   m,
                            double *w,
                            int *   ind,
                            double *bd,
                            int *   type,
                            int *   idef,
                            int *   ierr)
{
  /* System generated locals */
  integer i__1, i__2;
  doublereal d__1, d__2, d__3;

  /* Local variables */
  integer jdef;
  doublereal f;
  integer i, j, k;
  doublereal p, q, r, s, delta;
  integer k1, ii, jj;
  doublereal ep, qp;
  doublereal err, tot;



/*     this subroutine is a translation of the algol procedure ratqr, */
/*     num. math. 11, 264-272(1968) by reinsch and bauer. */
/*     handbook for auto. comp., vol.ii-linear algebra, 257-265(1971). */

/*     this subroutine finds the algebraically smallest or largest */
/*     eigenvalues of a symmetric tridiagonal matrix by the */
/*     rational qr method with newton corrections. */

/*     on input */

/*        n is the order of the matrix. */

/*        eps1 is a theoretical absolute error tolerance for the */
/*          computed eigenvalues.  if the input eps1 is non-positive, */
/*          or indeed smaller than its default value, it is reset */
/*          at each iteration to the respective default value, */
/*          namely, the product of the relative machine precision */
/*          and the magnitude of the current eigenvalue iterate. */
/*          the theoretical absolute error in the k-th eigenvalue */
/*          is usually not greater than k times eps1. */

/*        d contains the diagonal elements of the input matrix. */

/*        e contains the subdiagonal elements of the input matrix */
/*          in its last n-1 positions.  e(1) is arbitrary. */

/*        e2 contains the squares of the corresponding elements of e. */
/*          e2(1) is arbitrary. */

/*        m is the number of eigenvalues to be found. */

/*        idef should be set to 1 if the input matrix is known to be */
/*          positive definite, to -1 if the input matrix is known to */
/*          be negative definite, and to 0 otherwise. */

/*        type should be set to .true. if the smallest eigenvalues */
/*          are to be found, and to .false. if the largest eigenvalues */
/*          are to be found. */

/*     on output */

/*        eps1 is unaltered unless it has been reset to its */
/*          (last) default value. */

/*        d and e are unaltered (unless w overwrites d). */

/*        elements of e2, corresponding to elements of e regarded */
/*          as negligible, have been replaced by zero causing the */
/*          matrix to split into a direct sum of submatrices. */
/*          e2(1) is set to 0.0d0 if the smallest eigenvalues have been */
/*          found, and to 2.0d0 if the largest eigenvalues have been */
/*          found.  e2 is otherwise unaltered (unless overwritten by bd).
 */

/*        w contains the m algebraically smallest eigenvalues in */
/*          ascending order, or the m largest eigenvalues in */
/*          descending order.  if an error exit is made because of */
/*          an incorrect specification of idef, no eigenvalues */
/*          are found.  if the newton iterates for a particular */
/*          eigenvalue are not monotone, the best estimate obtained */
/*          is returned and ierr is set.  w may coincide with d. */

/*        ind contains in its first m positions the submatrix indices */
/*          associated with the corresponding eigenvalues in w -- */
/*          1 for eigenvalues belonging to the first submatrix from */
/*          the top, 2 for those belonging to the second submatrix, etc..
 */

/*        bd contains refined bounds for the theoretical errors of the */
/*          corresponding eigenvalues in w.  these bounds are usually */
/*          within the tolerance specified by eps1.  bd may coincide */
/*          with e2. */

/*        ierr is set to */
/*          zero       for normal return, */
/*          6*n+1      if  idef  is set to 1 and  type  to .true. */
/*                     when the matrix is not positive definite, or */
/*                     if  idef  is set to -1 and  type  to .false. */
/*                     when the matrix is not negative definite, */
/*          5*n+k      if successive iterates to the k-th eigenvalue */
/*                     are not monotone increasing, where k refers */
/*                     to the last such occurrence. */

/*     note that subroutine tridib is generally faster and more */
/*     accurate than ratqr if the eigenvalues are clustered. */

/*     questions and comments should be directed to burton s. garbow, */
/*     mathematics and computer science div, argonne national laboratory
 */

/*     this version dated august 1983. */

/*     ------------------------------------------------------------------
 */

  /* Parameter adjustments */
  --bd;
  --ind;
  --w;
  --e2;
  --e;
  --d;

  /* Function Body */
  *ierr = 0;
  jdef = *idef;
/*     .......... copy d array into w .......... */
  i__1 = *n;
  for (i = 1; i <= i__1; ++i)
  {
/* L20: */
    w[i] = d[i];
  }

  if (*type)
  {
    goto L40;
  }
  j = 1;
  goto L400;
L40:
  err = 0.;
  s = 0.;
/*     .......... look for small sub-diagonal entries and define */
/*                initial shift from lower gerschgorin bound. */
/*                copy e2 array into bd .......... */
  tot = w[1];
  q = 0.;
  j = 0;

  i__1 = *n;
  for (i = 1; i <= i__1; ++i)
  {
    p = q;
    if (i == 1)
    {
      goto L60;
    }
    d__3 = (d__1 = d[i], pfabs(d__1)) + (d__2 = d[i - 1], pfabs(d__2));
    if (p > epslon_(&d__3))
    {
      goto L80;
    }
L60:
    e2[i] = 0.;
L80:
    bd[i] = e2[i];
/*     .......... count also if element of e2 has underflowed ........
 * .. */
    if (e2[i] == 0.)
    {
      ++j;
    }
    ind[i] = j;
    q = 0.;
    if (i != *n)
    {
      q = (d__1 = e[i + 1], pfabs(d__1));
    }
/* Computing MIN */
    d__1 = w[i] - p - q;
    tot = pfmin(d__1, tot);
/* L100: */
  }

  if (jdef == 1 && tot < 0.)
  {
    goto L140;
  }

  i__1 = *n;
  for (i = 1; i <= i__1; ++i)
  {
/* L110: */
    w[i] -= tot;
  }

  goto L160;
L140:
  tot = 0.;

L160:
  i__1 = *m;
  for (k = 1; k <= i__1; ++k)
  {
/*     .......... next qr transformation .......... */
L180:
    tot += s;
    delta = w[*n] - s;
    i = *n;
    f = (d__1 = epslon_(&tot), pfabs(d__1));
    if (*eps1 < f)
    {
      *eps1 = f;
    }
    if (delta > *eps1)
    {
      goto L190;
    }
    if (delta < -(*eps1))
    {
      goto L1000;
    }
    goto L300;
/*     .......... replace small sub-diagonal squares by zero */
/*                to reduce the incidence of underflows .......... */
L190:
    if (k == *n)
    {
      goto L210;
    }
    k1 = k + 1;
    i__2 = *n;
    for (j = k1; j <= i__2; ++j)
    {
      d__2 = w[j] + w[j - 1];
/* Computing 2nd power */
      d__1 = epslon_(&d__2);
      if (bd[j] <= d__1 * d__1)
      {
        bd[j] = 0.;
      }
/* L200: */
    }

L210:
    f = bd[*n] / delta;
    qp = delta + f;
    p = 1.;
    if (k == *n)
    {
      goto L260;
    }
    k1 = *n - k;
/*     .......... for i=n-1 step -1 until k do -- .......... */
    i__2 = k1;
    for (ii = 1; ii <= i__2; ++ii)
    {
      i = *n - ii;
      q = w[i] - s - f;
      r = q / qp;
      p = p * r + 1.;
      ep = f * r;
      w[i + 1] = qp + ep;
      delta = q - ep;
      if (delta > *eps1)
      {
        goto L220;
      }
      if (delta < -(*eps1))
      {
        goto L1000;
      }
      goto L300;
L220:
      f = bd[i] / q;
      qp = delta + f;
      bd[i + 1] = qp * ep;
/* L240: */
    }

L260:
    w[k] = qp;
    s = qp / p;
    if (tot + s > tot)
    {
      goto L180;
    }
/*     .......... set error -- irregular end of iteration. */
/*                deflate minimum diagonal element .......... */
    *ierr = *n * 5 + k;
    s = 0.;
    delta = qp;

    i__2 = *n;
    for (j = k; j <= i__2; ++j)
    {
      if (w[j] > delta)
      {
        goto L280;
      }
      i = j;
      delta = w[j];
L280:
      ;
    }
/*     .......... convergence .......... */
L300:
    if (i < *n)
    {
      bd[i + 1] = bd[i] * f / qp;
    }
    ii = ind[i];
    if (i == k)
    {
      goto L340;
    }
    k1 = i - k;
/*     .......... for j=i-1 step -1 until k do -- .......... */
    i__2 = k1;
    for (jj = 1; jj <= i__2; ++jj)
    {
      j = i - jj;
      w[j + 1] = w[j] - s;
      bd[j + 1] = bd[j];
      ind[j + 1] = ind[j];
/* L320: */
    }

L340:
    w[k] = tot;
    err += pfabs(delta);
    bd[k] = err;
    ind[k] = ii;
/* L360: */
  }

  if (*type)
  {
    goto L1001;
  }
  f = bd[1];
  e2[1] = 2.;
  bd[1] = f;
  j = 2;
/*     .......... negate elements of w for largest values .......... */
L400:
  i__1 = *n;
  for (i = 1; i <= i__1; ++i)
  {
/* L500: */
    w[i] = -w[i];
  }

  jdef = -jdef;
  switch ((int)j)
  {
    case 1:  goto L40;

    case 2:  goto L1001;
  }
/*     .......... set error -- idef specified incorrectly .......... */
L1000:
  *ierr = *n * 6 + 1;
L1001:
  return 0;
} /* ratqr_ */

double epslon_(double *x)
{
  /* System generated locals */
  doublereal ret_val, d__1;

  /* Local variables */
  doublereal a, b, c, eps;


/*     estimate unit roundoff in quantities of size x. */


/*     this program should function properly on all systems */
/*     satisfying the following two assumptions, */
/*        1.  the base used in representing floating point */
/*            numbers is not a power of three. */
/*        2.  the quantity  a  in statement 10 is represented to */
/*            the accuracy used in floating point variables */
/*            that are stored in memory. */
/*     the statement number 10 and the go to 10 are intended to */
/*     force optimizing compilers to generate code satisfying */
/*     assumption 2. */
/*     under these assumptions, it should be true that, */
/*            a  is not exactly equal to four-thirds, */
/*            b  has a zero for its last bit or digit, */
/*            c  is not exactly equal to one, */
/*            eps  measures the separation of 1.0 from */
/*                 the next larger floating point number. */
/*     the developers of eispack would appreciate being informed */
/*     about any systems where these assumptions do not hold. */

/*     this version dated 4/6/83. */

  a = 1.3333333333333333;
L10:
  b = a - 1.;
  c = b + b + b;
  eps = (d__1 = c - 1., pfabs(d__1));
  if (eps == 0.)
  {
    goto L10;
  }
  ret_val = eps * pfabs(*x);
  return ret_val;
} /* epslon_ */

