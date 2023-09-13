/* dposl.f -- translated by f2c (version 19940927).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

#include "f2c.h"

/* Table of constant values */

static int c__1 = 1;

/* Subroutine */ int dposl_(
                            double *a,
                            int *   lda,
                            int *   n,
                            double *b)
{
  /* System generated locals */
  int a_dim1, a_offset, i__1, i__2;

  /* Local variables */

  extern double ddot_(
                      int *   n,
                      double *dx,
                      int *   incx,
                      double *dy,
                      int *   incy);

  /* Subroutine */ int daxpy_(
                              int *   n,
                              double *da,
                              double *dx,
                              int *   incx,
                              double *dy,
                              int *   incy);

  static int k;
  static double t;

  static int kb;


/*     dposl solves the double precision symmetric positive definite */
/*     system a * x = b */
/*     using the factors computed by dpoco or dpofa. */

/*     on entry */

/*        a       double precision(lda, n) */
/*                the output from dpoco or dpofa. */

/*        lda     int */
/*                the leading dimension of the array  a . */

/*        n       int */
/*                the order of the matrix  a . */

/*        b       double precision(n) */
/*                the right hand side vector. */

/*     on return */

/*        b       the solution vector  x . */

/*     error condition */

/*        a division by zero will occur if the input factor contains */
/*        a zero on the diagonal.  technically this indicates */
/*        singularity but it is usually caused by improper subroutine */
/*        arguments.  it will not occur if the subroutines are called */
/*        correctly and  info .eq. 0 . */

/*     to compute  inverse(a) * c  where  c  is a matrix */
/*     with  p  columns */
/*           call dpoco(a,lda,n,rcond,z,info) */
/*           if (rcond is too small .or. info .ne. 0) go to ... */
/*           do 10 j = 1, p */
/*              call dposl(a,lda,n,c(1,j)) */
/*        10 continue */

/*     linpack.  this version dated 08/14/78 . */
/*     cleve moler, university of new mexico, argonne national lab. */

/*     subroutines and functions */

/*     blas daxpy,ddot */

/*     internal variables */


/*     solve trans(r)*y = b */

  /* Parameter adjustments */
  a_dim1 = *lda;
  a_offset = a_dim1 + 1;
  a -= a_offset;
  --b;

  /* Function Body */
  i__1 = *n;
  for (k = 1; k <= i__1; ++k)
  {
    i__2 = k - 1;
    t = ddot_(&i__2, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
    b[k] = (b[k] - t) / a[k + k * a_dim1];
/* L10: */
  }

/*     solve r*x = y */

  i__1 = *n;
  for (kb = 1; kb <= i__1; ++kb)
  {
    k = *n + 1 - kb;
    b[k] /= a[k + k * a_dim1];
    t = -b[k];
    i__2 = k - 1;
    daxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
/* L20: */
  }
  return 0;
} /* dposl_ */

/* Subroutine */ int daxpy_(
                            int *   n,
                            double *da,
                            double *dx,
                            int *   incx,
                            double *dy,
                            int *   incy)
{
  /* System generated locals */
  int i__1;

  /* Local variables */
  static int i, m, ix, iy, mp1;


/*     constant times a vector plus a vector. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --dy;
  --dx;

  /* Function Body */
  if (*n <= 0)
  {
    return 0;
  }
  if (*da == 0.)
  {
    return 0;
  }
  if (*incx == 1 && *incy == 1)
  {
    goto L20;
  }

/*        code for unequal increments or equal increments */
/*          not equal to 1 */

  ix = 1;
  iy = 1;
  if (*incx < 0)
  {
    ix = (-(*n) + 1) * *incx + 1;
  }
  if (*incy < 0)
  {
    iy = (-(*n) + 1) * *incy + 1;
  }
  i__1 = *n;
  for (i = 1; i <= i__1; ++i)
  {
    dy[iy] += *da * dx[ix];
    ix += *incx;
    iy += *incy;
/* L10: */
  }
  return 0;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
  m = *n % 4;
  if (m == 0)
  {
    goto L40;
  }
  i__1 = m;
  for (i = 1; i <= i__1; ++i)
  {
    dy[i] += *da * dx[i];
/* L30: */
  }
  if (*n < 4)
  {
    return 0;
  }
L40:
  mp1 = m + 1;
  i__1 = *n;
  for (i = mp1; i <= i__1; i += 4)
  {
    dy[i] += *da * dx[i];
    dy[i + 1] += *da * dx[i + 1];
    dy[i + 2] += *da * dx[i + 2];
    dy[i + 3] += *da * dx[i + 3];
/* L50: */
  }
  return 0;
} /* daxpy_ */


