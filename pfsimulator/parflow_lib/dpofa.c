/* dpofa.f -- translated by f2c (version 19940927).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

#include "f2c.h"
#include <math.h>

/* Table of constant values */

static int c__1 = 1;

/* Subroutine */ int dpofa_(
                            double *a,
                            int *   lda,
                            int *   n,
                            int *   info)
{
  /* System generated locals */
  int a_dim1, a_offset, i__1, i__2, i__3;

  /* Builtin functions */
//    double sqrt();

  /* Local variables */
  extern double ddot_(
                      int *   n,
                      double *dx,
                      int *   incx,
                      double *dy,
                      int *   incy);
  static int j, k;
  static double s, t;
  static int jm1;


/*     dpofa factors a double precision symmetric positive definite */
/*     matrix. */

/*     dpofa is usually called by dpoco, but it can be called */
/*     directly with a saving in time if  rcond  is not needed. */
/*     (time for dpoco) = (1 + 18/n)*(time for dpofa) . */

/*     on entry */

/*        a       double precision(lda, n) */
/*                the symmetric matrix to be factored.  only the */
/*                diagonal and upper triangle are used. */

/*        lda     int */
/*                the leading dimension of the array  a . */

/*        n       int */
/*                the order of the matrix  a . */

/*     on return */

/*        a       an upper triangular matrix  r  so that  a = trans(r)*r
 */
/*                where  trans(r)  is the transpose. */
/*                the strict lower triangle is unaltered. */
/*                if  info .ne. 0 , the factorization is not complete. */

/*        info    int */
/*                = 0  for normal return. */
/*                = k  signals an error condition.  the leading minor */
/*                     of order  k  is not positive definite. */

/*     linpack.  this version dated 08/14/78 . */
/*     cleve moler, university of new mexico, argonne national lab. */

/*     subroutines and functions */

/*     blas ddot */
/*     fortran dsqrt */

/*     internal variables */

/*     begin block with ...exits to 40 */


  /* Parameter adjustments */
  a_dim1 = *lda;
  a_offset = a_dim1 + 1;
  a -= a_offset;

  /* Function Body */
  i__1 = *n;
  for (j = 1; j <= i__1; ++j)
  {
    *info = j;
    s = 0.;
    jm1 = j - 1;
    if (jm1 < 1)
    {
      goto L20;
    }
    i__2 = jm1;
    for (k = 1; k <= i__2; ++k)
    {
      i__3 = k - 1;
      t = a[k + j * a_dim1] - ddot_(&i__3, &a[k * a_dim1 + 1], &c__1, &
                                    a[j * a_dim1 + 1], &c__1);
      t /= a[k + k * a_dim1];
      a[k + j * a_dim1] = t;
      s += t * t;
/* L10: */
    }
L20:
    s = a[j + j * a_dim1] - s;
/*     ......exit */
    if (s <= 0.)
    {
      goto L40;
    }
    a[j + j * a_dim1] = sqrt(s);
/* L30: */
  }
  *info = 0;
L40:
  return 0;
} /* dpofa_ */

double ddot_(
             int *   n,
             double *dx,
             int *   incx,
             double *dy,
             int *   incy)
{
  /* System generated locals */
  int i__1;
  double ret_val;

  /* Local variables */
  static int i, m;
  static double dtemp;
  static int ix, iy, mp1;


/*     forms the dot product of two vectors. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --dy;
  --dx;

  /* Function Body */
  ret_val = 0.;
  dtemp = 0.;
  if (*n <= 0)
  {
    return ret_val;
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
    dtemp += dx[ix] * dy[iy];
    ix += *incx;
    iy += *incy;
/* L10: */
  }
  ret_val = dtemp;
  return ret_val;

/*        code for both increments equal to 1 */


/*        clean-up loop */

L20:
  m = *n % 5;
  if (m == 0)
  {
    goto L40;
  }
  i__1 = m;
  for (i = 1; i <= i__1; ++i)
  {
    dtemp += dx[i] * dy[i];
/* L30: */
  }
  if (*n < 5)
  {
    goto L60;
  }
L40:
  mp1 = m + 1;
  i__1 = *n;
  for (i = mp1; i <= i__1; i += 5)
  {
    dtemp = dtemp + dx[i] * dy[i] + dx[i + 1] * dy[i + 1] + dx[i + 2] *
            dy[i + 2] + dx[i + 3] * dy[i + 3] + dx[i + 4] * dy[i + 4];
/* L50: */
  }
L60:
  ret_val = dtemp;
  return ret_val;
} /* ddot_ */

