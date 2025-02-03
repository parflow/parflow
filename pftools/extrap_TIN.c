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
/****************************************************************
* File: extrap_TIN.c
*
* Written by:   Bill Bosl
*              Lawrence Livermore National Lab
*               e-mail: wjbosl@llnl.gov
*               phone:  (510) 423-2873
*
* Purpose:      This program will take a gms TIN file that has
*               NOT been triangulated and fit a least-squares
*               plane through the surface(s) described by each set
*               of points. It then creates new points, presumably
*               points outside the original domain, using the
*               computed plane. The primary use for this program
*               will be to extrapolate reasonably linear surfaces
*               defined by a set of points out to some larger domain.
*
*               More than one surface may be present in the input TIN
*               file. Each surface will be extrapolated to the same points
*               in the order found in the input file, and a new TIN
*               file will be written with the new points listed first.
*
****************************************************************/

#include <stdio.h>
#include "f2c.h"

#define N_NEW_PTS 8             /* Here we specify the number of new
                                 * points that will be added to each
                                 * of the surfaces. The x-y coordinates
                                 * for each of these points must be
                                 * entered below. */
#define TMPFILE "tmp.out"

main(argc, argv)
int argc;
char **argv;
{
  FILE         *infile, *outfile, *tmpfile;
  char line[80], string[50];
  int more_TINs;
  int nv, nt;
  float        *x, *y, *z;
  float a[3];
  int v1, v2, v3, vis;
  int i, j, offset;
  char         *end_of_file;

  float        *x_new,
               *y_new,
               *z_new;

  float A[9];
  float b[3];

  int ipvt[9];
  int info, lda;
  int job;


  /*-----------------------------------------------
   *  Statement about expected command line for this
   *  program.
   *  -----------------------------------------------*/
  if (argc != 3)
  {
    fprintf(stderr, "Usage:  extrapTIN infile.TIN outfile.TIN\n");
    exit(1);
  }

  /*-----------------------------------------------
   *  Define the x-y coordinates of the new points
   *  that are desired.
   *   -----------------------------------------------*/
  x_new = (float*)malloc(N_NEW_PTS * sizeof(float));
  y_new = (float*)malloc(N_NEW_PTS * sizeof(float));

  x_new[0] = 100.0;    y_new[0] = 100.0;
  x_new[1] = 100.0;    y_new[1] = 12800.0;
  x_new[2] = 12800.0;  y_new[2] = 100.0;
  x_new[3] = 12800.0;  y_new[3] = 12800.0;
  x_new[4] = 12800.0;  y_new[4] = 8000.0;
  x_new[5] = 12800.0;  y_new[5] = 9000.0;
  x_new[6] = 12800.0;  y_new[6] = 10000.0;
  x_new[7] = 12800.0;  y_new[7] = 11000.0;

  /* Corresponding z values will be computed from the least squares plane */
  z_new = (float*)malloc(N_NEW_PTS * sizeof(float));

  /*-----------------------------------------------
   *  Open the i/o files and read in the data
   *  -----------------------------------------------*/
  infile = fopen(argv[1], "r");
  outfile = fopen(argv[2], "w");

  fgets(line, 80, infile);
  if (strncmp(line, "TIN", 3))
  {
    printf("%s is not a gms TIN file\n", argv[1]);
    exit();
  }
  fprintf(outfile, "%s", line);

  fgets(line, 80, infile);
  if (!strncmp(line, "BEGT", 4))
  {
    more_TINs = 1;
    fprintf(outfile, "%s", line);
  }
  else
  {
    printf("Expecting BEGT; line = %s\n", line);
    exit();
  }

  while (more_TINs)
  {
    fgets(line, 80, infile);
    if (!strncmp(line, "TNAM", 4))
    {
      /* Ignore the TNAM line and go on to the next line */
      fprintf(outfile, "%s", line);
      fgets(line, 80, infile);
    }

    if (!strncmp(line, "MAT", 3))
    {
      fprintf(outfile, "%s", line);
      fgets(line, 80, infile);
    }

    if (strncmp(line, "VERT", 4))
    {
      printf("VERT appears to be out of place in file %s\n", argv[1]);
      printf("line = %s\n", line);
      exit();
    }
    else
      sscanf(line, "%s %d", string, &nv);

    x = (float*)malloc((nv + N_NEW_PTS) * sizeof(float));
    y = (float*)malloc((nv + N_NEW_PTS) * sizeof(float));
    z = (float*)malloc((nv + N_NEW_PTS) * sizeof(float));
    for (i = 0; i < nv; i++)
    {
      offset = i + N_NEW_PTS;
      fscanf(infile, "%f %f %f\n", x + offset, y + offset, z + offset);
    }

    /*-----------------------------------------------
     *  Now that we have the data, compute a[i], where
     *  z = a[0]*x + a[1]*y + a[2]
     *  -----------------------------------------------*/
    for (i = 0; i < 3; i++)
      b[i] = 0.0;
    for (i = 0; i < 9; i++)
      A[i] = 0.0;

    /* Note that matrix A is a Fortran-style vector
     *
     *  A[0,0] A[0,1] A[0,2]          A[0]  A[3]  A[6]
     *  A[1,0] A[1,1] A[1,2]   ==>    A[1]  A[4]  A[7]
     *  A[2,0] A[2,1] A[2,2]          A[2]  A[5]  A[8]
     */
    for (i = N_NEW_PTS; i < nv + N_NEW_PTS; i++)
    {
      A[0] += x[i] * x[i];
      A[1] += x[i] * y[i];
      A[2] += x[i];
      A[4] += y[i] * y[i];
      A[5] += y[i];
      b[0] += x[i] * z[i];
      b[1] += y[i] * z[i];
      b[2] += z[i];
    }
    A[3] = A[1];
    A[6] = A[2];
    A[7] = A[5];
    A[8] = nv;

    /* Compute the elements of a here */
    lda = 3;
    job = 0;
    sgefa(A, &lda, &lda, ipvt, &info);
    sgesl(A, &lda, &lda, ipvt, b, &job);

    a[0] = b[0];
    a[1] = b[1];
    a[2] = b[2];

    for (i = 0; i < N_NEW_PTS; i++)
    {
      x[i] = x_new[i];
      y[i] = y_new[i];
      z[i] = a[0] * x[i] + a[1] * y[i] + a[2];
    }

    /*-----------------------------------------------
     *  Print out the original and new vertices
     *  -----------------------------------------------*/
    nv += N_NEW_PTS;
    fprintf(outfile, "VERT %d\n", nv);

    for (i = 0; i < nv; i++)
    {
      fprintf(outfile, "%f %f %f\n", x[i], y[i], z[i]);
    }

    /* This should be the ENDT keyword */
    fgets(line, 80, infile);
    fprintf(outfile, "%s\n", "ENDT");

    /* This will be the end-of-file unless the BEGT keyword is found */
    end_of_file = fgets(line, 80, infile);

    more_TINs = 0;
    if (!strncmp(line, "TRI", 3))
    {
      fprintf("TRI keyword found in %s; this program works\n", argv[1]);
      fprintf("only on TIN files that have NOT been triangulated; only\n");
      fprintf("vertices should be present. Exiting ...\n");
      exit;
    }
    else if (end_of_file != NULL)
    {
      more_TINs = 1;
      fprintf(outfile, "%s", line);
    }

    free(x);
    free(y);
    free(z);
  }

  close(infile);
  close(outfile);
} /* end of extrapTIN function (main program) */

/*================================================================
 *  Following are linpack routines that were converted to C using
 *  f2c. These general linear system solvers are called by extrapTIN
 *  and allow this file to be self-contained.
 *  ================================================================*/

/* sgefa.f -- translated by f2c (version 19940927).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */

/* Table of constant values */

static integer c__1 = 1;

/* Subroutine */ int sgefa(a, lda, n, ipvt, info)
real * a;
integer *lda, *n, *ipvt, *info;
{
  /* System generated locals */
  integer a_dim1, a_offset, i__1, i__2, i__3;

  /* Local variables */
  static integer j, k, l;
  static real t;
  extern /* Subroutine */ int sscal_(), saxpy_();
  extern integer isamax_();
  static integer kp1, nm1;


/*     sgefa factors a real matrix by gaussian elimination. */

/*     sgefa is usually called by sgeco, but it can be called */
/*     directly with a saving in time if  rcond  is not needed. */
/*     (time for sgeco) = (1 + 9/n)*(time for sgefa) . */

/*     on entry */

/*        a       real(lda, n) */
/*                the matrix to be factored. */

/*        lda     integer */
/*                the leading dimension of the array  a . */

/*        n       integer */
/*                the order of the matrix  a . */

/*     on return */

/*        a       an upper triangular matrix and the multipliers */
/*                which were used to obtain it. */
/*                the factorization can be written  a = l*u  where */
/*                l  is a product of permutation and unit lower */
/*                triangular matrices and  u  is upper triangular. */

/*        ipvt    integer(n) */
/*                an integer vector of pivot indices. */

/*        info    integer */
/*                = 0  normal value. */
/*                = k  if  u(k,k) .eq. 0.0 .  this is not an error */
/*                     condition for this subroutine, but it does */
/*                     indicate that sgesl or sgedi will divide by zero */
/*                     if called.  use  rcond  in sgeco for a reliable */
/*                     indication of singularity. */

/*     linpack. this version dated 08/14/78 . */
/*     cleve moler, university of new mexico, argonne national lab. */

/*     subroutines and functions */

/*     blas saxpy,sscal,isamax */

/*     internal variables */



/*     gaussian elimination with partial pivoting */

  /* Parameter adjustments */
  a_dim1 = *lda;
  a_offset = a_dim1 + 1;
  a -= a_offset;
  --ipvt;

  /* Function Body */
  *info = 0;
  nm1 = *n - 1;
  if (nm1 < 1)
  {
    goto L70;
  }
  i__1 = nm1;
  for (k = 1; k <= i__1; ++k)
  {
    kp1 = k + 1;

/*        find l = pivot index */

    i__2 = *n - k + 1;
    l = isamax_(&i__2, &a[k + k * a_dim1], &c__1) + k - 1;
    ipvt[k] = l;

/*        zero pivot implies this column already triangularized */

    if (a[l + k * a_dim1] == (float)0.)
    {
      goto L40;
    }

/*           interchange if necessary */

    if (l == k)
    {
      goto L10;
    }
    t = a[l + k * a_dim1];
    a[l + k * a_dim1] = a[k + k * a_dim1];
    a[k + k * a_dim1] = t;
L10:

/*           compute multipliers */

    t = (float)-1. / a[k + k * a_dim1];
    i__2 = *n - k;
    sscal_(&i__2, &t, &a[k + 1 + k * a_dim1], &c__1);

/*           row elimination with column indexing */

    i__2 = *n;
    for (j = kp1; j <= i__2; ++j)
    {
      t = a[l + j * a_dim1];
      if (l == k)
      {
        goto L20;
      }
      a[l + j * a_dim1] = a[k + j * a_dim1];
      a[k + j * a_dim1] = t;
L20:
      i__3 = *n - k;
      saxpy_(&i__3, &t, &a[k + 1 + k * a_dim1], &c__1, &a[k + 1 + j *
                                                          a_dim1], &c__1);
/* L30: */
    }
    goto L50;
L40:
    *info = k;
L50:
/* L60: */
    ;
  }
L70:
  ipvt[*n] = *n;
  if (a[*n + *n * a_dim1] == (float)0.)
  {
    *info = *n;
  }
  return 0;
} /* sgefa_ */

integer isamax_(n, sx, incx)
integer * n;
real *sx;
integer *incx;
{
  /* System generated locals */
  integer ret_val, i__1;
  real r__1;

  /* Local variables */
  static real smax;
  static integer i, ix;


/*     finds the index of element having max. absolute value. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 3/93 to return if incx .le. 0. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --sx;

  /* Function Body */
  ret_val = 0;
  if (*n < 1 || *incx <= 0)
  {
    return ret_val;
  }
  ret_val = 1;
  if (*n == 1)
  {
    return ret_val;
  }
  if (*incx == 1)
  {
    goto L20;
  }

/*        code for increment not equal to 1 */

  ix = 1;
  smax = dabs(sx[1]);
  ix += *incx;
  i__1 = *n;
  for (i = 2; i <= i__1; ++i)
  {
    if ((r__1 = sx[ix], dabs(r__1)) <= smax)
    {
      goto L5;
    }
    ret_val = i;
    smax = (r__1 = sx[ix], dabs(r__1));
L5:
    ix += *incx;
/* L10: */
  }
  return ret_val;

/*        code for increment equal to 1 */

L20:
  smax = dabs(sx[1]);
  i__1 = *n;
  for (i = 2; i <= i__1; ++i)
  {
    if ((r__1 = sx[i], dabs(r__1)) <= smax)
    {
      goto L30;
    }
    ret_val = i;
    smax = (r__1 = sx[i], dabs(r__1));
L30:
    ;
  }
  return ret_val;
} /* isamax_ */

/* Subroutine */ int saxpy_(n, sa, sx, incx, sy, incy)
integer * n;
real *sa, *sx;
integer *incx;
real *sy;
integer *incy;
{
  /* System generated locals */
  integer i__1;

  /* Local variables */
  static integer i, m, ix, iy, mp1;


/*     constant times a vector plus a vector. */
/*     uses unrolled loop for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --sy;
  --sx;

  /* Function Body */
  if (*n <= 0)
  {
    return 0;
  }
  if (*sa == (float)0.)
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
    sy[iy] += *sa * sx[ix];
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
    sy[i] += *sa * sx[i];
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
    sy[i] += *sa * sx[i];
    sy[i + 1] += *sa * sx[i + 1];
    sy[i + 2] += *sa * sx[i + 2];
    sy[i + 3] += *sa * sx[i + 3];
/* L50: */
  }
  return 0;
} /* saxpy_ */

/* Subroutine */ int sscal_(n, sa, sx, incx)
integer * n;
real *sa, *sx;
integer *incx;
{
  /* System generated locals */
  integer i__1, i__2;

  /* Local variables */
  static integer i, m, nincx, mp1;


/*     scales a vector by a constant. */
/*     uses unrolled loops for increment equal to 1. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 3/93 to return if incx .le. 0. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --sx;

  /* Function Body */
  if (*n <= 0 || *incx <= 0)
  {
    return 0;
  }
  if (*incx == 1)
  {
    goto L20;
  }

/*        code for increment not equal to 1 */

  nincx = *n * *incx;
  i__1 = nincx;
  i__2 = *incx;
  for (i = 1; i__2 < 0 ? i >= i__1 : i <= i__1; i += i__2)
  {
    sx[i] = *sa * sx[i];
/* L10: */
  }
  return 0;

/*        code for increment equal to 1 */


/*        clean-up loop */

L20:
  m = *n % 5;
  if (m == 0)
  {
    goto L40;
  }
  i__2 = m;
  for (i = 1; i <= i__2; ++i)
  {
    sx[i] = *sa * sx[i];
/* L30: */
  }
  if (*n < 5)
  {
    return 0;
  }
L40:
  mp1 = m + 1;
  i__2 = *n;
  for (i = mp1; i <= i__2; i += 5)
  {
    sx[i] = *sa * sx[i];
    sx[i + 1] = *sa * sx[i + 1];
    sx[i + 2] = *sa * sx[i + 2];
    sx[i + 3] = *sa * sx[i + 3];
    sx[i + 4] = *sa * sx[i + 4];
/* L50: */
  }
  return 0;
} /* sscal_ */

/* sgesl.f -- translated by f2c (version 19940927).
 * You must link the resulting object file with the libraries:
 *      -lf2c -lm   (in that order)
 */


/* Subroutine */ int sgesl(a, lda, n, ipvt, b, job)
real * a;
integer *lda, *n, *ipvt;
real *b;
integer *job;
{
  /* System generated locals */
  integer a_dim1, a_offset, i__1, i__2;

  /* Local variables */
  extern doublereal sdot_();
  static integer k, l;
  static real t;
  extern /* Subroutine */ int saxpy_();
  static integer kb, nm1;


/*     sgesl solves the real system */
/*     a * x = b  or  trans(a) * x = b */
/*     using the factors computed by sgeco or sgefa. */

/*     on entry */

/*        a       real(lda, n) */
/*                the output from sgeco or sgefa. */

/*        lda     integer */
/*                the leading dimension of the array  a . */

/*        n       integer */
/*                the order of the matrix  a . */

/*        ipvt    integer(n) */
/*                the pivot vector from sgeco or sgefa. */

/*        b       real(n) */
/*                the right hand side vector. */

/*        job     integer */
/*                = 0         to solve  a*x = b , */
/*                = nonzero   to solve  trans(a)*x = b  where */
/*                            trans(a)  is the transpose. */

/*     on return */

/*        b       the solution vector  x . */

/*     error condition */

/*        a division by zero will occur if the input factor contains a */
/*        zero on the diagonal.  technically this indicates singularity */
/*        but it is often caused by improper arguments or improper */
/*        setting of lda .  it will not occur if the subroutines are */
/*        called correctly and if sgeco has set rcond .gt. 0.0 */
/*        or sgefa has set info .eq. 0 . */

/*     to compute  inverse(a) * c  where  c  is a matrix */
/*     with  p  columns */
/*           call sgeco(a,lda,n,ipvt,rcond,z) */
/*           if (rcond is too small) go to ... */
/*           do 10 j = 1, p */
/*              call sgesl(a,lda,n,ipvt,c(1,j),0) */
/*        10 continue */

/*     linpack. this version dated 08/14/78 . */
/*     cleve moler, university of new mexico, argonne national lab. */

/*     subroutines and functions */

/*     blas saxpy,sdot */

/*     internal variables */


  /* Parameter adjustments */
  a_dim1 = *lda;
  a_offset = a_dim1 + 1;
  a -= a_offset;
  --ipvt;
  --b;

  /* Function Body */
  nm1 = *n - 1;
  if (*job != 0)
  {
    goto L50;
  }

/*        job = 0 , solve  a * x = b */
/*        first solve  l*y = b */

  if (nm1 < 1)
  {
    goto L30;
  }
  i__1 = nm1;
  for (k = 1; k <= i__1; ++k)
  {
    l = ipvt[k];
    t = b[l];
    if (l == k)
    {
      goto L10;
    }
    b[l] = b[k];
    b[k] = t;
L10:
    i__2 = *n - k;
    saxpy_(&i__2, &t, &a[k + 1 + k * a_dim1], &c__1, &b[k + 1], &c__1);
/* L20: */
  }
L30:

/*        now solve  u*x = y */

  i__1 = *n;
  for (kb = 1; kb <= i__1; ++kb)
  {
    k = *n + 1 - kb;
    b[k] /= a[k + k * a_dim1];
    t = -(doublereal)b[k];
    i__2 = k - 1;
    saxpy_(&i__2, &t, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
/* L40: */
  }
  goto L100;
L50:

/*        job = nonzero, solve  trans(a) * x = b */
/*        first solve  trans(u)*y = b */

  i__1 = *n;
  for (k = 1; k <= i__1; ++k)
  {
    i__2 = k - 1;
    t = sdot_(&i__2, &a[k * a_dim1 + 1], &c__1, &b[1], &c__1);
    b[k] = (b[k] - t) / a[k + k * a_dim1];
/* L60: */
  }

/*        now solve trans(l)*x = y */

  if (nm1 < 1)
  {
    goto L90;
  }
  i__1 = nm1;
  for (kb = 1; kb <= i__1; ++kb)
  {
    k = *n - kb;
    i__2 = *n - k;
    b[k] += sdot_(&i__2, &a[k + 1 + k * a_dim1], &c__1, &b[k + 1], &c__1);
    l = ipvt[k];
    if (l == k)
    {
      goto L70;
    }
    t = b[l];
    b[l] = b[k];
    b[k] = t;
L70:
/* L80: */
    ;
  }
L90:
L100:
  return 0;
} /* sgesl_ */

doublereal sdot_(n, sx, incx, sy, incy)
integer * n;
real *sx;
integer *incx;
real *sy;
integer *incy;
{
  /* System generated locals */
  integer i__1;
  real ret_val;

  /* Local variables */
  static integer i, m;
  static real stemp;
  static integer ix, iy, mp1;


/*     forms the dot product of two vectors. */
/*     uses unrolled loops for increments equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


  /* Parameter adjustments */
  --sy;
  --sx;

  /* Function Body */
  stemp = (float)0.;
  ret_val = (float)0.;
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
    stemp += sx[ix] * sy[iy];
    ix += *incx;
    iy += *incy;
/* L10: */
  }
  ret_val = stemp;
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
    stemp += sx[i] * sy[i];
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
    stemp = stemp + sx[i] * sy[i] + sx[i + 1] * sy[i + 1] + sx[i + 2] *
            sy[i + 2] + sx[i + 3] * sy[i + 3] + sx[i + 4] * sy[i + 4];
/* L50: */
  }
L60:
  ret_val = stemp;
  return ret_val;
} /* sdot_ */

