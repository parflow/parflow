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
* C to Fortran interfacing macros
*
*****************************************************************************/

/* advect.f */
#if defined(_CRAYMPP)
#define ADVECT ADVECT
#elif defined(__bg__)
#define ADVECT advect
#else
#define ADVECT advect_
#endif

#define CALL_ADVECT(s, sn, uedge, vedge, wedge, phi,                            \
                    slx, sly, slz,                                              \
                    lo, hi, dlo, dhi, hx, dt, fstord,                           \
                    sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,       \
                    dxscr, dyscr, dzscr, dzfrm)                                 \
        ADVECT(s, sn, uedge, vedge, wedge, phi,                                 \
               slx, sly, slz,                                                   \
               lo, hi, dlo, dhi, hx, &dt, &fstord,                              \
               sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,            \
               dxscr, dyscr, dzscr, dzfrm)

void ADVECT(double *s, double *sn,
            double *uedge, double *vedge, double *wedge, double *phi,
            double *slx, double *sly, double *slz,
            int *lo, int *hi, int *dlo, int *dhi, double *hx, double *dt, int *fstord,
            double *sbot, double *stop, double *sbotp,
            double *sfrt, double *sbck,
            double *sleft, double *sright, double *sfluxz,
            double *dxscr, double *dyscr, double *dzscr, double *dzfrm);

/* sadvect.f */
#if defined(_CRAYMPP)
#define SADVECT SADVECT
#else define (__bg__)
#define SADVECT sadvect
#else
#define SADVECT sadvect_
#endif

#define CALL_SADVECT(s, sn, uedge, vedge, wedge, betaedge, phi,                  \
                     viscosity, density, gravity,                                \
                     slx, sly, slz,                                              \
                     lohi, dlohi, hx, dt,                                        \
                     sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,       \
                     dxscr, dyscr, dzscr, dzfrm)                                 \
        SADVECT(s, sn, uedge, vedge, wedge, betaedge, phi,                       \
                viscosity, density, &gravity,                                    \
                slx, sly, slz,                                                   \
                lohi, dlohi, hx, &dt,                                            \
                sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,            \
                dxscr, dyscr, dzscr, dzfrm)

void SADVECT(double *s, double *sn,
             double *uedge, double *vedge, double *wedge, double *betaedge, double *phi,
             double *viscosity, double *density, double *gravity,
             double *slx, double *sly, double *slz,
             int *lohi, int *dlohi, double *hx, double *dt,
             double *sbot, double *stop, double *sbotp,
             double *sfrt, double *sbck,
             double *sleft, double *sright, double *sfluxz,
             double *dxscr, double *dyscr, double *dzscr, double *dzfrm);

/* sk: ftest.f90*/
#define FTEST ftest_

#define CALL_FTEST(outflow_log) \
        FTEST(outflow_log);

void FTEST(double *outflow_log);
