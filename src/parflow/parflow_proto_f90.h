/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * C to Fortran interfacing macros
 *
 *****************************************************************************/

/* advect.f */
#if defined(_CRAYMPP) 
#define ADVECT ADVECT
#else
#define ADVECT advect_
#endif

#define CALL_ADVECT(s, sn, uedge, vedge, wedge, phi,\
                    slx, sly, slz,\
                    lo, hi, dlo, dhi, hx, dt, fstord,\
                    sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,\
                    dxscr, dyscr, dzscr, dzfrm) \
             ADVECT(s, sn, uedge, vedge, wedge, phi,\
                    slx, sly, slz,\
                    lo, hi, dlo, dhi, hx, &dt, &fstord,\
                    sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,\
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
#else
#define SADVECT sadvect_
#endif

#define CALL_SADVECT(s, sn, uedge, vedge, wedge, betaedge, phi,\
                     viscosity, density, gravity,\
                     slx, sly, slz,\
                     lohi, dlohi, hx, dt, \
                     sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,\
                     dxscr, dyscr, dzscr, dzfrm) \
             SADVECT(s, sn, uedge, vedge, wedge, betaedge, phi,\
                     viscosity, density, &gravity,\
                     slx, sly, slz,\
                     lohi, dlohi, hx, &dt, \
                     sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz,\
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
