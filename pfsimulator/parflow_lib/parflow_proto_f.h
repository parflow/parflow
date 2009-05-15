/*BHEADER**********************************************************************

  Copyright (c) 1995-2009, Lawrence Livermore National Security,
  LLC. Produced at the Lawrence Livermore National Laboratory. Written
  by the Parflow Team (see the CONTRIBUTORS file)
  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.

  This file is part of Parflow. For details, see
  http://www.llnl.gov/casc/parflow

  Please read the COPYRIGHT file or Our Notice and the LICENSE file
  for the GNU Lesser General Public License.

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License (as published
  by the Free Software Foundation) version 2.1 dated February 1999.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
  and conditions of the GNU General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
  USA
**********************************************************************EHEADER*/

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


/* sk: clm.F90*/
#define CLM_LSM clm_lsm_

#define CALL_CLM_LSM(pressure_data,saturation_data,evap_trans_data,mask,porosity_data, \
                     dt, t, dx, dy, dz, ix, iy, nx, ny, nz, nx_f, ny_f, nz_f, ip, p, q, r, rank, clm_dump_interval, clm_1d_out, clm_file_dir , clm_file_dir_length, clm_bin_out_dir) \
	CLM_LSM(pressure_data, saturation_data,evap_trans_data,mask,porosity_data, \
                &dt, &t, &dx, &dy, &dz, &ix, &iy, &nx, &ny, &nz, &nx_f, &ny_f, &nz_f, &ip, &p, &q, &r, &rank, &clm_dump_interval, &clm_1d_out, clm_file_dir, &clm_file_dir_length, &clm_bin_out_dir);

void CLM_LSM( double *pressure_data, double *saturation_data, double *evap_trans_data, double *mask, double *porosity_data, 
              double *dt, double *t, double *dx, double *dy, double *dz, int *ix, int *iy, int *nx, int *ny, int *nz, 
              int *nx_f, int *ny_f, int *nz_f, int *ip, int *p, int *q, int *r, int *rank, int *clm_dump_interval, int *clm_1d_out, char *clm_file_dir, int *clm_file_dir_length, int *clm_bin_out_dir);
