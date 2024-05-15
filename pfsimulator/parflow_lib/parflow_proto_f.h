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

#ifndef PARFLOW_PROTO_F_H
#define PARFLOW_PROTO_F_H

/* These macros are used to get around a parsing issue with uncrustify. */
#ifdef __cplusplus
/** *INDENT-OFF* */
#define BEGIN_EXTERN_C extern "C" {
#define END_EXTERN_C }
/** *INDENT-ON* */
#else
#define BEGIN_EXTERN_C
#define END_EXTERN_C
#endif

BEGIN_EXTERN_C

/* advect.f */
#if defined(_CRAYMPP)
#define ADVECT ADVECT
#elif defined(__bg__)
#define ADVECT advect
#else
#define ADVECT advect_
#endif

#define CALL_ADVECT(s, sn, uedge, vedge, wedge, phi,                      \
                    slx, sly, slz,                                        \
                    lo, hi, dlo, dhi, hx, dt, fstord,                     \
                    sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz, \
                    dxscr, dyscr, dzscr, dzfrm)                           \
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
#elif defined(__bg__)
#define SADVECT sadvect
#else
#define SADVECT sadvect_
#endif

#define CALL_SADVECT(s, sn, uedge, vedge, wedge, betaedge, phi,            \
                     viscosity, density, gravity,                          \
                     slx, sly, slz,                                        \
                     lohi, dlohi, hx, dt,                                  \
                     sbot, stop, sbotp, sfrt, sbck, sleft, sright, sfluxz, \
                     dxscr, dyscr, dzscr, dzfrm)                           \
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


/* sk: clm.F90*/


#if defined(_CRAYMPP)
#define CLM_LSM CLM_LSM
#elif defined(__bg__)
#define CLM_LSM clm_lsm
#else
#define CLM_LSM clm_lsm_
#endif

#define CALL_CLM_LSM(pressure_data, saturation_data, evap_trans_data, mask, porosity_data,                                                                                     \
                     dz_mult_data, istep, dt, t, start_time, dx, dy, dz, ix, iy, nx, ny, nz,                                                                                   \
                     nx_f, ny_f, nz_f, nz_rz, ip, p, q, r, gnx, gny, rank,                                                                                                     \
                     sw_data, lw_data, prcp_data, tas_data, u_data, v_data, patm_data, qatm_data,                                                                              \
                     lai_data, sai_data, z0m_data, displa_data,                                                                                                                \
                     slope_x_data, slope_y_data,                                                                                                                               \
                     eflx_lh_tot_data, eflx_lwrad_out_data, eflx_sh_tot_data, eflx_soil_grnd_data,                                                                             \
                     qflx_evap_tot_data, qflx_evap_grnd_data, qflx_evap_soi_data, qflx_evap_veg_data, qflx_tran_veg_data,                                                      \
                     qflx_infl_data, swe_out_data, t_grnd_data, t_soil_data,                                                                                                   \
                     clm_dump_interval, clm_1d_out, clm_forc_veg, clm_file_dir, clm_file_dir_length, clm_bin_out_dir, write_CLM_binary, slope_accounting_CLM,                                       \
                     clm_beta_function, clm_veg_function, clm_veg_wilting, clm_veg_fieldc, clm_res_sat,                                                                        \
                     clm_irr_type, clm_irr_cycle, clm_irr_rate, clm_irr_start, clm_irr_stop,                                                                                   \
                     clm_irr_threshold, qirr, qirr_inst, iflag, clm_irr_thresholdtype, soi_z, clm_next, clm_write_logs, clm_last_rst, clm_daily_rst, clm_nlevsoi, clm_nlevlak) \
  CLM_LSM(pressure_data, saturation_data, evap_trans_data, mask, porosity_data,                                                                                                \
          dz_mult_data, &istep, &dt, &t, &start_time, &dx, &dy, &dz, &ix, &iy, &nx, &ny, &nz, &nx_f, &ny_f, &nz_f, &nz_rz, &ip, &p, &q, &r, &gnx, &gny, &rank,                 \
          sw_data, lw_data, prcp_data, tas_data, u_data, v_data, patm_data, qatm_data,                                                                                         \
          lai_data, sai_data, z0m_data, displa_data,                                                                                                                           \
          slope_x_data, slope_y_data,                                                                                                                                          \
          eflx_lh_tot_data, eflx_lwrad_out_data, eflx_sh_tot_data, eflx_soil_grnd_data,                                                                                        \
          qflx_evap_tot_data, qflx_evap_grnd_data, qflx_evap_soi_data, qflx_evap_veg_data, qflx_tran_veg_data,                                                                 \
          qflx_infl_data, swe_out_data, t_grnd_data, t_soil_data,                                                                                                              \
          &clm_dump_interval, &clm_1d_out, &clm_forc_veg, clm_file_dir, &clm_file_dir_length, &clm_bin_out_dir,                                                                \
          &write_CLM_binary, &slope_accounting_CLM, &clm_beta_function, &clm_veg_function, &clm_veg_wilting, &clm_veg_fieldc,                                                                         \
          &clm_res_sat, &clm_irr_type, &clm_irr_cycle, &clm_irr_rate, &clm_irr_start, &clm_irr_stop,                                                                           \
          &clm_irr_threshold, qirr, qirr_inst, iflag, &clm_irr_thresholdtype, &soi_z, &clm_next, &clm_write_logs, &clm_last_rst, &clm_daily_rst, &clm_nlevsoi, &clm_nlevlak);

void CLM_LSM(double *pressure_data, double *saturation_data, double *evap_trans_data, double *mask, double *porosity_data,
             double *dz_mult_data, int *istep, double *dt, double *t, double *start_time,
             double *dx, double *dy, double *dz, int *ix, int *iy, int *nx, int *ny, int *nz,
             int *nx_f, int *ny_f, int *nz_f, int *nz_rz, int *ip, int *p, int *q, int *r, int *gnx, int *gny, int *rank,
             double *sw_data, double *lw_data, double *prcp_data, double *tas_data, double *u_data, double *v_data, double *patm_data, double *qatm_data,
             double *lai_data, double *sai_data, double *z0m_data, double *displa_data,
             double *slope_x_data, double *slope_y_data,
             double *eflx_lh_tot_data, double *eflx_lwrad_out_data, double *eflx_sh_tot_data, double *eflx_soil_grnd_data, double *qflx_eval_tot_data,
             double *qflx_evap_grnd_data, double *qflx_evap_soi_data, double *qflx_evap_veg_data, double *qflx_tran_veg_data,
             double *qflx_infl_data, double *swe_out_data, double *t_grnd_data, double *t_soil_data, int *clm_dump_interval, int *clm_1d_out,
             int *clm_forc_veg, char *clm_file_dir, int *clm_file_dir_length, int *clm_bin_out_dir, int *write_CLM_binary, int *slope_accounting_CLM, int *clm_beta_function,
             int *clm_veg_function, double *clm_veg_wilting, double *clm_veg_fieldc, double *clm_res_sat,
             int *clm_irr_type, int *clm_irr_cycle, double *clm_irr_rate, double *clm_irr_start, double *clm_irr_stop,
             double *clm_irr_threshold, double *qirr, double *qirr_inst, double *iflag, int *clm_irr_thresholdtype, int *soi_z,
             int *clm_next, int *clm_write_logs, int *clm_last_rst, int *clm_daily_rst, int *clm_nlevsoi, int *clm_nlevlak);

/* @RMM CRUNCHFLOW.F90*/
//#define CRUNCHFLOW crunchflow_
//#define CALL_CRUNCHFLOW();

//    void CRUCHFLOW( );

END_EXTERN_C

#endif
