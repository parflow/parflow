#define oas_pfl_define oas_pfl_define_
#define CALL_oas_pfl_define(nx, ny, dx, dy, ix, iy, sw_lon, sw_lat, nlon, nlat, pfl_step, pfl_stop) \
        oas_pfl_define(&nx, &ny, &dx, &dy, &ix, &iy, &sw_lon, &sw_lat, &nlon, &nlat, &pfl_step, &pfl_stop)
void oas_pfl_define(int *nx, int *ny,
                    double *dx, double *dy,
                    int *ix, int *iy,
                    double *sw_lon, double *sw_lat,
                    int *nlon, int *nlat,
                    double *pfl_step, double *pfl_stop);

#define oas_pfl_snd oas_pfl_snd_
#define CALL_oas_pfl_snd(kid, kstep, kdata, nx, ny, kinfo, kindex) \
        oas_pfl_snd(&kid, &kstep, &kdata, &nx, &ny, &kinfo, &kindex)
void oas_pfl_snd(int *kid, int *kstep, double *kdata,
                 int *nx, int *ny, int *kinfo, int *kindex);

#define oas_pfl_rcv oas_pfl_rcv_
#define CALL_oas_pfl_rcv(kid, kstep, kdata, nx, ny, kinfo) \
        oas_pfl_rcv(&kid, &kstep, &kdata, &nx, &ny, &kinfo)
void oas_pfl_rcv(int *kid, int *kstep, double *kdata,
                 int *nx, int *ny, int *kinfo);


#define send_fld2_clm send_fld2_clm_
#define CALL_send_fld2_clm(pressure, saturation, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep, porosity, dz) \
        send_fld2_clm(pressure, saturation, topo, &ix, &iy, &nx, &ny, &nz, &nx_f, &ny_f, &pstep, porosity, dz)
void send_fld2_clm(double *pressure, double *saturation, double *topo, int *ix, int*iy,
                   int *nx, int *ny, int *nz, int *nx_f, int *ny_f, double *pstep, double *porosity, double *dz);

#define receive_fld2_clm receive_fld2_clm_
#define CALL_receive_fld2_clm(evap_trans, topo, ix, iy, nx, ny, nz, nx_f, ny_f, pstep) \
        receive_fld2_clm(evap_trans, topo, &ix, &iy, &nx, &ny, &nz, &nx_f, &ny_f, &pstep)
void receive_fld2_clm(double *evap_trans, double *topo, int *ix, int*iy,
                      int *nx, int *ny, int *nz, int *nx_f, int *ny_f, double *pstep);
