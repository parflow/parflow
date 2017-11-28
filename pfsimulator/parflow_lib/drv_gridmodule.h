struct clmgrid {
  /*use clm_varpar, only : 10*/

/*=== GRID SPACE User-Defined Parameters ==================================*/

/* Leaf constants*/

  double dewmx;

/* Roughness lengths*/

  double zlnd;
  double zsno;
  double csoilc;

/* Soil parameters*/

  double wtfact;
  double trsmx0;
  double scalez;
  double hkdepth;

/* Land surface parameters*/

  double latdeg;
  double londeg;

  double sand[10];
  double clay[10];

  double fgrd[4];
  int pveg[4];        /*treat them normal arrays and allocate with nt; nt was fixed hereto 4 ?*/

  int mask;

/*=== CLM Forcing parameters*/

  double forc_hgt_u;
  double forc_hgt_t;
  double forc_hgt_q;

/*=== Land Surface Fluxes*/

  double qflx_evap_tot;
  double eflx_sh_tot;
  double eflx_lh_tot;
  double eflx_lwrad_out;
  double t_ref2m;
  double t_rad;

/*=== CLM Vegetation parameters*/

  double rootfr;

/*=== CLM Soil parameters*/

  double smpmax;
  int isoicol;

/*=== Numerical finite-difference*/

  double capr;
  double cnfac;
  double smpmin;
  double ssi;
  double wim;
  double pondmx;

/*=== End Variable List ===================================================*/
};



