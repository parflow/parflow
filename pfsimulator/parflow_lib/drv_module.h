struct drvdec {
/*=== Driver User-Defined Parameters ======================================*/
  double min;
  double udef;

  char vegtf[40];
  char vegpf[40];
  char poutf1d[40];
  char metf1d[40];
  char outf1d[40];
  char rstf[40];

/*=== CLM Parameters ======================================================*/
  int nch;

/*=== Driver Parameters ====================================================*/
  int nc;
  int nr;
  int nt;
  int startcode;
  int sss;
  int sdoy;
  int smn;
  int shr;
  int sda;
  int smo;
  int syr;
  int ess;
  int emn;
  int edoy;
  int ehr;
  int eda;
  int emo;
  int eyr;
  int ts;
  double writeintc;
  int maxt;

/*=== Timing Variables ==========*/
  double time;
  double etime;
  int pda;
  int doy, yr, mo, da, hr, mn, ss;
  int endtime;
  double day, gmt, eday, egmt, sgmt;

/*=== Arguments ==========================================================*/
  double ctime;
  int cyr, cmo, cda;
  int chr, cmn, css;

/*=== Initial CLM conditions =============================================*/
  double t_ini;
  double h2osno_ini;
  double sw_ini;

/*=== CLM diagnostic parameters ==========================================*/
  int surfind;
  int soilind;
  int snowind;

  int vclass;
  int clm_ic;

/*@== Overland routing parameters*/
  int rout, cout;
  double sovout;
  double dt;
  double max_sl;

/*@== CLM.PF variables*/
  int sat_flag;
  double dx, dy, dz;

/*=== End Variable List ===================================================*/
};
