##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

clean: testclean

testclean:
	@rm -f global_nt.scr
	@rm -f clm_elog.txt.*
	@rm -f washita.para.out.dat.*
	@rm -f balance.txt.*
	@rm -f alma_washita.output.txt.*
	@rm -f washita.output.txt.*
	@rm -fr qflx_infl
	@rm -f clm.out.pftcl
	@rm -fr qflx_top_soil
	@rm -fr swe_out
	@rm -fr qflx_evap_veg
	@rm -fr diag_out
	@rm -fr eflx_soil_grnd
	@rm -fr qflx_evap_soi
	@rm -fr qflx_tran_veg
	@rm -fr qflx_evap_tot
	@rm -fr t_grnd
	@rm -fr eflx_lwrad_out
	@rm -fr eflx_sh_tot
	@rm -fr eflx_lh_tot
	@rm -fr qflx_evap_grnd
	@rm -f *.pfb*
	@rm -f *.silo*
	@rm -f *.pfsb*
	@rm -f *.log
	@rm -f *.out.timing.csv	
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt
	@rm -fr clm.out
	@rm -fr test.log.*
	@rm -fr drv_clmin.dat.*
	@rm -f drv_clmin.dat drv_vegm.dat drv_vegp.dat displa.dat 
	@rm -fr core*
	@rm -fr clm_output.txt.*
	@rm -fr clm_varDZ_output.txt.*
	@rm -fr clm_varDZ.out
	@rm -f clm_varDZ.out.pftcl
	@rm -f clm_varDZ_elog.txt.*	
	@rm -fr clm_slope_output.txt.*
	@rm -fr clm_slope.out
	@rm -f clm_slope.out.pftcl
	@rm -f clm_slope_elog.txt.*
	@rm -fr clm_forc_veg_output.txt.*
	@rm -fr clm_forc_veg.out
	@rm -f clm_forc_veg.out.pftcl
	@rm -f clm_forc_veg_elog.txt.*	
	@rm -fr clm-reuse-ts-* swe.out.csv
	@rm -f *.pfmetadata
	@rm -f z0m.dat sai.dat lai.dat narr_1hr.sc3.txt.0
	@rm -fr test_output
