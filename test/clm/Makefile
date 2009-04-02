##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

default: check

TESTS := \

ifeq (${PARFLOW_HAVE_CLM},yes)
TESTS += clm.tcl
endif


check:
	@-for i in $(TESTS); \
	do \
	        make --no-print-directory testclean; \
		tclsh $$i; \
	done

clean: testclean

testclean:
	@rm -f global_nt.scr
	@rm -f clm_elog.txt.0
	@rm -f washita.para.out.dat.0
	@rm -f balance.txt.0
	@rm -f alma_washita.output.txt.0
	@rm -f washita.output.txt.0
	@rm -f qflx_infl/*.bin.0
	@rm -f clm.out.pftcl
	@rm -f qflx_top_soil/*.bin.0
	@rm -f swe_out/*.bin.0
	@rm -f qflx_evap_veg/*.bin.0
	@rm -f diag_out/*.dat.0
	@rm -f eflx_soil_grnd/*.bin.0
	@rm -f qflx_evap_soi/*.bin.0
	@rm -f qflx_tran_veg/*.bin.0
	@rm -f qflx_evap_tot/*.bin.0
	@rm -f t_grnd/*.bin.0
	@rm -f eflx_lwrad_out/*.bin.0
	@rm -f eflx_sh_tot/*.bin.0
	@rm -f eflx_lh_tot/*.bin.0
	@rm -f qflx_evap_grnd/*.bin.0
	@rm -f *.pfb*
	@rm -f *.silo*
	@rm -f *.pfsb*
	@rm -f *.log
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt
	@rm -fr default_richards.out

