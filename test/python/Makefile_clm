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
ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += clm.tcl \
         clm_forc_veg.tcl \
	 clm_varDZ.tcl
endif
endif

PARALLEL_TESTS =

ifeq (${AMPS},mpi1)
ifeq (${PARFLOW_HAVE_CLM},yes)
ifeq (${PARFLOW_HAVE_HYPRE},yes)
	PARALLEL_TESTS += \
		clm.tcl \
		clm.jac.tcl \
        clm_forc_veg.tcl \
		clm_varDZ.tcl
endif
endif
endif

SAMRAI_TESTS :=
ifeq (${PARFLOW_HAVE_CLM},yes)
ifeq (${PARFLOW_HAVE_SAMRAI},yes)
ifeq (${PARFLOW_HAVE_HYPRE},yes)
SAMRAI_TESTS += \
	clm_samrai.tcl
endif
endif
endif

check: check_samrai check_sequential check_parallel

check_sequential:
	@-for i in $(TESTS); \
	do \
	        make --no-print-directory testclean; \
		echo "*****************************************************************************"; \
		echo "Running $$i on 1 1 1"; \
		tclsh $$i 1 1 1; \
	done

check_parallel:
	@-for i in $(PARALLEL_TESTS); \
	do \
		for processor_topology in "1 2 1" "2 1 1" "2 2 1" "3 3 1" "1 4 1" "4 1 1"; \
		do \
		        make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done


check_samrai:
	@-for i in $(SAMRAI_TESTS); \
	do \
		for processor_topology in "1 1 1 1" "2 1 1 1"; \
		do \
		        make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done



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
	@rm -fr default_richards.out
	@rm -fr clm.out
	@rm -fr test.log.*
	@rm -fr drv_clmin.dat.*
	@rm -fr drv_vegm.dat.*
	@rm -fr core*
	@rm -fr clm_output.txt.*
	@rm -fr clm_varDZ_output.txt.*
	@rm -fr clm_varDZ.out
	@rm -f clm_varDZ.out.pftcl
	@rm -f clm_varDZ_elog.txt.*
	@rm -fr clm_forc_veg_output.txt.*
	@rm -fr clm_forc_veg.out
	@rm -f clm_forc_veg.out.pftcl
	@rm -f clm_forc_veg_elog.txt.*	
	@rm -fr clm-reuse-ts-* swe.out.csv
	@rm -f *.pfmetadata
