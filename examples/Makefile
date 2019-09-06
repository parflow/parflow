##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

default: check

TESTS := bc.00.tcl cond.tcl default_richards.tcl default_single.tcl forsyth2.tcl forsyth5.tcl impes.internalbc.tcl impes.plinear.tcl impes.simple.tcl llnl.tcl llnl-perm.tcl richards.plinear.tcl test_X3Y2PlusSinXYPlus1.tcl test_XPlusYPlusZ.tcl test_X.tcl test_XYZTPlus1.tcl well_00.tcl well_01.tcl

CHECK_TARGETS=check_sequential 

test: check

check:
	@(logfile=check.$$$$.checklog; \
        export PF_TEST="yes";\
        make $(CHECK_TARGETS) | tee $$logfile; \
        num_run=`grep Running $$logfile | wc -l`; \
        num_passed=`grep PASSED $$logfile | wc -l`; \
        num_ctest_run=0; \
        for i in $$(grep "tests failed out of" $$logfile | awk '{print $$9}'); \
        do \
        num_ctest_run=$$(( $$num_ctest_run + $$i)); \
        done; \
        num_ctest_failed=0; \
        for i in $$(grep "tests failed out of" $$logfile | awk '{print $$4}'); \
        do \
        num_ctest_failed=$$(( $$num_ctest_failed + $$i)); \
        done; \
        num_run=$$(( $$num_run + $$num_ctest_run )); \
        num_passed=$$(( $$num_passed + $$num_ctest_run - $$num_ctest_failed )); \
        echo; echo; \
        echo "##############################################################################"; \
        echo ; \
        echo "     $$((num_passed*100/num_run))% tests passed, $$((num_run-num_passed)) tests failed out of $$num_run"; \
        echo; \
        if [ $$num_run -eq 0 ]; \
        then \
        echo "     FAILED : no tests run"; \
        else \
        if [ $$num_passed -eq $$num_run ]; \
        then \
        echo "     PASSED All"; \
        else \
        echo "     FAILED Tests"; \
        fi; \
        fi; \
        echo; \
        echo "##############################################################################"; \
	rm $$logfile; \
	[ $$num_run -eq 0 -o $$num_passed -eq $$num_run ] \
        )


check_sequential:
	@-for i in $(TESTS); \
	do \
	        make --no-print-directory testclean; \
		echo "*****************************************************************************"; \
		echo "Running $$i"; \
		tclsh $$i 1 1 1; \
	done



clean: testclean

testclean:
	@rm -f *.pfb*
	@rm -f *.silo*
	@rm -f *.pfsb*
	@rm -f *.log
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt
	@rm -f *.out.timing.csv
	@rm -fr default_richards.out 
	@rm -fr *.out.wells
	@rm -fr indicator_field.out
	@rm -fr water_balance.out
	@rm -fr default_overland.out
	@rm -fr test.log.*
	@rm -fr richards_hydrostatic_equalibrium.out
	@rm -fr core.*
	@rm -fr samrai_grid.tmp.tcl
	@rm -f samrai_grid2D.tmp.tcl
	@rm -fr LW_var_dz_spinup.out
	@rm -fr default_single.out
	@rm -f *.nc
	@rm -f *.pfmetadata
