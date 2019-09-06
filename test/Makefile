##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

SUBDIRS = clm

default: check

TESTS := \
	default_single.tcl \
	default_richards_wells.tcl \
	forsyth2.tcl \
	harvey.flow.tcl \
	harvey_flow_pgs.tcl \
	crater2D.tcl \
	crater2D_vangtable_spline.tcl \
	crater2D_vangtable_linear.tcl \
	small_domain.tcl \
	richards_hydrostatic_equalibrium.tcl

ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += \
	default_richards.tcl \
	default_overland.tcl \
	LW_var_dz.tcl \
	LW_var_dz_spinup.tcl \
	terrain_following_grid_overland.tcl \
	var_dz_1D.tcl \
	pfmg.tcl \
	smg.tcl \
	pfmg_octree.tcl \
	overland_slopingslab_KWE.tcl \
	overland_tiltedV_KWE.tcl \
	overland_slopingslab_DWE.tcl \
	overland_tiltedV_DWE.tcl\
	overland_FlatICP.tcl \
	richards_FBx.tcl \
	richards_FBy.tcl  \
	richards_box_proctest.tcl
endif

ifeq (${PARFLOW_HAVE_SILO},yes)
TESTS += \
	indicator_field.tcl

ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += \
	water_balance_y.tcl \
	water_balance_x.tcl \
	default_richards_with_silo.tcl
endif
endif

ifeq (${PARFLOW_HAVE_NETCDF},yes)
ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += \
	default_richards_with_netcdf.tcl
endif
endif

SAMRAI_TESTS :=
SAMRAI_TESTS_WITH_PATCH_COUNT :=
ifeq (${PARFLOW_HAVE_SAMRAI},yes)

SAMRAI_TESTS += \
	samrai_compute_domain.tcl

SAMRAI_TESTS_WITH_PATCH_COUNT += \
	samrai.tcl

endif

PARALLEL_3DTOPO_TESTS :=
PARALLEL_2DTOPO_TESTS :=

ifeq (${AMPS},mpi1)

PARALLEL_3DTOPO_TESTS += \
	default_single.tcl

ifeq (${PARFLOW_HAVE_HYPRE},yes)

PARALLEL_3DTOPO_TESTS += \
	default_richards.tcl

PARALLEL_2DTOPO_TESTS += \
	default_overland.tcl \
	default_overland.pfmg.jac.tcl \
	default_overland.pfmg_octree.jac.tcl \
	LW_var_dz.tcl \
	LW_var_dz_spinup.tcl \
	richards_FBx.tcl \
	richards_FBy.tcl  \
	richards_box_proctest.tcl

ifeq (${PARFLOW_HAVE_SILO},yes)
TESTS += \
	water_balance_x.hardflow.nojac.tcl \
	water_balance_x.hardflow.jac.tcl
endif

endif

endif

CHECK_TARGETS=check_sequential check_parallel check_samrai

test: check

check:
	@(logfile=check.$$$$.checklog; \
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
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make check_sequential); \
	done

check_parallel:
	@-for i in $(PARALLEL_3DTOPO_TESTS); \
	do \
		for processor_topology in "1 1 2" "1 2 1" "2 1 1" "2 2 2" "3 3 3" "1 1 4" "1 4 1" "4 1 1"; \
		do \
		        make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done
	@-for i in $(PARALLEL_2DTOPO_TESTS); \
	do \
		for processor_topology in "1 2 1" "2 1 1" "1 4 1" "4 1 1"; \
		do \
		        make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make check_parallel); \
	done


check_samrai:
	 @-for i in $(SAMRAI_TESTS); \
	 do \
		for processor_topology in "2 2 1" "2 2 1" "2 2 1" "2 2 1" "2 2 1" "2 2 1" "2 2 1"; \
	 	do \
			make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done; \
		for processor_topology in "1 1 1" "1 2 1" "2 1 1" "2 2 1" "3 3 1" "1 4 1" "4 1 1"; \
	 	do \
			make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done
	@-for i in $(SAMRAI_TESTS_WITH_PATCH_COUNT); \
	do \
		for processor_topology in "1 1 1 1" "1 1 1 2" "1 1 1 3" "2 1 1 2" "1 2 1 2" "1 1 2 2" "2 1 1 3" "1 2 1 3" "1 1 2 3" "2 1 1 4" "1 2 1 4" "1 1 2 4" ; \
		do \
		        make --no-print-directory testclean; \
			echo "*****************************************************************************"; \
			echo "Running $$i on $$processor_topology"; \
			tclsh $$i $$processor_topology; \
		done \
	done
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make check_samrai); \
	done


clean: testclean
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make --no-print-directory clean); \
	done

testclean:
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
