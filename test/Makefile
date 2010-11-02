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
	test_X.tcl \
	default_single.tcl \
	uvm_flow.tcl \
	default_richards.tcl \
	default_richards_wells.tcl \
	forsyth2.tcl \
	harvey.flow.tcl \
	harvey_flow_pgs.tcl \
	default_overland.tcl \
	crater2D.tcl \
	crater2D_vangtable_spline.tcl \
	crater2D_vangtable_linear.tcl \
	small_domain.tcl \
	richards_hydrostatic_equalibrium.tcl

ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += \
	pfmg.tcl \
	smg.tcl \
	pfmg_octree.tcl
endif

ifeq (${PARFLOW_HAVE_SILO},yes)
TESTS += \
	default_richards_with_silo.tcl \
	indicator_field.tcl \
	water_balance_y.tcl \
	water_balance_x.tcl 
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
	uvm_flow.tcl \
	default_single.tcl \
	default_richards.tcl \
	default_richards_wells.tcl 

PARALLEL_2DTOPO_TESTS += \
	default_overland.tcl \
	default_overland.pfmg.jac.tcl \
	default_overland.pfmg_octree.jac.tcl \
	water_balance_x.hardflow.nojac.tcl \
	water_balance_x.hardflow.jac.tcl 
endif

check: check_sequential check_parallel check_samrai 

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
			echo "time tclsh $$i $$processor_topology"; \
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
