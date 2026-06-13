# -----------------------------------------------------------------------------
# Alquimia reactive-transport verification tests (react_trans).
#
# 1D CrunchFlow cases that reproduce the published profiles of
# Molins et al. 2025 (GMD 18:3241, Fig. 2): a non-reactive tracer and
# calcite dissolution. Reference outputs in correct_output/ were generated
# by this port and match the paper. These run only when Alquimia is enabled
# and require a CrunchFlow-backed Alquimia install at runtime.
# -----------------------------------------------------------------------------
include(ParflowTest)

set(CHEMISTRY_TESTS tracer_crunch.tcl calcite_crunch.tcl)

foreach(inputfile ${CHEMISTRY_TESTS})
    pf_add_sequential_test(${inputfile})
endforeach()
