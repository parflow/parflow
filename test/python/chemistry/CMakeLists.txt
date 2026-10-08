# -----------------------------------------------------------------------------
# Alquimia reactive-transport verification tests, Python Run API (react_trans).
#
# Python translations of the TCL cases in test/chemistry/: a non-reactive
# tracer and calcite dissolution, reproducing the published profiles of
# Molins et al. 2025 (GMD 18:3241, Fig. 2). They reuse the engine input decks,
# thermodynamic databases, and reference outputs under test/chemistry/. Added
# only when Alquimia is enabled (parent gate) and require a CrunchFlow-backed
# Alquimia install at runtime.
# -----------------------------------------------------------------------------
include(ParflowTest)

set(CHEMISTRY_TESTS tracer_crunch calcite_crunch)

foreach(inputfile ${CHEMISTRY_TESTS})
    pf_add_py_test(${inputfile})
endforeach()
