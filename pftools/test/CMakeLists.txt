
#-----------------------------------------------------------------------------
# Mask utilities tests
#-----------------------------------------------------------------------------
foreach(testname mask-test-1 mask-test-2 mask-test-3 mask-test-4 mask-test-5 mask-test-6 mask-test-7 mask-test-8 mask-test-9 mask-test-10 mask-test-11 mask-test-12)
  pf_add_mask_test(${testname} "${testname}.asc" TEST_ARGS "--bottom-patch-label" "2" "--side-patch-label" "3")
endforeach()

pf_add_mask_test("depth-test-1" "depth-test-1.asc" TEST_ARGS "--bottom-patch-label" "2" "--side-patch-label" "3" "--z-top" "1000.0" "--z-bottom" "0.0")
pf_add_mask_test("depth-test-2" "depth-test-2.asc" TEST_ARGS "--bottom-patch-label" "2" "--side-patch-label" "3" "--z-top" "1000.0" "--z-bottom" "100.0")

foreach(testname multiple-mask-test-1)
  pf_add_multi_mask_test(${testname} "${testname}.asc")
endforeach()

