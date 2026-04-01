# ParFlow Development (parflow_claude)

## Project Context

This is a development clone of ParFlow used for implementing the Ippisch air-entry modification and saturation lookup table feature. The upstream ParFlow repository is at github.com/parflow/parflow.

## Build & Test

- **Build**: `cd build && make -j8 && make install`
- **PARFLOW_DIR**: `/Users/reed/parflow/parflow_claude`
- **PF_SRC**: `/Users/reed/parflow/parflow_claude`
- **Conda env**: `subsettools`
- **Run tests**: `python $PF_SRC/test/python/<test>.py` (NOT ctest)
- **Runtime libs**: libgfortran.5.dylib, libgcc_s.1.1.dylib, libquadmath.0.dylib must be in `$PARFLOW_DIR/lib/`

## Code Formatting

- **C**: `/Users/reed/parflow/libraries/uncrustify-uncrustify-0.79.0/build/uncrustify -c bin/parflow.cfg --replace --no-backup <file.c>`
- **Python**: `black <file.py>`

## Architecture Notes

### problem_phase_rel_perm.c

This file has TWO separate evaluation sections for the by-region (data_from_file == 0) path:
1. **Surface loop** (`GrGeomSurfLoop`) — Kr at cell faces adjacent to geometry surfaces
2. **Interior loop** (`GrGeomInLoop`, labeled "Compute rel. perms. on interior") — Kr for interior cells

Both have CALCFCN/CALCDER branching with table (spline/linear) and direct-eval sub-branches. Any constitutive relation modification MUST be applied to both sections.

### Correct output workflow

1. Update test script to use `pf_test_file` / `pf_test_file_with_abs` comparisons
2. Run test (comparison fails on missing files)
3. Copy output PFBs to `test/correct_output/`
4. Re-run to verify PASSED
