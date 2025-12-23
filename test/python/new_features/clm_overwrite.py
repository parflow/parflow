import os
import sys

from parflow import Run
from parflow.tools.core import update_run_from_args
from parflow.tools.builders import CLMImporter
from parflow.tools.export import CLMExporter, NotOverwritableException
from parflow.tools.fs import cp, get_absolute_path, rm


default_driver_file_names = {
    "input": "drv_clmin.dat",
    "map": "drv_vegm.dat",
    "parameters": "drv_vegp.dat",
}

default_driver_file_paths = {
    "input": "../../input/drv_clmin.dat.old",
    "map": "../../tcl/clm/drv_vegm.dat",
    "parameters": "../../tcl/clm/drv_vegp.dat",
}


# ---------------------------------------------------------
# Testing overwriting driver files with CLMExporter
# ---------------------------------------------------------


def test_overwrite_failure(clm):
    # First, copy the files over
    copy_driver_files()

    # Import the files
    CLMImporter(clm).set_default_land_names().files()

    exporter = CLMExporter(clm)

    # These should all raise exceptions
    should_raise = [
        exporter.write,
        exporter.write_input,
        exporter.write_map,
        exporter.write_parameters,
    ]
    for func in should_raise:
        assert_exception(func, NotOverwritableException)

    # Cleanup
    remove_driver_files()


def test_overwrite_success(clm):
    # First, copy the files over
    copy_driver_files()

    # Import the files
    CLMImporter(clm).set_default_land_names().files()

    exporter = CLMExporter(clm)

    # Allow overwriting to occur
    clm_solver = clm.Solver.CLM
    clm_solver.OverwriteDrvClmin = True
    clm_solver.OverwriteDrvVegm = True
    clm_solver.OverwriteDrvVegp = True

    # These should all run
    should_run = [
        exporter.write,
        exporter.write_input,
        exporter.write_map,
        exporter.write_parameters,
    ]
    for func in should_run:
        func()

    # Turn these off. We can still overwrite since the files were generated.
    clm_solver.OverwriteDrvClmin = False
    clm_solver.OverwriteDrvVegm = False
    clm_solver.OverwriteDrvVegp = False

    for func in should_run:
        func()

    # Re-copy the driver files over
    copy_driver_files()

    # These should all raise exceptions now
    for func in should_run:
        assert_exception(func, NotOverwritableException)

    # Set the argparse argument to be True
    clm._process_args_.overwrite_clm_driver_files = True
    update_run_from_args(clm, clm._process_args_)

    # These should now be true
    assert clm_solver.OverwriteDrvClmin is True
    assert clm_solver.OverwriteDrvVegm is True
    assert clm_solver.OverwriteDrvVegp is True

    # And the functions should run again
    for func in should_run:
        func()

    # Turn these off to clean up.
    clm._process_args_.overwrite_clm_driver_files = False
    clm_solver.OverwriteDrvClmin = False
    clm_solver.OverwriteDrvVegm = False
    clm_solver.OverwriteDrvVegp = False

    # Cleanup
    remove_driver_files()


def test_write_allowed(clm):
    # First, copy the files over
    copy_driver_files()

    originally_written_times = driver_file_written_times()

    def files_unchanged():
        return originally_written_times == driver_file_written_times()

    # Import the files
    CLMImporter(clm).set_default_land_names().files()

    exporter = CLMExporter(clm)

    warning_func = "_print_not_written_warning"
    # This should not print any warnings even though it didn't write,
    # because we did not change any settings.
    assert was_called(exporter, warning_func, exporter.write_allowed) is False

    # Prove the files were not written to
    assert files_unchanged()

    # Change some settings. Warnings should be printed in every case.
    clm_solver = clm.Solver.CLM
    clm_solver.Input.Timing.StartYear = "2020"
    assert was_called(exporter, warning_func, exporter.write_allowed) is True

    # Prove the files were not written to
    assert files_unchanged()

    # Re-import the files to erase history
    CLMImporter(clm).set_default_land_names().files()

    clm_solver.Vegetation.Map.Sand.Type = "Constant"
    clm_solver.Vegetation.Map.Sand.Value = 0.3
    assert was_called(exporter, warning_func, exporter.write_allowed) is True

    # Prove the files were not written to
    assert files_unchanged()

    # Re-import the files to erase history
    CLMImporter(clm).set_default_land_names().files()

    clm_solver.Vegetation.Parameters.forest_en.WaterType = 5
    assert was_called(exporter, warning_func, exporter.write_allowed) is True

    # Prove the files were not written to
    assert files_unchanged()

    # Re-import the files to erase history
    CLMImporter(clm).set_default_land_names().files()
    # Should not be called this time
    assert was_called(exporter, warning_func, exporter.write_allowed) is False

    # Prove the files were not written to
    assert files_unchanged()

    # Cleanup
    remove_driver_files()


def assert_exception(func, *args, **kwargs):
    exc = args[-1]
    args = args[:-1]
    try:
        func(*args, **kwargs)
    except exc:
        return
    else:
        sys.exit(f"{exc} did not occur")


def was_called(obj, func_name, run_func, *args, **kwargs):
    was_called = False
    prev = getattr(obj, func_name)

    def called(*args, **kwargs):
        nonlocal was_called
        was_called = True
        prev(*args, **kwargs)

    setattr(obj, func_name, called)
    run_func(*args, **kwargs)
    setattr(obj, func_name, prev)
    return was_called


def copy_driver_files():
    for name, file_name in default_driver_file_names.items():
        path = default_driver_file_paths[name]
        cp(path, file_name)


def remove_driver_files():
    for file_name in default_driver_file_names.values():
        rm(file_name)


def driver_file_written_times():
    return {
        key: os.stat(get_absolute_path(x)).st_mtime
        for key, x in default_driver_file_names.items()
    }


if __name__ == "__main__":
    clm = Run("clm", __file__)

    # These are required to write out the map
    clm.ComputationalGrid.NX = 5
    clm.ComputationalGrid.NY = 5

    test_overwrite_failure(clm)
    test_overwrite_success(clm)
    test_write_allowed(clm)
