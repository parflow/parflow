"""
Timing of the pystencils kernel build.

    pystencils_timing.py <log file> codegen=<OUTPUT> <command...>   (RULE_LAUNCH_CUSTOM)
    pystencils_timing.py <log file> compile=<SOURCE> <command...>   (RULE_LAUNCH_COMPILE)
        Run the command, print its wall time and append it to the log file.
        The kernel name is taken from the given output/source file.

    pystencils_timing.py <log file> summary [<kernel...>]
        Print the latest codegen and compile time of each kernel (default: all in the log)
        and their totals.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

STEPS = ("codegen", "compile")


def run(log_file, step, path, cmd):
    # custom commands without outputs (e.g. the POST_BUILD summary) are not timed
    if not path:
        return subprocess.run(cmd).returncode

    kernel = os.path.splitext(os.path.basename(path))[0]

    start = time.perf_counter()
    returncode = subprocess.run(cmd).returncode
    elapsed = time.perf_counter() - start

    status = "" if returncode == 0 else f" (failed with exit code {returncode})"
    print(f"-- pystencils {step}: {kernel} took {elapsed:.2f} s{status}", flush=True)

    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{step}\t{kernel}\t{elapsed:.3f}\t{returncode}\n")

    return returncode


def summary(log_file, kernels):
    # the log accumulates over (incremental) builds, so use the latest successful time per kernel
    latest = {}
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                _, step, kernel, seconds, returncode = line.rstrip("\n").split("\t")
                if returncode == "0":
                    latest[step, kernel] = float(seconds)

    kernels = kernels or list(dict.fromkeys(kernel for _, kernel in latest))

    def row(name, times):
        return f"   {name:<{width}}" + "".join(f"{'-' if t is None else f'{t:.2f}':>10}" for t in times)

    width = max(len(name) for name in kernels + ["kernel"])
    rows = [[latest.get((step, kernel)) for step in STEPS] for kernel in kernels]
    totals = [sum(t for t in col if t is not None) for col in zip(*rows)]

    print("-- pystencils kernel timings in s (latest build of each kernel):")
    print(f"   {'kernel':<{width}}" + "".join(f"{step:>10}" for step in STEPS))
    for kernel, times in zip(kernels, rows):
        print(row(kernel, times))
    print(row("total", totals), flush=True)

    return 0


if __name__ == "__main__":
    log_file, mode, *args = sys.argv[1:]

    if mode == "summary":
        sys.exit(summary(log_file, args))

    step, _, path = mode.partition("=")
    sys.exit(run(log_file, step, path, args))
