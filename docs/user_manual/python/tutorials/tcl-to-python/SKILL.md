---
name: parflow-tcl-to-python
description: Convert a ParFlow TCL runscript (pfset, pfrun, pfdist) into a Python PFTools Run script. Use when the user asks to convert a .tcl ParFlow input, migrate from TCL to Python, or translate pfset keys to run.Key assignments.
---

# Convert ParFlow TCL runscripts to Python

Produce a complete, runnable Python PFTools script from a ParFlow TCL runscript. Preserve comments and section headers. Do not invent keys. Prefer a working user runscript over a line-for-line clone of ParFlow's internal test harness.

## Output skeleton

```python
from parflow import Run
from parflow.tools.fs import mkdir, cp, get_absolute_path  # only if the TCL copies/makes dirs

run_name = "<tcl_stem_or_runname>"
<run> = Run(run_name, __file__)

# converted keys ...

<run>.run()
```

Use the TCL file stem as the Python identifier and `Run` name, unless the TCL sets `runname` / `pfrun $runname` — then use that name.

Replace this TCL preamble:

```tcl
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*
```

with the `from parflow import Run` / `Run(...)` block above. Do not emit those TCL lines as comments unless the user asked to keep them.

## `pfset` → Python assignment

```
pfset Key.Path.Token   value
```

becomes

```
<run>.Key.Path.Token = <python_value>
```

### Values

| TCL value | Python value |
| --- | --- |
| number (`18`, `1.0`, `1e-9`) | number (`18`, `1.0`, `1e-9`) |
| unquoted word (`Richards`, `Box`) | string (`"Richards"`, `"Box"`) |
| quoted string (`"left right"`) | string (`"left right"`) |
| `True` / `False` | boolean `True` / `False` (not strings) |
| `$varname` | Python name `varname` (define the variable first) |
| `[lindex $argv N]` | `argparse` `-p`/`-q`/`-r` with default `1`, or literal `1` if topology is not important |

Join TCL line continuations (`\`) into one Python string or parenthesized assignment. Do not leave a backslash-continued string that is invalid Python.

### Tokens that are not valid Python identifiers

Do **not** rewrite hyphens to underscores. Hyphens in patch or geometry names must stay identical to the TCL database tokens (and to `Geom.*.Patches` / `BCPressure.PatchNames`).

Use bracket notation for any token that is not a valid Python identifier (hyphens, leading digits):

```python
<run>.Patch["x-lower"].BCPressure.Type = "FluxConst"
<run>.Patch["z-upper"].BCPressure.Type = "OverlandFlow"
```

Integer tokens (for example `Cell.0.dzScale.Value` or `Patch.z-upper.BCPressure.0.Value`) use a leading underscore, which is the PFTools prefix:

```python
<run>.Cell._0.dzScale.Value = 1.0
<run>.Patch["z-upper"].BCPressure._0.Value = rec_flux
```

User-defined names listed in `GeomInput.Names`, `Cycle.Names`, `Phase.Names`, `Wells.Names`, and similar keys must be assigned **before** they are used as tokens later in the script.

## TCL ParFlow commands

| TCL | Python |
| --- | --- |
| `pfrun $runname` / `pfrun name` | `<run>.run()` |
| `pfwritedb $runname` | `<run>.write()` (optional; `run()` already writes the database) |
| `pfdist file.pfb` | `<run>.dist("file.pfb")` |
| `pfdist -nz N file.pfb` | `<run>.dist("file.pfb", NZ=N)` |
| `pfundist ...` | omit for a typical user script, or `<run>.undist()` if the user needs it |
| `pfget Key.Path` | `<run>.Key.Path` |
| `file mkdir dir` | `mkdir("dir")` from `parflow.tools.fs` |
| `file copy -force src dest` | `cp("src", "dest")` from `parflow.tools.fs` |
| `cd dir` | `chdir("dir")` from `parflow.tools.fs` |
| `puts "..."` | `print("...")` |
| `set name value` | `name = value` (same value rules as `pfset`) |
| `expr {..}` | Python arithmetic |
| `format` / `foreach` | f-strings / `for` loops |

`$env(PARFLOW_DIR)` and similar environment paths can stay as `"$PARFLOW_DIR/..."` strings passed to `parflow.tools.fs` helpers, which expand them.

Place `dist()` calls after the `Process.Topology` keys are set, and before `<run>.run()`.

## What to skip

Unless the user asks to keep the test harness:

- `source pftest.tcl` and `pftestFile` / `pftestFileWithAbs` blocks
- `correct_output` symlinks and PASS/FAIL printing used only in ParFlow's test suite

End a user runscript with `<run>.run()`.

Leave unused or solver-incompatible keys as they appear in the TCL (do not delete them unless they are clearly test-only). If a key is invalid Python after conversion, fix the syntax; do not drop the key.

## Side-by-side key reference

TCL `pfset` and Python `run.Key =` forms for every ParFlow key are documented together in the ParFlow User Manual chapter **ParFlow Input Keys**. Use that chapter when a token or value is ambiguous.

## Minimal example

TCL:

```tcl
lappend auto_path $env(PARFLOW_DIR)/bin
package require parflow
namespace import Parflow::*

pfset FileVersion 4
pfset Process.Topology.P 1
pfset GeomInput.Names "domain_input"
pfset GeomInput.domain_input.InputType Box
pfset Geom.domain.Patches "x-lower x-upper z-upper"
pfset Patch.z-upper.BCPressure.Type OverlandFlow
pfdist -nz 1 slopes.pfb
pfrun default_richards
```

Python:

```python
from parflow import Run

default_richards = Run("default_richards", __file__)

default_richards.FileVersion = 4
default_richards.Process.Topology.P = 1
default_richards.GeomInput.Names = "domain_input"
default_richards.GeomInput.domain_input.InputType = "Box"
default_richards.Geom.domain.Patches = "x-lower x-upper z-upper"
default_richards.Patch["z-upper"].BCPressure.Type = "OverlandFlow"
default_richards.dist("slopes.pfb", NZ=1)
default_richards.run()
```

## Checklist before finishing

- [ ] Script instantiates `Run(name, __file__)` and ends with `.run()`
- [ ] Every `pfset` became an assignment; values have the correct Python type
- [ ] Hyphenated tokens use `["token"]`; integer tokens use `_N`
- [ ] `Names` keys appear before those names are used as tokens
- [ ] `pfdist` became `.dist()`; file copies use `parflow.tools.fs`
- [ ] Test-harness TCL is omitted unless requested
