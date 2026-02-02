# Richards Solver Code Flow

This document describes the code progression and control flow for ParFlow's Richards equation solver, the primary solver for variably saturated groundwater flow with optional land surface model (CLM) coupling.  This is the primary solver used for watershed type applications from small to large scale (e.g. EU-CORDEX, CONUS, CONCN domains).  

## Overview

The Richards solver is implemented in [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c) (~7000 lines) and orchestrates:
- Grid and data structure initialization
- Time-stepping with adaptive timestep control
- CLM land surface model coupling (optional)
- Nonlinear solve via Newton-Krylov (KINSOL)
- Multi-format output (PFB, NetCDF, SILO)

## Entry Point

The solver is invoked from the top-level `Solve()` function in [`solver.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver.c):

```c
// solver.c
void Solve() {
    solver = NewSolver();  // Creates Richards/IMPES/Diffusion based on "Solver" key
    PFModuleInvokeType(SolverInvoke, solver, ());
}
```

For Richards problems, this calls `SolverRichards()`.

---

## High-Level Control Flow

```
SolverRichards()
│
├── SetupRichards()
│   ├── Initialize metadata/provenance
│   ├── SetProblemData (geometry, properties)
│   ├── ComputeTop/ComputeBottom (active region)
│   ├── Compute terrain-following slopes
│   ├── Print subsurface data (optional)
│   └── Allocate logging structures
│
├── AdvanceRichards()
│   └── do-while(take_more_time_steps)
│       │
│       ├── SelectTimeStep (outer timestep)
│       ├── Read CLM forcing data
│       ├── CALL_CLM_LSM (land surface)
│       │
│       ├── do-while(!converged)          [inner loop]
│       │   ├── Adjust dt to forcing boundary
│       │   ├── t += dt
│       │   └── NonlinSolverInvoke (KINSOL)
│       │
│       ├── Post-convergence corrections
│       ├── Update density, saturation
│       ├── Accumulate water balance
│       └── Write output files
│
└── TeardownRichards()
    ├── Finalize metadata
    ├── Free vectors and grids
    └── Print statistics
```

---

## Problem Module Architecture

ParFlow uses a modular architecture where each physical property (porosity, permeability, saturation, etc.) is implemented as a **PFModule**. Each module has three key functions:

| Function | Purpose | When Called |
|----------|---------|-------------|
| `*NewPublicXtra()` | Parse keys from input database | Once at startup |
| `*InitInstanceXtra()` | Create grids, allocate vectors | Once per grid |
| Main function (e.g., `Porosity()`) | Evaluate and populate data | During `SetProblemData()` |

### Key Parsing Pattern

All modules follow this pattern in `*NewPublicXtra()`:

```c
// 1. Get the type selector
switch_name = GetString("Geom.Porosity.Type");
sim_type = NA_NameToIndexExitOnError(switch_na, switch_name, key);

// 2. Get geometry names to apply this property
geom_names = GetString("Geom.Porosity.GeomNames");
geo_names_na = NA_NewNameArray(geom_names);

// 3. For each geometry, read parameters
for (i = 0; i < num_regions; i++) {
    region = NA_IndexToName(geo_names_na, i);
    sprintf(key, "Geom.%s.Porosity.Value", region);
    values[i] = GetDouble(key);
}
```

### Input Database Functions

**Location:** [`input_database.h`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/input_database.h)

| Function | Purpose | Example |
|----------|---------|---------|
| `GetString(key)` | Read string value | `GetString("Phase.Names")` |
| `GetDouble(key)` | Read floating-point | `GetDouble("Geom.domain.Porosity.Value")` |
| `GetInt(key)` | Read integer | `GetInt("TimingInfo.StartCount")` |
| `GetStringDefault(key, default)` | String with fallback | `GetStringDefault("Solver.CLM.MetPath", ".")` |
| `GetDoubleDefault(key, default)` | Double with fallback | `GetDoubleDefault("Solver.CLM.SnowTCrit", 2.5)` |
| `GetIntDefault(key, default)` | Integer with fallback | `GetIntDefault("Solver.PrintSubsurfData", 0)` |

---

## Problem Initialization: NewProblem()

**Location:** [`problem.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem.c), `NewProblem()`

Creates the master `Problem` structure containing all PFModules for physics.

### Initialization Sequence

```c
Problem *NewProblem(int solver) {
    // 1. Check file version
    version_number = GetInt("FileVersion");

    // 2. Initialize geometry modules
    ProblemGeometries(problem) = PFModuleNewModule(Geometries, ());
    ProblemDomain(problem) = PFModuleNewModule(Domain, ());

    // 3. Parse timing info
    ProblemStartTime(problem) = GetDouble("TimingInfo.StartTime");
    ProblemStopTime(problem) = GetDouble("TimingInfo.StopTime");
    ProblemDumpInterval(problem) = GetDouble("TimingInfo.DumpInterval");

    // 4. Parse phase info
    phases = GetString("Phase.Names");
    GlobalsPhaseNames = NA_NewNameArray(phases);

    // 5. Create physics modules
    ProblemGravity(problem) = GetDouble("Gravity");
    ProblemPhaseDensity(problem) = PFModuleNewModule(PhaseDensity, (num_phases));
    ProblemPermeability(problem) = PFModuleNewModule(Permeability, ());
    ProblemPorosity(problem) = PFModuleNewModule(Porosity, ());
    ProblemPhaseRelPerm(problem) = PFModuleNewModule(PhaseRelPerm, ());
    ProblemSaturation(problem) = PFModuleNewModule(Saturation, ());
    // ... more modules

    // 6. Create boundary condition modules
    ProblemBCPressure(problem) = PFModuleNewModule(BCPressure, ());
    ProblemICPhasePressure(problem) = PFModuleNewModule(ICPhasePressure, ());

    return problem;
}
```

### Problem Modules Reference

| Module | Source File | Keys Parsed |
|--------|-------------|-------------|
| `Geometries` | [`problem_geometries.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_geometries.c) | [`GeomInput.*`](https://parflow.readthedocs.io/en/latest/keys.html#geometries) |
| `Domain` | [`problem_domain.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_domain.c) | [`Domain.GeomName`](https://parflow.readthedocs.io/en/latest/keys.html#domain) |
| `Permeability` | [`permeability.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/permeability.c) | [`Geom.*.Perm.*`](https://parflow.readthedocs.io/en/latest/keys.html#permeability) |
| `Porosity` | [`problem_porosity.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_porosity.c) | [`Geom.*.Porosity.*`](https://parflow.readthedocs.io/en/latest/keys.html#porosity) |
| `Saturation` | [`problem_saturation.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_saturation.c) | [`Geom.*.Saturation.*`](https://parflow.readthedocs.io/en/latest/keys.html#saturation) |
| `PhaseRelPerm` | [`problem_phase_rel_perm.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_phase_rel_perm.c) | [`Geom.*.RelPerm.*`](https://parflow.readthedocs.io/en/latest/keys.html#relative-permeability) |
| `SpecStorage` | [`problem_spec_storage.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_spec_storage.c) | [`SpecificStorage.*`](https://parflow.readthedocs.io/en/latest/keys.html#specific-storage) |
| `BCPressure` | [`problem_bc_pressure.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_bc_pressure.c) | [`Patch.*.BCPressure.*`](https://parflow.readthedocs.io/en/latest/keys.html#boundary-conditions-pressure) |
| `ICPhasePressure` | [`problem_ic_phase_pressure.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_ic_phase_pressure.c) | [`ICPressure.*`](https://parflow.readthedocs.io/en/latest/keys.html#initial-conditions-pressure) |
| `XSlope` | [`problem_toposlope_x.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_toposlope_x.c) | [`TopoSlopesX.*`](https://parflow.readthedocs.io/en/latest/keys.html#topographic-slopes) |
| `YSlope` | [`problem_toposlope_y.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_toposlope_y.c) | [`TopoSlopesY.*`](https://parflow.readthedocs.io/en/latest/keys.html#topographic-slopes) |
| `Mannings` | [`problem_mannings.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_mannings.c) | [`Mannings.*`](https://parflow.readthedocs.io/en/latest/keys.html#manning-s-roughness-coefficients) |

---

## SetProblemData: Evaluating Property Fields

**Location:** [`set_problem_data.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/set_problem_data.c), `SetProblemData()`

After modules are created, `SetProblemData()` **evaluates** each module to populate the actual data vectors.

### Evaluation Order

The order matters because some modules depend on others:

```c
void SetProblemData(ProblemData *problem_data) {
    // 1. Geometry first (defines solid regions)
    PFModuleInvoke(Geometries, geometries, (problem_data));
    PFModuleInvoke(Domain, domain, (problem_data));

    // 2. Subsurface properties (need geometry)
    PFModuleInvoke(Permeability, permeability,
        (problem_data,
         ProblemDataPermeabilityX(problem_data),  // Output vectors
         ProblemDataPermeabilityY(problem_data),
         ProblemDataPermeabilityZ(problem_data),
         ProblemDataNumSolids(problem_data),
         ProblemDataSolids(problem_data),
         ProblemDataGrSolids(problem_data)));

    PFModuleInvoke(Porosity, porosity,
        (problem_data,
         ProblemDataPorosity(problem_data),
         ...));

    PFModuleInvoke(SpecStorage, specific_storage,
        (problem_data,
         ProblemDataSpecificStorage(problem_data)));

    // 3. Surface properties (need domain defined)
    PFModuleInvoke(XSlope, x_slope,
        (problem_data,
         ProblemDataTSlopeX(problem_data),
         ProblemDataPorosity(problem_data)));  // Uses porosity for masking

    PFModuleInvoke(YSlope, y_slope, ...);
    PFModuleInvoke(Mannings, mann, ...);

    // 4. Variable dz (TFG)
    PFModuleInvoke(dzScale, dz_mult,
        (problem_data,
         ProblemDataZmult(problem_data)));

    // 5. Flow barriers
    PFModuleInvoke(FBx, FBx, ...);
    PFModuleInvoke(FBy, FBy, ...);
    PFModuleInvoke(FBz, FBz, ...);

    // 6. Boundary conditions and wells
    PFModuleInvoke(BCPressure, bc_pressure, ...);
    PFModuleInvoke(Wells, wells, ...);
}
```

### Example: How Porosity is Evaluated

**Location:** [`problem_porosity.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_porosity.c), `Porosity()`

```c
void Porosity(ProblemData *problem_data, Vector *porosity, ...) {
    // 1. Initialize all cells to default value
    InitVectorAll(porosity, 1.0);

    // 2. Loop over geometry regions from config
    for (i = 0; i < num_geo_indexes; i++) {
        j = geo_indexes[i];
        // 3. Call the field simulator for this region
        PFModuleInvoke(PorosityFieldSimulators[i],
            (geounits[j], gr_geounits[j], porosity));
    }

    // 4. Special handling: reset porosity to 1.0 in wells
    for (well = 0; well < num_wells; well++) {
        // Wells are fully connected (porosity = 1)
        ForEachWellCell(...) {
            porosity[cell] = 1.0;
        }
    }
}
```

### Field Simulator Types

Each property module supports multiple input types:

| Type | Description | Keys |
|------|-------------|------|
| `Constant` | Uniform value per region | `Geom.*.Porosity.Value` |
| `PFBFile` | Read from ParFlow binary | `Geom.*.Porosity.FileName` |
| `NCFile` | Read from NetCDF | `Geom.*.Porosity.FileName` |

**Example configuration:**
```python
# Constant porosity for domain
run.Geom.Porosity.GeomNames = 'domain'
run.Geom.domain.Porosity.Type = 'Constant'
run.Geom.domain.Porosity.Value = 0.4

# Or from file
run.Geom.domain.Porosity.Type = 'PFBFile'
run.Geom.domain.Porosity.FileName = 'porosity.pfb'
```

---

## Saturation and Relative Permeability: Constitutive Relations

**Location:** [`problem_saturation.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_saturation.c), [`problem_phase_rel_perm.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_phase_rel_perm.c)

These modules are special because they're **called during each nonlinear iteration** (not just at setup).

### Van Genuchten Parameters

See [Saturation keys](https://parflow.readthedocs.io/en/latest/keys.html#saturation) and [Relative Permeability keys](https://parflow.readthedocs.io/en/latest/keys.html#relative-permeability) in the ParFlow manual.

```python
# Type selection
run.Phase.Saturation.Type = 'VanGenuchten'
run.Phase.Saturation.GeomNames = 'domain'

# Parameters per geometry
run.Geom.domain.Saturation.Alpha = 3.5    # [1/m]
run.Geom.domain.Saturation.N = 2.0        # [-]
run.Geom.domain.Saturation.SRes = 0.1     # [-] residual
run.Geom.domain.Saturation.SSat = 1.0     # [-] saturated

# Relative permeability (usually same parameters)
run.Phase.RelPerm.Type = 'VanGenuchten'
run.Phase.RelPerm.GeomNames = 'domain'
run.Geom.domain.RelPerm.Alpha = 3.5
run.Geom.domain.RelPerm.N = 2.0
```

### Spatially-Variable Parameters

For heterogeneous domains, parameters can be read from files:

```python
run.Phase.Saturation.VanGenuchten.File = 1  # Enable file input
run.Geom.domain.Saturation.Alpha.Filename = 'alpha.pfb'
run.Geom.domain.Saturation.N.Filename = 'n.pfb'
run.Geom.domain.Saturation.SRes.Filename = 's_res.pfb'
run.Geom.domain.Saturation.SSat.Filename = 's_sat.pfb'
```

---

## Initialization: SetupRichards()

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `SetupRichards()`

### Purpose
Prepare all data structures before time-stepping begins.

### Key Operations

1. **Metadata Initialization**
   ```c
   PFModuleInvokeType(MetadataInvoke, public_xtra->metadata,
                      (js_inputs, "parflow"));
   ```
   Creates JSON provenance tracking for reproducibility.

2. **Problem Data Setup**
   ```c
   PFModuleInvokeType(SetProblemDataInvoke, set_problem_data,
                      (problem_data));
   ```
   Calls `SetProblemData()` to evaluate all property modules (see above).

3. **Domain Geometry**
   ```c
   ComputeTop(problem, top, ...);
   ComputeBottom(problem, bottom, ...);
   ```
   Determines active/inactive regions and top/bottom surfaces for overland flow.

4. **Terrain-Following Grid** (if enabled)
   ```c
   if (public_xtra->terrain_following_grid) {
       // Compute topographic slopes for TFG formulation
       ComputeSlopes(...);
   }
   ```

5. **CLM Initialization** (if `HAVE_CLM`)
   - Allocate forcing data vectors
   - Initialize 1D forcing arrays
   - Set up CLM output grids

---

## Initial Conditions

**Location:** [`problem_ic_phase_pressure.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_ic_phase_pressure.c)

Initial conditions set the pressure field at simulation start.

### IC Types

| Type | Key Value | Description |
|------|-----------|-------------|
| 0 | `Constant` | Uniform pressure per region |
| 1 | `HydroStaticDepth` | Hydrostatic from reference elevation |
| 2 | `HydroStaticPatch` | Hydrostatic from reference patch |
| 3 | `PFBFile` | Read from ParFlow binary |
| 4 | `NCFile` | Read from NetCDF |

### Key Parsing (`ICPhasePressureNewPublicXtra()`)

```c
// Select IC type
switch_name = GetString("ICPressure.Type");
public_xtra->type = NA_NameToIndex(type_na, switch_name);

// Get regions to apply IC
switch_name = GetString("ICPressure.GeomNames");
public_xtra->regions = NA_NewNameArray(switch_name);

// For each region, get parameters
for (ir = 0; ir < num_regions; ir++) {
    region = NA_IndexToName(public_xtra->regions, ir);
    sprintf(key, "Geom.%s.ICPressure.Value", region);
    values[ir] = GetDouble(key);
}
```

### Configuration Examples

See [Initial Conditions keys](https://parflow.readthedocs.io/en/latest/keys.html#initial-conditions-pressure) in the ParFlow manual.

```python
# Constant IC
run.ICPressure.Type = 'Constant'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.Value = -1.0  # Unsaturated

# Hydrostatic from depth
run.ICPressure.Type = 'HydroStaticDepth'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.Value = 0.0        # Pressure at ref elevation
run.Geom.domain.ICPressure.RefElevation = 5.0  # Water table depth

# From restart file
run.ICPressure.Type = 'PFBFile'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.FileName = 'restart.out.press.00100.pfb'
```

---

## Boundary Conditions

**Location:** [`bc_pressure_package.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/bc_pressure_package.c), [`problem_bc_pressure.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_bc_pressure.c)

Boundary conditions are applied to named patches on the domain boundary.

### BC Types

| Type | Key Value | Code | Description |
|------|-----------|------|-------------|
| 0 | `DirEquilRefPatch` | Dirichlet | Hydrostatic from reference patch |
| 1 | `DirEquilPLinear` | Dirichlet | Piecewise linear pressure |
| 2 | `FluxConst` | Neumann | Constant flux |
| 3 | `FluxVolumetric` | Neumann | Volumetric flux rate |
| 4 | `PressureFile` | Dirichlet | Pressure from PFB file |
| 5 | `FluxFile` | Neumann | Flux from PFB file |
| 7 | `OverlandFlow` | Mixed | Overland with rainfall |
| 10 | `OverlandKinematic` | Mixed | Kinematic wave routing |
| 11 | `OverlandDiffusive` | Mixed | Diffusive wave routing |

### Patch Definition

Patches are defined on geometry solids:

```python
# Define domain geometry with patches
run.GeomInput.Names = 'domain_input'
run.GeomInput.domain_input.InputType = 'Box'
run.GeomInput.domain_input.GeomName = 'domain'

run.Geom.domain.Patches = 'left right front back bottom top'

# Lower values define patch extents
run.Geom.domain.Lower.X = 0.0
run.Geom.domain.Lower.Y = 0.0
run.Geom.domain.Lower.Z = 0.0
run.Geom.domain.Upper.X = 100.0
run.Geom.domain.Upper.Y = 100.0
run.Geom.domain.Upper.Z = 10.0
```

### BC Configuration

See [Boundary Conditions keys](https://parflow.readthedocs.io/en/latest/keys.html#boundary-conditions-pressure) in the ParFlow manual.

```python
# Apply BCs to patches
run.BCPressure.PatchNames = 'left right front back bottom top'

# No-flow sides
for patch in ['left', 'right', 'front', 'back']:
    run.Patch[patch].BCPressure.Type = 'FluxConst'
    run.Patch[patch].BCPressure.Cycle = 'constant'
    run.Patch[patch].BCPressure.alltime.Value = 0.0

# Fixed water table at bottom
run.Patch.bottom.BCPressure.Type = 'DirEquilRefPatch'
run.Patch.bottom.BCPressure.RefGeom = 'domain'
run.Patch.bottom.BCPressure.RefPatch = 'bottom'
run.Patch.bottom.BCPressure.Cycle = 'constant'
run.Patch.bottom.BCPressure.alltime.Value = 0.0

# Overland flow at top
run.Patch.top.BCPressure.Type = 'OverlandKinematic'
run.Patch.top.BCPressure.Cycle = 'constant'
run.Patch.top.BCPressure.alltime.Value = -0.001  # Rainfall rate [L/T]
```

### Time Cycling

BCs can vary in time using cycles:

```python
# Define a daily cycle
run.Cycle.Names = 'daily'
run.Cycle.daily.Names = 'day night'
run.Cycle.daily.day.Length = 12
run.Cycle.daily.night.Length = 12
run.Cycle.daily.Repeat = -1  # Repeat forever

# Apply different rainfall day vs night
run.Patch.top.BCPressure.Cycle = 'daily'
run.Patch.top.BCPressure.day.Value = -0.002    # Higher during day
run.Patch.top.BCPressure.night.Value = -0.0005  # Lower at night
```

---

## Solver Configuration: SolverRichardsNewPublicXtra()

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `SolverRichardsNewPublicXtra()`

This function reads solver configuration from the input database.

### Key Configuration Categories

| Category | Example Keys | Purpose |
|----------|--------------|---------|
| **Solver** | `Solver.Nonlinear.*` | Nonlinear solver parameters |
| **Output** | `Solver.PrintSubsurfData` | What to write to files |
| **Timing** | `Solver.MaxIter` | Iteration limits |
| **CLM** | `Solver.CLM.*` | Land surface model (see CLM docs) |

### Nonlinear Solver Keys

See [Richards Solver keys](https://parflow.readthedocs.io/en/latest/keys.html#richards-equation-solver-parameters) in the ParFlow manual.

```python
# Newton-Krylov parameters
run.Solver.Nonlinear.MaxIter = 100
run.Solver.Nonlinear.ResidualTol = 1e-6
run.Solver.Nonlinear.StepTol = 1e-30
run.Solver.Nonlinear.EtaChoice = 'Walker1'
run.Solver.Nonlinear.Globalization = 'LineSearch'

# Linear solver
run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.KrylovDimension = 50
run.Solver.Linear.MaxRestart = 2
```

---

## Instance Initialization: SolverRichardsInitInstanceXtra()

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `SolverRichardsInitInstanceXtra()`

Creates computational grids and allocates runtime vectors.

### Grid Creation

```c
// Main 3D flow grid (cell-centered)
instance_xtra->grid = NewGrid(...);

// 2D surface grid (for overland flow)
instance_xtra->grid2d = NewGrid(..., nz=1);

// Staggered velocity grids (face-centered)
instance_xtra->x_grid = NewGrid(..., nx+1);  // x-faces
instance_xtra->y_grid = NewGrid(..., ny+1);  // y-faces
instance_xtra->z_grid = NewGrid(..., nz+1);  // z-faces
```

### Vector Allocations

**State Vectors:**
- `pressure`, `old_pressure` - Current and previous pressure
- `saturation`, `old_saturation` - Water saturation
- `density`, `old_density` - Fluid density

**Flux Vectors:**
- `evap_trans` - Evapotranspiration from CLM
- `x_velocity`, `y_velocity`, `z_velocity` - Darcy velocities
- `q_overlnd_x`, `q_overlnd_y` - Overland flow velocities

**CLM Vectors** (when `HAVE_CLM`):
- Forcing: `sw_forc`, `lw_forc`, `prcp_forc`, `tas_forc`, `u_forc`, `v_forc`, `patm_forc`, `qatm_forc`
- Vegetation: `lai_forc`, `sai_forc`, `z0m_forc`, `displa_forc`
- Output: `eflx_lh_tot`, `eflx_sh_tot`, `qflx_evap_tot`, `swe_out`, `t_grnd`, ...

---

## Main Time-Stepping Loop: AdvanceRichards()

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `AdvanceRichards()`

This is the core simulation loop that advances the solution through time.

### Loop Structure

```c
do {
    // 1. Select outer timestep
    ct += cdt;  // cumulative time advances by cdt

    // 2. Read CLM forcing for this interval
    // 3. Call CLM land surface model

    // 4. Inner convergence loop
    do {
        // Adjust dt if needed
        t += dt;
        // Nonlinear solve
        retval = NonlinSolverInvoke(...);
        converged = (retval == 0);
    } while (!converged && conv_failures < max_failures);

    // 5. Post-convergence processing
    // 6. Output files

    take_more_time_steps = (iteration < max_iterations) && (t < stop_time);
} while (take_more_time_steps);
```

### Two-Level Time Structure

ParFlow uses a two-level time structure:

| Variable | Purpose | Updated By |
|----------|---------|------------|
| `ct` | Cumulative time (forcing boundaries) | `ct += cdt` at loop start |
| `t` | Physics time (solver state) | `t += dt` in inner loop |
| `cdt` | Outer timestep (forcing interval) | `SelectTimeStep` module |
| `dt` | Inner timestep (solver) | May be reduced to hit `ct` |

The inner timestep `dt` must respect forcing boundaries:
```c
if (t + dt > ct) {
    new_dt = ct - t;  // Reduce to hit boundary exactly
    dt = new_dt;
}
```

---

## CLM Integration

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `AdvanceRichards()` CLM forcing and coupling section

When compiled with `PARFLOW_HAVE_CLM=ON`, the Richards solver integrates with the Common Land Model for land surface processes.

### CLM Coupling Overview

1. **Forcing data** is read at each outer timestep (modes: 1D uniform, 2D spatial PFB, 3D preloaded, NetCDF)
2. **CLM is called** once per outer timestep via `CALL_CLM_LSM` macro
3. **ET flux** (`evap_trans`) is returned to ParFlow as a sink term
4. CLM is **not called** during inner Newton iterations

### Key CLM Configuration

See [CLM Solver keys](https://parflow.readthedocs.io/en/latest/keys.html#clm-solver-parameters) in the ParFlow manual.

```python
run.Solver.LSM = 'CLM'
run.Solver.CLM.MetForcing = '2D'        # 1D, 2D, 3D, or 4 (NetCDF)
run.Solver.CLM.MetFilePath = './forcing'
run.Solver.CLM.MetFileName = 'NLDAS'
```

**For detailed CLM documentation, see:**
- `docs/CLM_Snow_Parameterizations.md` - Snow model options
- `docs/CLM_SNOW_Code_Flow.md` - CLM code flow and physics

---

## Inner Convergence Loop

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `AdvanceRichards()` inner do-while loop

Solves the nonlinear Richards equation for a single timestep.

### Nonlinear Solver Invocation

```c
retval = PFModuleInvokeType(NonlinSolverInvoke, nonlin_solver,
    (pressure,          // Solution vector (in/out)
     density,           // Phase density
     old_density,       // Previous density
     saturation,        // Saturation (output)
     old_saturation,    // Previous saturation
     old_pressure,      // Previous pressure
     dt,                // Timestep size
     t,                 // Current time
     evap_trans,        // ET flux from CLM
     ovrl_bc_flx,       // Overland BC flux
     ...));
```

### Convergence Handling

```c
converged = (retval == 0);

if (!converged) {
    conv_failures++;
    if (conv_failures >= max_failures) {
        // Too many failures - abort or reduce timestep
        break;
    }
    // Retry with same or smaller timestep
}
```

### What Happens Inside NonlinSolverInvoke

See [`kinsol_nonlin_solver.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_nonlin_solver.c):

1. **Setup:** Configure KINSOL with tolerances, max iterations
2. **Iterate:** Newton-Krylov iterations
   - Evaluate residual: `NlFunctionEval()`
   - Solve linear system: SPGMR with preconditioner
   - Update solution: `p_new = p_old + α * δp`
3. **Check convergence:** ||F(p)|| < tol
4. **Return:** 0 = success, nonzero = failure

---

## Post-Convergence Processing

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `AdvanceRichards()` post-convergence section

After successful convergence:

### 1. Spinup Correction (Optional)
```c
if (spinup == 1) {
    // Force surface pressures to zero
    // Prevents ponding during initialization
    ForEachTopCell(...) {
        pressure[i] = 0.0;
    }
}
```

### 2. Surface Pressure Reset (Optional)
```c
if (reset_surface_pressure == 1) {
    // Cap excessive surface pressures
    ForEachTopCell(...) {
        if (pressure[i] > threshold) {
            pressure[i] = threshold;
        }
    }
}
```

### 3. Velocity Update
```c
// Sync ghost cells across processors
InitVectorUpdate(x_velocity, VectorUpdateAll);
InitVectorUpdate(y_velocity, VectorUpdateAll);
InitVectorUpdate(z_velocity, VectorUpdateAll);
FinalizeVectorUpdate(...);
```

### 4. Density and Saturation Update
```c
// Recompute from converged pressure
PFModuleInvokeType(PhaseDensityInvoke, phase_density,
    (0, pressure, density, ...));

PFModuleInvokeType(SaturationInvoke, problem_saturation,
    (saturation, pressure, density, ...));
```

### 5. Water Balance Accumulation
```c
// Integrate ET over timestep
evap_trans_sum += dt * evap_trans;

// Track overland outflow
overland_sum += dt * ovrl_bc_flx;
```

---

## Output and File Writing

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `AdvanceRichards()` output writing section

Output is written at intervals specified by `Solver.DumpInterval`.

### Output Formats

| Format | Files | Use Case |
|--------|-------|----------|
| **PFB** | `.pfb` | Native binary, fast I/O |
| **NetCDF** | `.nc` | Climate-standard, self-describing |
| **SILO** | `.silo` | Visualization (VisIt, ParaView) |

### Variables Written

**Core hydrologic:**
- Pressure, saturation
- Velocities (x, y, z)
- Porosity, permeability

**CLM outputs:**
- Latent/sensible heat flux
- Evaporation components
- Snow water equivalent
- Ground/soil temperature
- Infiltration, irrigation

### Writing Pattern

```c
if (dump_files) {
    // Pressure
    if (print_press) {
        sprintf(filename, "%s.out.press.%05d.pfb", file_prefix, file_number);
        WritePFBinary(filename, pressure);
    }

    // Saturation
    if (print_satur) {
        sprintf(filename, "%s.out.satur.%05d.pfb", file_prefix, file_number);
        WritePFBinary(filename, saturation);
    }

    // CLM variables (when HAVE_CLM)
    if (clm_dump_files) {
        WritePFBinary(..., eflx_lh_tot);
        WritePFBinary(..., swe_out);
        // ... etc
    }
}
```

---

## Finalization: TeardownRichards()

**Location:** [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c), `TeardownRichards()`

Cleanup after simulation completes:

```c
void TeardownRichards() {
    // 1. Finalize metadata
    PFModuleInvokeType(MetadataFinalizeInvoke, metadata, ());

    // 2. Free state vectors
    FreeVector(pressure);
    FreeVector(saturation);
    FreeVector(density);
    // ... all allocated vectors

    // 3. Free grids
    FreeGrid(grid);
    FreeGrid(grid2d);
    FreeGrid(x_grid);
    // ...

    // 4. Free CLM arrays
    if (clm_metforce == 1) {
        tfree(sw1d);
        tfree(lw1d);
        // ... 1D forcing arrays
    }

    // 5. Print statistics
    PrintWellStatistics();
}
```

---

## Key Data Structures

### PublicXtra (Configuration)

Holds all parsed configuration parameters:

```c
typedef struct {
    Problem *problem;
    int max_iterations;
    double dump_interval;

    // Print flags
    int print_press, print_satur, print_velocities;

    // CLM parameters
    int clm_metforce;
    char *clm_metpath;
    int clm_snow_partition;
    double clm_snow_tcrit;
    // ... many more

    // 1D forcing arrays
    double *sw1d, *lw1d, *prcp1d, *tas1d;
    // ...
} PublicXtra;
```

### InstanceXtra (Runtime State)

Holds grids, vectors, and module references:

```c
typedef struct {
    // Grids
    Grid *grid, *grid2d;
    Grid *x_grid, *y_grid, *z_grid;

    // State vectors
    Vector *pressure, *old_pressure;
    Vector *saturation, *old_saturation;
    Vector *density, *old_density;

    // Flux vectors
    Vector *evap_trans, *evap_trans_sum;
    Vector *x_velocity, *y_velocity, *z_velocity;

    // Module references
    PFModule *nonlin_solver;
    PFModule *problem_saturation;
    PFModule *phase_density;

    // Iteration tracking
    int iteration_number;
    int file_number;
} InstanceXtra;
```

---

## Key Files Reference

| File | Key Functions | Purpose |
|------|---------------|---------|
| [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c) | `SolverRichards()`, `SetupRichards()`, `AdvanceRichards()`, `TeardownRichards()` | Main solver orchestration |
| [`nl_function_eval.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/nl_function_eval.c) | `NlFunctionEval()` | Residual F(p) computation |
| [`richards_jacobian_eval.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/richards_jacobian_eval.c) | `RichardsJacobianEval()` | Jacobian ∂F/∂p assembly |
| [`kinsol_nonlin_solver.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_nonlin_solver.c) | `KINSolNonlinSolver()` | Newton-Krylov interface |
| [`problem_saturation.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_saturation.c) | `Saturation()`, `VanGenuchtenSaturation()` | S(p) relationships |
| [`problem_phase_rel_perm.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/problem_phase_rel_perm.c) | `PhaseRelPerm()`, `VanGenuchtenPhaseRelPerm()` | kr(p) relationships |
| [`bc_pressure_package.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/bc_pressure_package.c) | `BCPressurePackage()` | Boundary condition parsing |
| [`overlandflow_eval_Kin.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/overlandflow_eval_Kin.c) | `OverlandFlowEvalKin()` | Overland flow (kinematic) |
| [`overlandflow_eval_diffusive.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/overlandflow_eval_diffusive.c) | `OverlandFlowEvalDiff()` | Overland flow (diffusive) |

---

## Debugging Tips

### Enable Verbose Output
```python
run.Solver.Nonlinear.PrintFlag = 'HighVerbosity'
run.Solver.PrintSubsurfData = True
```

### Check Convergence
Look for in log output:
- `KINSol: residual = ...` - Nonlinear residual per iteration
- `converged` or `failed` messages
- Timestep reductions

### Common Issues

1. **Convergence failure:** Reduce initial timestep, check BCs
2. **Mass balance errors:** Check CLM coupling, well fluxes
3. **Slow performance:** Consider PFMG preconditioner, adjust Krylov dimension
