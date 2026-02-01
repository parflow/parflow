# CLM Snow Parameterization Options

**Author:** Reed Maxwell (RMM)
**Date:** January 2025
**ParFlow Version:** parflow_21_Jan_26

## Overview

This document describes new snow parameterization options added to ParFlow-CLM to improve snow accumulation and melt simulation. These options are based on parameterizations from the PySnow model that showed improved performance compared to baseline CLM when validated against SNOTEL observations.

Three main enhancements were added:
1. **Wetbulb-based rain-snow partitioning** - Uses wet-bulb temperature instead of air temperature to determine precipitation phase
2. **Thin snow damping** - Reduces melt energy for shallow snowpacks to prevent premature melt
3. **SZA-based snow damping** - Reduces melt energy at high solar zenith angles where CLM underestimates albedo

## New ParFlow Input Keys

All keys use the standard ParFlow CLM namespace: `Solver.CLM.*`

### Snow Partitioning

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.CLM.SnowPartition` | string | `CLM` | Rain-snow partitioning method |
| `Solver.CLM.WetbulbThreshold` | double | `274.15` | Wetbulb temperature threshold for snow [K] |

**SnowPartition Options:**
- `CLM` - Default CLM linear partitioning based on air temperature
- `WetbulbThreshold` - All snow below wetbulb threshold, all rain above
- `WetbulbLinear` - Linear transition over 2K range centered on wetbulb threshold

### Thin Snow Damping

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.CLM.ThinSnowDamping` | double | `0.0` | Damping factor for thin snow melt energy [0-1] |
| `Solver.CLM.ThinSnowThreshold` | double | `50.0` | SWE threshold below which damping applies [mm] |

**Notes:**
- Setting `ThinSnowDamping` to `0.0` disables thin snow damping (default behavior)
- A value of `0.7` means melt energy is reduced to 70% when SWE=0, linearly increasing to 100% at the threshold
- Based on PySnow validation, `ThinSnowDamping=0.7` with `ThinSnowThreshold=50.0` showed good results

### SZA-Based Snow Damping

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.CLM.SZASnowDamping` | double | `1.0` | Damping factor at high SZA [0-1], 1.0=disabled |
| `Solver.CLM.SZADampingCoszenRef` | double | `0.5` | Reference coszen below which damping applies (60°) |
| `Solver.CLM.SZADampingCoszenMin` | double | `0.1` | Coszen at which maximum damping applies (~84°) |

**Notes:**
- Setting `SZASnowDamping` to `1.0` disables SZA damping (default behavior)
- Damping varies **linearly** with coszen between the reference and minimum bounds
- Physical basis: CLM's narrowband optical parameters assume SZA=60° (coszen=0.5). At higher SZA, actual snow albedo is higher than CLM assumes, so less energy should be available for melt.
- Both thin snow and SZA damping can be enabled simultaneously; they combine **multiplicatively**

**Example damping values with `SZASnowDamping=0.8`:**
- At SZA ≤ 60° (coszen ≥ 0.5): melt energy × 1.0 (no damping)
- At SZA = 72° (coszen ≈ 0.3): melt energy × 0.9
- At SZA ≥ 84° (coszen ≤ 0.1): melt energy × 0.8

## Example Usage

### Python (pftools)

```python
# Enable wetbulb threshold partitioning
run.Solver.CLM.SnowPartition = "WetbulbThreshold"
run.Solver.CLM.WetbulbThreshold = 274.15  # 1°C in Kelvin

# Enable thin snow damping
run.Solver.CLM.ThinSnowDamping = 0.7
run.Solver.CLM.ThinSnowThreshold = 50.0  # mm SWE

# Enable SZA-based snow damping
run.Solver.CLM.SZASnowDamping = 0.8          # 80% melt at high SZA
run.Solver.CLM.SZADampingCoszenRef = 0.5     # Start damping below 60° SZA
run.Solver.CLM.SZADampingCoszenMin = 0.1     # Max damping at ~84° SZA
```

### TCL

```tcl
pfset Solver.CLM.SnowPartition          WetbulbThreshold
pfset Solver.CLM.WetbulbThreshold       274.15
pfset Solver.CLM.ThinSnowDamping        0.7
pfset Solver.CLM.ThinSnowThreshold      50.0
pfset Solver.CLM.SZASnowDamping         0.8
pfset Solver.CLM.SZADampingCoszenRef    0.5
pfset Solver.CLM.SZADampingCoszenMin    0.1
```

---

## Technical Implementation Details

### Files Modified

#### 1. `pfsimulator/clm/clmtype.F90`

Added new tile-level parameters to the `clm1d` derived type:

```fortran
! Snow parameterization options @RMM 2025
     integer  :: snow_partition_type   ! rain-snow partition: 0=CLM linear, 1=wetbulb threshold, 2=wetbulb linear
     real(r8) :: tw_threshold          ! wetbulb temperature threshold for snow [K], default 274.15
     real(r8) :: thin_snow_damping     ! damping factor for thin snow energy [0-1], 0=off
     real(r8) :: thin_snow_threshold   ! SWE threshold for damping [kg/m2 or mm], default 50.0
```

**Location:** After irrigation parameters (~line 260)

#### 2. `pfsimulator/clm/clm.F90`

Added parameter passing from ParFlow C interface to CLM Fortran tiles.

**Changes:**
- Added 4 new arguments to `clm_lsm` subroutine signature
- Added variable declarations for the new parameters
- Added tile assignment loop to copy parameters to each CLM tile

```fortran
! Argument list addition:
snow_partition_typepf,tw_thresholdpf,thin_snow_dampingpf,thin_snow_thresholdpf)

! Variable declarations:
integer  :: snow_partition_typepf
real(r8) :: tw_thresholdpf
real(r8) :: thin_snow_dampingpf
real(r8) :: thin_snow_thresholdpf

! Tile assignment:
clm(t)%snow_partition_type  = snow_partition_typepf
clm(t)%tw_threshold         = tw_thresholdpf
clm(t)%thin_snow_damping    = thin_snow_dampingpf
clm(t)%thin_snow_threshold  = thin_snow_thresholdpf
```

#### 3. `pfsimulator/parflow_lib/solver_richards.c`

Added ParFlow input key parsing and structure members.

**Structure members added** (in `PublicXtra`):
```c
/* Snow parameterization options @RMM 2025 */
int clm_snow_partition;       /* CLM snow partition type: 0=linear, 1=wetbulb threshold, 2=wetbulb linear */
double clm_tw_threshold;      /* CLM wetbulb temperature threshold for snow [K] */
double clm_thin_snow_damping; /* CLM thin snow energy damping factor [0-1] */
double clm_thin_snow_threshold; /* CLM SWE threshold for damping [kg/m2] */
```

**Input parsing** (in `SolverRichardsNewPublicXtra`):
- Uses `NA_NewNameArray` for the string switch (CLM/WetbulbThreshold/WetbulbLinear)
- Uses `GetDoubleDefault` for the numeric parameters

**CALL_CLM_LSM macro call** updated to pass new parameters.

#### 4. `pfsimulator/parflow_lib/parflow_proto_f.h`

Updated C-Fortran interface:
- Extended `CALL_CLM_LSM` macro to include 4 new parameters
- Extended `CLM_LSM` function prototype

```c
#define CALL_CLM_LSM(..., clm_nlevsoi, clm_nlevlak,                            \
                clm_snow_partition, clm_tw_threshold, clm_thin_snow_damping, clm_thin_snow_threshold)

void CLM_LSM(..., int *clm_nlevsoi, int *clm_nlevlak,
             int *clm_snow_partition, double *clm_tw_threshold,
             double *clm_thin_snow_damping, double *clm_thin_snow_threshold);
```

#### 5. `pfsimulator/clm/clm_hydro_canopy.F90`

Implemented wetbulb calculation and rain-snow partitioning switch.

**Added parameters for saturation vapor pressure:**
```fortran
real(r8), parameter :: es_a = 611.2d0      ! [Pa] reference saturation vapor pressure
real(r8), parameter :: es_b = 17.67d0      ! coefficient for Clausius-Clapeyron
real(r8), parameter :: es_c = 243.5d0      ! [C] coefficient for Clausius-Clapeyron
```

**Added local variables:**
```fortran
real(r8) :: t_c             ! air temperature in Celsius
real(r8) :: t_wb            ! wetbulb temperature in Celsius
real(r8) :: t_wb_k          ! wetbulb temperature in Kelvin
real(r8) :: rh_pct          ! relative humidity in percent
real(r8) :: e_sat           ! saturation vapor pressure [Pa]
real(r8) :: q_sat           ! saturation specific humidity [kg/kg]
```

**Wetbulb calculation** uses Stull (2011) psychrometric approximation:
```fortran
! Stull (2011) wet-bulb temperature approximation (result in Celsius)
t_wb = t_c * atan(0.151977d0 * sqrt(rh_pct + 8.313659d0)) &
     + atan(t_c + rh_pct) &
     - atan(rh_pct - 1.676331d0) &
     + 0.00391838d0 * (rh_pct**1.5d0) * atan(0.023101d0 * rh_pct) &
     - 4.686035d0
```

**Select case structure** replaces original linear partitioning:
- Case 0 (default): Original CLM air temperature linear method
- Case 1: Wetbulb threshold (binary)
- Case 2: Wetbulb linear (2K transition range)

#### 6. `pfsimulator/clm/clm_meltfreeze.F90`

Added thin snow damping to melt energy calculation.

**Added local variable:**
```fortran
real(r8) :: damping_factor    ! factor to reduce melt energy for thin snowpacks
```

**Damping logic** (applied after energy calculation, before phase change):
```fortran
if (clm%thin_snow_damping > 0.0d0 .and. clm%thin_snow_damping < 1.0d0) then
   do j = clm%snl+1, 0  ! Only snow layers (indices <= 0)
      if (clm%imelt(j) == 1 .and. hm(j) > 0.0d0) then  ! Only for melting
         ! Linear interpolation from damping at SWE=0 to 1.0 at threshold
         if (clm%h2osno <= 0.0d0) then
            damping_factor = clm%thin_snow_damping
         else if (clm%h2osno >= clm%thin_snow_threshold) then
            damping_factor = 1.0d0
         else
            damping_factor = clm%thin_snow_damping + &
                 (1.0d0 - clm%thin_snow_damping) * &
                 (clm%h2osno / clm%thin_snow_threshold)
         endif
         hm(j) = hm(j) * damping_factor
      endif
   enddo
endif
```

---

## Scientific Background

### Wetbulb Rain-Snow Partitioning

Traditional CLM uses air temperature with a linear transition between 0°C and 2°C to partition precipitation into rain and snow. However, wet-bulb temperature is a better predictor of precipitation phase because it accounts for evaporative cooling of falling hydrometeors.

The Stull (2011) formula provides an accurate approximation of wet-bulb temperature from air temperature and relative humidity without requiring iterative calculation.

**Reference:**
Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. Journal of Applied Meteorology and Climatology, 50(11), 2267-2269.

### Thin Snow Damping

Shallow snowpacks in CLM tend to melt too quickly because the full surface energy is applied to a small mass of snow. The thin snow damping parameterization reduces the energy available for melting when SWE is below a threshold, representing:
1. Patchy snow coverage not captured by the model
2. Thermal buffering from underlying soil
3. Reduced albedo feedback for thin snow

The damping factor scales linearly from the specified value at SWE=0 to 1.0 (no damping) at the threshold SWE.

---

## Validation Results

Based on comparison against 8 SNOTEL sites across 5 water years (2020-2024):

| Configuration | Mean Peak SWE Bias |
|--------------|-------------------|
| CLM Baseline | -19% |
| Wetbulb + Damping (0.7) | -3% |

The wetbulb partitioning with thin snow damping significantly reduces the negative bias in peak SWE, particularly at sites with frequent near-freezing precipitation events.

---

## Compilation

After modifying these files, rebuild ParFlow:

```bash
cd /path/to/parflow/build
make
make install
```

No CMake reconfiguration is required as no new source files were added.

---

## Testing

A comprehensive test can compare the new parameterizations against baseline CLM to validate behavior.

### Test Configurations

**PFCLM Variants:**
| Name | Description |
|------|-------------|
| `CLM_Baseline` | Default CLM (air temp linear partitioning) |
| `CLM_Wetbulb` | Wetbulb threshold rain-snow partitioning |
| `CLM_Wetbulb_Damp` | Wetbulb + thin snow damping (0.7, 50mm) |

**PySnow Variants:**
| Name | Description |
|------|-------------|
| `PySnow_Baseline` | Air temp threshold |
| `PySnow_Wetbulb` | Wetbulb threshold |
| `PySnow_Wetbulb_Damp` | Wetbulb + damping (0.7, 50mm) |

### Usage

```bash
# Test single site/year
python test_snow_parameterizations.py --site "CSS Lab" --wy 2024

# Skip PFCLM runs (PySnow comparison only)
python test_snow_parameterizations.py --site "CSS Lab" --wy 2024 --skip-pfclm

# Force re-run existing outputs
python test_snow_parameterizations.py --site "CSS Lab" --wy 2024 --force

# Run all sites and years
python test_snow_parameterizations.py --all
```

### Output

- Comparison plots saved to: `comparison/plots/parameterization_tests/`
- PFCLM outputs saved to: `comparison/pfclm_outputs/{site}_WY{year}/{config_name}/`
- Summary metrics printed to console

### Note on Key Validation

The test script uses `skip_validation=True` when calling `Run.run()` to bypass ParFlow's key database validation for the new snow parameterization keys. This is necessary until the keys are added to the official ParFlow key database.
