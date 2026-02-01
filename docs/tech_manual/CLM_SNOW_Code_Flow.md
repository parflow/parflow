# CLM Snow Model: Code Flow and Physics Documentation

This document traces the code flow through the CLM (Common Land Model) snow model as implemented in ParFlow, identifying the physics at each step. The analysis is based on the `feature/clm_snow` branch which includes new parameterization options (January 2025).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Structures](#2-data-structures)
3. [Atmospheric Forcing and Initial Rain-Snow Separation](#3-atmospheric-forcing-and-initial-rain-snow-separation)
4. [Timestep Sequence](#4-timestep-sequence)
5. [Vegetation-Snow Interactions](#5-vegetation-snow-interactions)
   - [5.1 Snow Burial of Vegetation](#51-snow-burial-of-vegetation)
   - [5.2 Canopy Interception of Precipitation](#52-canopy-interception-of-precipitation)
   - [5.3 Canopy Snow and Two-Stream Radiation](#53-canopy-snow-and-two-stream-radiation)
6. [Detailed Physics](#6-detailed-physics)
   - [6.1 Precipitation Partitioning](#61-precipitation-partitioning)
   - [6.2 Snow Albedo](#62-snow-albedo)
   - [6.3 Surface Energy Balance](#63-surface-energy-balance)
   - [6.4 Phase Change (Melt/Freeze)](#64-phase-change-meltfreeze)
   - [6.5 Snow Hydrology](#65-snow-hydrology)
   - [6.6 Snow Compaction](#66-snow-compaction)
   - [6.7 Layer Management](#67-layer-management)
   - [6.8 Snow Age Evolution](#68-snow-age-evolution)
7. [Layer Indexing Convention](#7-layer-indexing-convention)
8. [Physical Constants](#8-physical-constants)
9. [New Parameterizations (feature/clm_snow)](#9-new-parameterizations)
10. [Sensitivity Analysis (Ryken et al. 2020)](#10-sensitivity-analysis-ryken-et-al-2020)
11. [Snow Thermal Conductivity](#11-snow-thermal-conductivity)

---

## 1. Overview

The CLM snow model is a multi-layer scheme (up to 5 snow layers) that simulates:

- **Accumulation**: Rain-snow partitioning and layer initialization
- **Energy balance**: Radiation, turbulent fluxes, heat conduction
- **Phase change**: Melting and refreezing within the snowpack
- **Hydrology**: Water percolation through the snowpack
- **Metamorphism**: Compaction from destructive, overburden, and melt processes
- **Albedo feedback**: Age-dependent albedo with multiple scheme options

### Key Files

| File | Purpose |
|------|---------|
| `clm_main.F90` | Main timestep orchestrator |
| `clm_hydro_canopy.F90` | Precipitation interception, rain-snow partitioning |
| `clm_snowalb.F90` | Snow albedo calculation (3 schemes) |
| `clm_thermal.F90` | Energy balance and heat conduction |
| `clm_meltfreeze.F90` | Phase change with damping mechanisms |
| `clm_hydro_snow.F90` | Water percolation through snowpack |
| `clm_compact.F90` | Snow compaction (3 metamorphism types) |
| `clm_combin.F90` | Merge thin snow layers |
| `clm_subdiv.F90` | Split thick snow layers |
| `clm_snowage.F90` | Snow age evolution for albedo |
| `clmtype.F90` | Data structure definitions |

---

## 2. Data Structures

The `clm1d` derived type in `clmtype.F90` contains all snow state variables.

### Snow Layer State

```fortran
integer  :: snl                    ! Number of snow layers (always <= 0, e.g., -3 means 3 layers)
real(r8) :: h2osno                 ! Snow water equivalent (SWE) [kg/m2]
real(r8) :: h2osno_old             ! SWE from previous timestep
real(r8) :: snowdp                 ! Snow depth [m]
real(r8) :: snowage                ! Non-dimensional snow age [-]
real(r8) :: frac_sno               ! Fractional snow cover [-]
```

### Layer Arrays (indices -nlevsno+1:0 for snow, 1:nlevsoi for soil)

```fortran
real(r8) :: h2osoi_liq(-nlevsno+1:max_nlevsoi)  ! Liquid water [kg/m2]
real(r8) :: h2osoi_ice(-nlevsno+1:max_nlevsoi)  ! Ice content [kg/m2]
real(r8) :: t_soisno  (-nlevsno+1:max_nlevsoi)  ! Temperature [K]
real(r8) :: dz        (-nlevsno+1:max_nlevsoi)  ! Layer thickness [m]
real(r8) :: z         (-nlevsno+1:max_nlevsoi)  ! Layer center depth [m]
real(r8) :: zi        (-nlevsno+0:max_nlevsoi)  ! Interface depth [m]
integer  :: imelt     (-nlevsno+1:max_nlevsoi)  ! Melt (1) or freeze (2) flag
real(r8) :: frac_iceold(-nlevsno+1:max_nlevsoi) ! Ice fraction at previous timestep
```

### Snow Fluxes

```fortran
real(r8) :: qflx_snow_grnd     ! Snowfall rate onto ground [kg/(m2 s)]
real(r8) :: qflx_rain_grnd     ! Rainfall rate onto ground [kg/(m2 s)]
real(r8) :: qflx_snomelt       ! Snowmelt rate [kg/(m2 s)]
real(r8) :: qflx_sub_snow      ! Sublimation from snow [mm/s]
real(r8) :: qflx_dew_snow      ! Dew deposition on snow [mm/s]
real(r8) :: qflx_top_soil      ! Water flux into soil from snow [mm/s]
real(r8) :: eflx_snomelt       ! Latent heat of snowmelt [W/m2]
```

### New Parameterization Options (feature/clm_snow)

```fortran
! Rain-snow partitioning (5 methods)
integer  :: snow_partition_type     ! 0=CLM, 1=wetbulb thresh, 2=wetbulb linear, 3=Dai, 4=Jennings
real(r8) :: tw_threshold            ! Wetbulb temperature threshold [K]
real(r8) :: snow_tcrit              ! Initial T classification threshold above tfrz [K], default 2.5
real(r8) :: snow_t_low              ! CLM method lower T threshold [K], default 273.16
real(r8) :: snow_t_high             ! CLM method upper T threshold [K], default 275.16
real(r8) :: snow_transition_width   ! WetbulbLinear half-width [K], default 1.0
real(r8) :: dai_a                   ! Dai (2008) coefficient a, default -48.2292
real(r8) :: dai_b                   ! Dai (2008) coefficient b, default 0.7205
real(r8) :: dai_c                   ! Dai (2008) coefficient c, default 1.1662
real(r8) :: dai_d                   ! Dai (2008) coefficient d, default 1.0223
real(r8) :: jennings_a              ! Jennings (2018) intercept, default -10.04
real(r8) :: jennings_b              ! Jennings (2018) T coefficient, default 1.41
real(r8) :: jennings_g              ! Jennings (2018) RH coefficient, default 0.09

! Thin snow damping
real(r8) :: thin_snow_damping       ! Damping factor [0-1], 0=off
real(r8) :: thin_snow_threshold     ! SWE threshold [kg/m2]

! SZA-based melt damping
real(r8) :: coszen                  ! Cosine of solar zenith angle
real(r8) :: sza_snow_damping        ! Damping factor [0-1], 1.0=disabled
real(r8) :: sza_damping_coszen_ref  ! Reference coszen (default 0.5, ~60 deg)
real(r8) :: sza_damping_coszen_min  ! Coszen at max damping (default 0.1, ~84 deg)

! Snow albedo schemes
integer  :: albedo_scheme           ! 0=CLM, 1=VIC, 2=Tarboton
real(r8) :: albedo_vis_new          ! Fresh snow VIS albedo [0-1]
real(r8) :: albedo_nir_new          ! Fresh snow NIR albedo [0-1]
real(r8) :: albedo_min              ! Minimum albedo floor [0-1]
real(r8) :: albedo_decay_vis        ! VIS decay coefficient
real(r8) :: albedo_decay_nir        ! NIR decay coefficient
real(r8) :: albedo_accum_a          ! VIC cold-phase decay base
real(r8) :: albedo_thaw_a           ! VIC melt-phase decay base
```

---

## 3. Atmospheric Forcing and Initial Rain-Snow Separation

**File:** `drv_getforce.F90`

**Purpose:** Convert ParFlow meteorological forcing arrays to CLM tile space and perform initial rain-snow separation based on air temperature.

### Forcing Processing Flow

When ParFlow calls CLM, atmospheric forcing is passed as 2D arrays. The `drv_getforce` routine maps these to CLM's tile-based data structure:

```
ParFlow Arrays                    CLM Variables
─────────────────                 ─────────────
sw_pf    ────────────────────►   forc_solad(1,2), forc_solai(1,2)
lw_pf    ────────────────────►   forc_lwrad
prcp_pf  ────────────────────►   forc_rain, forc_snow, itypprc
tas_pf   ────────────────────►   forc_t
u_pf, v_pf ──────────────────►   forc_u, forc_v
patm_pf  ────────────────────►   forc_pbot, forc_rho
qatm_pf  ────────────────────►   forc_q
```

### Solar Radiation Partitioning

Incoming shortwave radiation is split into direct/diffuse and VIS/NIR bands using fixed fractions:

$$S_{vis,dir} = 0.35 \cdot S_{total}$$

$$S_{nir,dir} = 0.35 \cdot S_{total}$$

$$S_{vis,dif} = 0.15 \cdot S_{total}$$

$$S_{nir,dif} = 0.15 \cdot S_{total}$$

**Code location:** `drv_getforce.F90:106-109`

### Initial Rain-Snow Separation

The first rain-snow check occurs in `drv_getforce` based on **air temperature only**:

$$\text{itypprc} = \begin{cases}
0 & \text{if } P = 0 \text{ (no precipitation)} \\
1 & \text{if } T_{air} > T_{frz} + T_{crit} \text{ (rain)} \\
2 & \text{if } T_{air} \leq T_{frz} + T_{crit} \text{ (snow)}
\end{cases}$$

Where:
- $T_{frz} = 273.16$ K (freezing point)
- $T_{crit}$ = configurable via `Solver.CLM.SnowTCrit` (default 2.5 K, from Snow Hydrology, 1956)

**Code location:** `drv_getforce.F90:115-129`

**Note:** The `snow_tcrit` parameter is now read from the clm1d struct rather than the hardcoded `tcrit` constant in `clm_varcon.F90`. This allows users to adjust the initial classification threshold.

This **pre-classification** sets `itypprc` which controls downstream behavior:
- `itypprc = 0`: No precipitation, skip partitioning in `clm_hydro_canopy`
- `itypprc = 1`: All rain (warm conditions), skip partitioning
- `itypprc = 2`: Mixed or snow possible, use detailed partitioning

### Air Density Calculation

Air density is computed from the ideal gas law:

$$\rho_{air} = \frac{P_{atm}}{R_d \cdot T_{air}}$$

Where $R_d = 287.04$ J/(kg·K) is the gas constant for dry air.

**Code location:** `drv_getforce.F90:103`

### Optional Vegetation Forcing

When `clm_forc_veg = 1`, vegetation properties are read from ParFlow forcing arrays rather than computed internally:

```fortran
clm%elai   = lai_pf(l)     ! Exposed LAI
clm%esai   = sai_pf(l)     ! Exposed SAI
clm%z0m    = z0m_pf(l)     ! Roughness length
clm%displa = displa_pf(l)  ! Displacement height
```

This allows external phenology models to drive CLM vegetation dynamics.

---

## 4. Timestep Sequence

The main physics sequence is orchestrated in `clm_main.F90` (lines 180-360). Here is the snow-relevant flow:

```
drv_getforce()              → ATMOSPHERIC FORCING (Section 3)
│   ├── Map PF arrays to CLM tiles
│   ├── Split SW into VIS/NIR, direct/diffuse
│   └── Initial rain-snow separation (air temp based)
│
clm_main()
│
├── 1. clm_dynvegpar()     → VEGETATION DYNAMICS (Section 5.1)
│       ├── Seasonal LAI/SAI adjustment
│       ├── Snow burial of vegetation (fb calculation)
│       └── frac_veg_nosno and frac_sno
│
├── 2. clm_coszen()        → Compute cosine of solar zenith angle
│       └── Store in clm%coszen for use in meltfreeze
│
├── 3. clm_surfalb()       → SURFACE ALBEDO (Section 6.2)
│       ├── clm_snowalb()  → Snow albedo (direct beam)
│       ├── clm_snowalb()  → Snow albedo (diffuse)
│       ├── clm_soilalb()  → Soil albedo
│       └── clm_twostream()→ Canopy radiative transfer (Section 5.3)
│
├── 4. clm_surfrad()       → Net radiative fluxes
│
├── 5. Store previous timestep values
│       ├── h2osno_old = h2osno
│       ├── h2ocan_old = h2ocan
│       └── frac_iceold(j) for all snow layers
│
├── 6. clm_hydro_irrig()   → Irrigation scheduling
│
├── 7. clm_hydro_canopy()  → CANOPY INTERCEPTION & PARTITIONING (Sections 5.2, 6.1)
│       ├── Canopy interception (fpi calculation)
│       ├── Throughfall and drip
│       ├── Detailed rain-snow partitioning (3 methods)
│       ├── Fresh snow density (Alta relationship)
│       └── Initialize snow layer if snowdp >= 0.01 m
│
├── 8. clm_thermal()       → ENERGY BALANCE (Section 6.3)
│       ├── clm_leaftem()  → Leaf temperature (if vegetated)
│       ├── clm_thermalk() → Thermal conductivity (Section 11)
│       ├── clm_tridia()   → Tridiagonal matrix solver
│       └── clm_meltfreeze() → PHASE CHANGE with damping (Section 6.4)
│
├── 9. clm_hydro_snow()    → SNOW HYDROLOGY (Section 6.5)
│
├── 10. IF snow exists (snl < 0):
│       ├── clm_compact()  → COMPACTION (Section 6.6)
│       ├── clm_combin()   → Merge thin layers (Section 6.7)
│       └── clm_subdiv()   → Split thick layers (Section 6.7)
│
├── 11. Update t_grnd = t_soisno(snl+1)
│
└── 12. clm_snowage()      → SNOW AGE update (Section 6.8)
```

---

## 5. Vegetation-Snow Interactions

Vegetation significantly modulates snow dynamics through interception, burial, and radiation effects. The Ryken et al. (2020) sensitivity analysis showed that vegetated areas (e.g., evergreen needleleaf with `frac_veg = 0.8`) are **insensitive to snow albedo parameters** because the canopy intercepts most shortwave radiation.

### 5.1 Snow Burial of Vegetation

**File:** `clm_dynvegpar.F90:56-71`

Snow can bury low vegetation, reducing the effective leaf and stem area indices. The burial fraction is computed from snow depth relative to vegetation height:

$$f_b = \min\left(1, \frac{0.1 \cdot d_{snow}}{z_{0m}}\right)$$

Where:
- $d_{snow}$ = snow depth [m]
- $z_{0m}$ = roughness length [m] (proxy for vegetation height, assuming $h_{veg} \approx 10 \cdot z_{0m}$)

The exposed LAI and SAI are then:

$$\text{LAI}_{exposed} = \text{LAI}_{total} \cdot (1 - f_b)$$

$$\text{SAI}_{exposed} = \text{SAI}_{total} \cdot (1 - f_b)$$

**Physical interpretation:** As snow accumulates, it progressively buries short vegetation. When $d_{snow} = 10 \cdot z_{0m}$ (i.e., snow depth equals vegetation height), all vegetation is buried ($f_b = 1$).

If exposed LAI + SAI < 0.05, the grid cell is treated as bare ground:
```fortran
if ((clm%elai + clm%esai) >= 0.05) then
   clm%frac_veg_nosno = 1   ! Vegetated
else
   clm%frac_veg_nosno = 0   ! Non-vegetated (bare/buried)
endif
```

### 5.2 Canopy Interception of Precipitation

**File:** `clm_hydro_canopy.F90:100-135`

Precipitation falling on a vegetated canopy is partitioned into:
1. **Intercepted precipitation** - held on canopy surfaces
2. **Direct throughfall** - passes directly to ground

#### Interception Fraction

The fraction of precipitation intercepted by the canopy:

$$f_{pi} = 0.25 \cdot \left(1 - e^{-0.5 \cdot (\text{LAI} + \text{SAI})}\right)$$

**Physical basis:** Exponential extinction through the canopy. Denser canopies intercept more precipitation.

#### Canopy Water Capacity

Maximum water storage on canopy:

$$W_{can,max} = \delta_{max} \cdot f_{veg} \cdot (\text{LAI} + \text{SAI})$$

Where $\delta_{max}$ = `dewmx` is the maximum dew storage per unit LAI+SAI (typically 0.1 mm).

#### Throughfall and Drip

Direct throughfall:
$$q_{through} = P \cdot (1 - f_{pi}) \cdot f_{veg}$$

Intercepted water:
$$q_{intercept} = P \cdot f_{pi} \cdot f_{veg}$$

Canopy drip (when storage exceeds capacity):
$$q_{drip} = \max\left(0, \frac{W_{can} - W_{can,max}}{\Delta t}\right)$$

Total precipitation reaching ground:
$$q_{prec,ground} = q_{through} + q_{drip}$$

**Code note:** The leaf water capacity is assumed the same for liquid and solid water, though snow typically has ~2× the holding capacity.

### 5.3 Canopy Snow and Two-Stream Radiation

**File:** `clm_twostream.F90`

#### Canopy Snow Interception Effect on Radiation

When vegetation temperature is below freezing ($T_{veg} < T_{frz}$), intercepted water on the canopy is assumed to be snow/ice. The canopy optical properties are adjusted:

$$\omega = (1 - f_{wet}) \cdot \omega_{leaf} + f_{wet} \cdot \omega_{snow}$$

$$\beta_d = \frac{(1 - f_{wet}) \cdot \omega_{leaf} \cdot \beta_{d,leaf} + f_{wet} \cdot \omega_{snow} \cdot \beta_{d,snow}}{\omega}$$

$$\beta_i = \frac{(1 - f_{wet}) \cdot \omega_{leaf} \cdot \beta_{i,leaf} + f_{wet} \cdot \omega_{snow} \cdot \beta_{i,snow}}{\omega}$$

Where:
- $\omega$ = single scattering albedo (fraction scattered vs absorbed)
- $\beta_d$, $\beta_i$ = upscatter parameters for direct and diffuse radiation
- $f_{wet}$ = fraction of canopy covered by water/snow
- $\omega_{snow}$ = [0.8, 0.4] for [VIS, NIR] bands

**Code location:** `clm_twostream.F90:89-102`

#### Wet/Dry Canopy Fractions

**File:** `clm_hydro_canopy.F90:267-281`

The fraction of canopy covered by intercepted water:

$$f_{wet} = \min\left(1, \left(\frac{W_{can}}{\delta_{max} \cdot (\text{LAI} + \text{SAI})}\right)^{2/3}\right)$$

The dry fraction available for transpiration (leaves only, stems don't transpire):

$$f_{dry} = (1 - f_{wet}) \cdot \frac{\text{LAI}}{\text{LAI} + \text{SAI}}$$

#### Two-Stream Canopy Radiative Transfer

The two-stream approximation (Dickinson 1983; Sellers 1985) computes radiation fluxes through the vegetated canopy:

**Optical depth for direct beam:**
$$K = \frac{G(\mu)}{\mu}$$

Where $G(\mu) = \phi_1 + \phi_2 \cdot \mu$ is the projected leaf/stem area in the solar direction, and:
- $\phi_1 = 0.5 - 0.633\chi - 0.330\chi^2$
- $\phi_2 = 0.877(1 - 2\phi_1)$
- $\chi$ = leaf angle distribution parameter (`xl`, range -0.4 to 0.6)
- $\mu$ = $\cos(\theta_z)$

**Average inverse diffuse optical depth:**
$$\bar{\mu} = \frac{1 - \frac{\phi_1}{\phi_2}\ln\left(\frac{\phi_1 + \phi_2}{\phi_1}\right)}{\phi_2}$$

The two-stream equations solve for upward ($I^+$) and downward ($I^-$) diffuse fluxes, yielding:
- **Absorbed by vegetation:** $f_{abs} = 1 - f_{refl} - (1-\alpha_{grd,dir}) \cdot f_{trans,dir} - (1-\alpha_{grd,dif}) \cdot f_{trans,dif}$
- **Reflected by canopy:** $f_{refl}$
- **Transmitted (direct):** $f_{trans,dir}$
- **Transmitted (diffuse):** $f_{trans,dif}$

**Sunlit canopy fraction:**
$$f_{sun} = \frac{1 - e^{-K \cdot \text{VAI}}}{K \cdot \text{VAI}}$$

Where VAI = LAI + SAI (vegetation area index).

#### Fractional Snow Cover on Ground

**File:** `clm_dynvegpar.F90:84-94`

The fraction of bare ground covered by snow is computed using a configurable scheme.

##### Scheme Selection (feature/clm_snow branch)

**Key:** `Solver.CLM.FracSnoScheme` (default: "CLM")

```fortran
select case (clm%frac_sno_type)

case (0)  ! CLM default
   clm%frac_sno = clm%snowdp / (10.0d0 * clm%frac_sno_roughness + clm%snowdp)

case default  ! Future formulations TBD
   clm%frac_sno = clm%snowdp / (10.0d0 * clm%frac_sno_roughness + clm%snowdp)

end select
```

##### CLM Default Formulation (Type 0)

$$f_{sno} = \frac{d_{snow}}{10 \cdot z_{rough} + d_{snow}}$$

Where:
- $d_{snow}$ = snow depth [m]
- $z_{rough}$ = `FracSnoRoughness` parameter [m] (default 0.01 m)

**Key:** `Solver.CLM.FracSnoRoughness` (default: 0.01)

The default value of 0.01 m matches CLM's `zlnd` parameter for backward compatibility.

##### Physical Interpretation

| $z_{rough}$ | Effect | Use Case |
|-------------|--------|----------|
| 0.005 m | Higher frac_sno for given depth | Smooth surfaces |
| 0.01 m | Default (matches zlnd) | Standard bare ground |
| 0.02 m | Lower frac_sno for given depth | Rough terrain, sparse vegetation |

##### Effect on Ground Albedo

Fractional snow cover blends snow and soil albedos:
$$\alpha_{ground} = f_{sno} \cdot \alpha_{snow} + (1 - f_{sno}) \cdot \alpha_{soil}$$

---

## 6. Detailed Physics

### 6.1 Precipitation Partitioning

**File:** `clm_hydro_canopy.F90` (lines 167-260)

**Purpose:** Determine the fraction of precipitation falling as snow vs. rain after canopy interception.

#### Physics: Rain-Snow Separation

Five methods are available, selected by `snow_partition_type`. The liquid fraction $f_{liq}$ determines partitioning:

$$q_{rain} = f_{liq} \cdot q_{prec,ground}$$
$$q_{snow} = (1 - f_{liq}) \cdot q_{prec,ground}$$

##### Method 0: CLM Default (Air Temperature)

Now uses configurable thresholds `snow_t_low` and `snow_t_high`:

$$f_{liq} = \begin{cases}
0 & T_{air} \leq T_{low} \\
0.4 \cdot \frac{T_{air} - T_{low}}{T_{high} - T_{low}} & T_{low} < T_{air} < T_{high} \\
0.4 & T_{air} \geq T_{high}
\end{cases}$$

Default values: $T_{low} = 273.16$ K (tfrz), $T_{high} = 275.16$ K (tfrz + 2).

**Physics basis:** Linear interpolation assuming mixed-phase precipitation near 0°C. The maximum liquid fraction of 0.4 reflects observations that some snow can persist at warm temperatures.

##### Method 1: Wetbulb Threshold (NEW)

The wetbulb temperature is computed using the Stull (2011) psychrometric approximation:

$$T_{wb} = T \cdot \arctan\left(0.151977\sqrt{RH + 8.313659}\right) + \arctan(T + RH) - \arctan(RH - 1.676331) + 0.00391838 \cdot RH^{1.5} \cdot \arctan(0.023101 \cdot RH) - 4.686035$$

Where:
- $T$ = air temperature [°C]
- $RH$ = relative humidity [%]

Relative humidity is computed from specific humidity:

$$e_{sat} = 611.2 \cdot \exp\left(\frac{17.67 \cdot T_C}{T_C + 243.5}\right)$$

$$q_{sat} = \frac{0.622 \cdot e_{sat}}{P_{atm} - 0.378 \cdot e_{sat}}$$

$$RH = 100 \cdot \frac{q}{q_{sat}}$$

Then:

$$f_{liq} = \begin{cases}
0 & T_{wb} \leq T_{threshold} \text{ (all snow)} \\
1 & T_{wb} > T_{threshold} \text{ (all rain)}
\end{cases}$$

**Physics basis:** Wetbulb temperature accounts for evaporative cooling of falling hydrometeors. In humid conditions, a falling snowflake experiences less evaporation and can survive to warmer surface temperatures.

**Code location:** `clm_hydro_canopy.F90:177-213`

##### Method 2: Wetbulb Linear (NEW)

Now uses configurable `snow_transition_width` for the transition zone half-width:

$$f_{liq} = \begin{cases}
0 & T_{wb} \leq T_{threshold} - W \\
\frac{T_{wb} - (T_{threshold} - W)}{2W} & T_{threshold} - W < T_{wb} < T_{threshold} + W \\
1 & T_{wb} \geq T_{threshold} + W
\end{cases}$$

Where $W$ = `snow_transition_width` (default 1.0 K for 2K total range).

**Physics basis:** Allows for mixed-phase precipitation in marginal conditions with a configurable transition zone.

##### Method 3: Dai (2008) Sigmoidal (NEW)

Uses a sigmoidal (hyperbolic tangent) function of air temperature with four configurable coefficients:

$$p_{snow} = a + b \cdot \tanh(c \cdot (T_C - d))$$

$$f_{liq} = 1 - \max(0, \min(1, p_{snow}))$$

Where:
- $T_C$ = air temperature in Celsius
- $a$ = intercept coefficient (default -48.2292)
- $b$ = amplitude coefficient (default 0.7205)
- $c$ = temperature sensitivity (default 1.1662)
- $d$ = temperature offset in °C (default 1.0223)

**Code location:** `clm_hydro_canopy.F90:229-234`

**Physics basis:** The sigmoidal form captures the smooth transition between rain and snow better than linear methods. Coefficients derived from global precipitation phase observations in Dai (2008). Reference: Dai (2008) JAMC, doi:10.1175/2007JAMC1571.1.

##### Method 4: Jennings (2018) Bivariate Logistic (NEW)

Uses bivariate logistic regression with both air temperature and relative humidity:

$$p_{snow} = \frac{1}{1 + \exp(a + b \cdot T_C + g \cdot RH)}$$

$$f_{liq} = 1 - p_{snow}$$

Where:
- $T_C$ = air temperature in Celsius
- $RH$ = relative humidity in percent
- $a$ = intercept (default -10.04)
- $b$ = temperature coefficient (default 1.41)
- $g$ = relative humidity coefficient (default 0.09)

Relative humidity is calculated from specific humidity using the same formulas as the wetbulb methods.

**Code location:** `clm_hydro_canopy.F90:236-253`

**Physics basis:** Humidity affects the rain-snow threshold because at low RH, evaporative cooling of falling hydrometeors is stronger, allowing snow to persist at warmer air temperatures. The Jennings method explicitly accounts for this effect, improving predictions in dry climates. Reference: Jennings et al. (2018) Nat Commun, doi:10.1038/s41467-018-03629-7.

#### Fresh Snow Density (`bifall`)

**File:** `clm_hydro_canopy.F90:240-246`

Uses the Alta relationship (Anderson 1976; LaChapelle 1961):

$$\rho_{snow,new} = \begin{cases}
50 & T_{air} \leq T_{frz} - 15 \text{ K (cold, light snow)} \\
50 + 1.7 \cdot (T_{air} - T_{frz} + 15)^{1.5} & T_{frz} - 15 < T_{air} \leq T_{frz} + 2 \\
189 & T_{air} > T_{frz} + 2 \text{ K (warm, wet snow)}
\end{cases}$$

```fortran
if (clm%forc_t > tfrz + 2.) then
   bifall = 189.                                        ! warm, wet snow
else if (clm%forc_t > tfrz - 15.) then
   bifall = 50. + 1.7*(clm%forc_t - tfrz + 15.)**1.5   ! transition
else
   bifall = 50.                                         ! cold, light snow
endif
```

Units: kg/m³

**Physics basis:** Fresh snow density increases with temperature because warmer conditions produce larger, wetter snow crystals that pack more densely.

#### Snow Depth and SWE Update

**File:** `clm_hydro_canopy.F90:248-252`

First, partition precipitation reaching ground into rain and snow:

$$q_{snow} = q_{prec,ground} \cdot (1 - f_{liq})$$
$$q_{rain} = q_{prec,ground} \cdot f_{liq}$$

Then compute snow depth change rate and update state:

$$\frac{d(d_{snow})}{dt} = \frac{q_{snow}}{\rho_{snow,new}}$$

$$d_{snow}^{n+1} = d_{snow}^n + \frac{d(d_{snow})}{dt} \cdot \Delta t$$

$$SWE^{n+1} = SWE^n + q_{snow} \cdot \Delta t$$

```fortran
clm%qflx_snow_grnd = clm%qflx_prec_grnd*(1.-flfall)   ! [kg/(m2 s)]
clm%qflx_rain_grnd = clm%qflx_prec_grnd*flfall        ! [kg/(m2 s)]
dz_snowf = clm%qflx_snow_grnd/bifall                  ! [m/s]
clm%snowdp = clm%snowdp + dz_snowf*clm%dtime          ! [m]
clm%h2osno = clm%h2osno + clm%qflx_snow_grnd*clm%dtime  ! [kg/m2] = [mm]
```

| Variable | Units | Description |
|----------|-------|-------------|
| `qflx_prec_grnd` | kg/(m²·s) | Total precipitation rate reaching ground |
| `qflx_snow_grnd` | kg/(m²·s) | Snowfall rate onto ground |
| `qflx_rain_grnd` | kg/(m²·s) | Rainfall rate onto ground |
| `flfall` | - | Liquid fraction (0 = all snow, 1 = all rain) |
| `bifall` | kg/m³ | Fresh snow density $\rho_{snow,new}$ (see [Fresh Snow Density](#fresh-snow-density-bifall) above) |
| `dz_snowf` | m/s | Snow depth accumulation rate |
| `snowdp` | m | Total snow depth |
| `h2osno` | kg/m² | Snow water equivalent (1 kg/m² = 1 mm) |
| `dtime` | s | CLM timestep |

**Note:** For wetland points (`itypwat == istwet`) with warm ground ($T_{grnd} \geq T_{frz}$), snow is immediately removed:
```fortran
if (clm%itypwat==istwet .AND. clm%t_grnd>=tfrz) then
   clm%h2osno=0.
   clm%snowdp=0.
   clm%snowage=0.
endif
```

#### Snow Layer Initialization

When `snowdp >= 0.01 m` and no snow layer exists (`snl = 0`):
```fortran
snl = -1                              ! Create 1 layer
dz(0) = snowdp                        ! Layer thickness [m]
z(0) = -0.5 * dz(0)                   ! Layer center depth [m]
zi(-1) = -dz(0)                       ! Top interface depth [m]
snowage = 0.0                         ! Fresh snow
t_soisno(0) = min(tfrz, forc_t)       ! Temperature capped at freezing
h2osoi_ice(0) = h2osno                ! Ice content = SWE [kg/m2]
h2osoi_liq(0) = 0.0                   ! No liquid initially
```

#### Snow Depth Decrease

Snow depth decreases through three mechanisms: compaction (densification without mass loss), melt (mass loss), and layer management recalculation.

##### 1. Compaction (no mass loss)

**File:** `clm_compact.F90:108-112`

Each layer thickness `dz` decreases due to the three metamorphism terms (see Section 6.6):

```fortran
pdzdtc = ddz1 + ddz2 + ddz3           ! Total fractional rate [1/s]
clm%dz(i) = clm%dz(i) * (1. + pdzdtc*clm%dtime)
```

Since all `pdzdtc` terms are negative, each layer shrinks:

$$\Delta z_j^{n+1} = \Delta z_j^n \cdot \left(1 + \frac{d\Delta z}{dt}\bigg|_{total} \cdot \Delta t\right)$$

Snow depth is then recalculated as $d_{snow} = \sum \Delta z_j$.

##### 2. Melt (mass loss)

**File:** `clm_meltfreeze.F90:201-206`

For a single thin snow layer (`snl+1 == 1`), snow depth decreases proportionally to mass loss:

```fortran
if ((clm%snl+1 == 1) .AND. (clm%h2osno > 0.) .AND. (xm(j) > 0.)) then
   temp1 = clm%h2osno                           ! old SWE [kg/m2]
   clm%h2osno = max(dble(0.), temp1-xm(j))      ! new SWE after melt
   propor = clm%h2osno / temp1                  ! fraction remaining
   clm%snowdp = propor * clm%snowdp             ! scale depth proportionally
   clm%qflx_snomelt = max(dble(0.), (temp1-clm%h2osno)) / clm%dtime
endif
```

$$d_{snow}^{n+1} = d_{snow}^n \cdot \frac{SWE^{n+1}}{SWE^n}$$

This assumes constant density during melt for the thin-snow case.

##### 3. Layer management recalculation

**File:** `clm_combin.F90:87-96`

After layer merging in `clm_combin`, snow depth is recalculated from the sum of all layer thicknesses:

```fortran
clm%h2osno = 0.
clm%snowdp = 0.
do j = clm%snl + 1, 0
   clm%h2osno = clm%h2osno + clm%h2osoi_ice(j) + clm%h2osoi_liq(j)
   clm%snowdp = clm%snowdp + clm%dz(j)
enddo
```

$$d_{snow} = \sum_{j=snl+1}^{0} \Delta z_j$$

$$SWE = \sum_{j=snl+1}^{0} \left(m_{ice,j} + m_{liq,j}\right)$$

##### 4. Complete removal threshold

**File:** `clm_combin.F90:100-108`

If snow depth falls below 1 cm, all snow layers are removed:

```fortran
if (clm%snowdp < 0.01) then       ! all snow gone
   clm%snl = 0
   clm%h2osno = zwice             ! retain any remaining ice mass
   if (clm%h2osno <= 0.) clm%snowdp = 0.
   clm%h2osoi_liq(1) = clm%h2osoi_liq(1) + zwliq  ! liquid ponds on soil
endif
```

**Physics basis:** Very thin snow layers are numerically problematic and physically unrealistic to resolve. Remaining liquid water is added to the top soil layer.

##### Summary of Snow Depth Changes

| Process | Mechanism | Mass Loss? | File:Lines |
|---------|-----------|------------|------------|
| Snowfall | Accumulation increases `dz` | No (gain) | `clm_hydro_canopy.F90:251` |
| Destructive metamorphism | Crystal sintering shrinks `dz` | No | `clm_compact.F90:82-92` |
| Overburden compaction | Weight compresses lower layers | No | `clm_compact.F90:96` |
| Melt compaction | Ice loss collapses structure | No | `clm_compact.F90:100-104` |
| Surface melt (thin snow) | Proportional depth reduction | Yes | `clm_meltfreeze.F90:201-206` |
| Layer recalculation | $d_{snow} = \sum \Delta z_j$ | Depends | `clm_combin.F90:87-96` |
| Removal threshold | `snowdp < 0.01 m` → remove | Yes | `clm_combin.F90:100-108` |

---

### 6.2 Snow Albedo

**File:** `clm_snowalb.F90`

**Purpose:** Calculate snow albedo for visible (VIS) and near-infrared (NIR) bands.

#### Physics: Three Albedo Schemes

##### Scheme 0: CLM Default (Age-Based)

The dimensionless age factor:
$$A = 1 - \frac{1}{1 + \tau_{snow}}$$

where $\tau_{snow}$ is the non-dimensional snow age (see Section 6.8).

Albedo decay:
$$\alpha_{VIS} = \alpha_{VIS,new} \cdot (1 - c_{VIS} \cdot A)$$
$$\alpha_{NIR} = \alpha_{NIR,new} \cdot (1 - c_{NIR} \cdot A)$$

Where:
- $\alpha_{VIS,new} \approx 0.95$ (fresh snow VIS albedo)
- $\alpha_{NIR,new} \approx 0.65$ (fresh snow NIR albedo)
- $c_{VIS} = 0.2$ (VIS decay constant)
- $c_{NIR} = 0.5$ (NIR decay constant)

**Physics basis:** Snow grain growth over time increases absorption. NIR decays faster than VIS because ice is more absorptive in NIR.

##### Scheme 1: VIC (Dual Cold/Warm Rates)

$$t_{days} = \frac{\tau_{snow}}{86400}$$

$$\alpha = \begin{cases}
\alpha_0 \cdot a_{cold}^{t_{days}^{0.58}} & T_{grnd} < T_{frz} \text{ (cold/accumulating)} \\
\alpha_0 \cdot a_{warm}^{t_{days}^{0.46}} & T_{grnd} \geq T_{frz} \text{ (warm/melting)}
\end{cases}$$

Default parameters:
- $a_{cold} = 0.94$ (cold-phase decay base)
- $a_{warm} = 0.82$ (melt-phase decay base)

**Physics basis:** The VIC scheme (Andreadis et al. 2009) captures the physics that grain growth is much faster near the melting point due to liquid water film-mediated recrystallization.

##### Scheme 2: Tarboton (Arrhenius Temperature Dependence)

Temperature-dependent aging rate factor:
$$r = \exp\left(5000 \cdot \left(\frac{1}{T_{frz}} - \frac{1}{T_{grnd}}\right)\right)$$

Capped at $r_{max} = 10$ to prevent numerical instability.

Effective age:
$$A_{eff} = \frac{t_{days} \cdot r}{1 + t_{days} \cdot r}$$

Albedo:
$$\alpha = \alpha_0 \cdot (1 - c \cdot A_{eff})$$

**Physics basis:** Uses Arrhenius-type temperature dependence (Tarboton & Luce 1996). The 5000 K activation energy corresponds to water vapor diffusion in ice. Aging rate increases exponentially as temperature approaches freezing.

#### Zenith Angle Correction (Direct Beam Only)

For direct beam albedo, a correction factor accounts for increased scattering at high solar zenith angles:

$$C = \frac{1 + 1/s_l}{1 + 2 \cdot s_l \cdot \cos\theta_z} - \frac{1}{s_l}$$

$$\Delta\alpha = 0.4 \cdot C \cdot (1 - \alpha)$$

$$\alpha_{direct} = \alpha + \Delta\alpha$$

Where $s_l \approx 2$ is the shape parameter.

**Physics basis:** At low sun angles (high SZA), photons travel longer paths through snow, experiencing more scattering events before being absorbed. This increases the effective albedo.

#### Minimum Albedo Floor

$$\alpha = \max(\alpha_{min}, \alpha)$$

Typically $\alpha_{min} = 0.4$.

**Physics basis:** Prevents unrealistically low albedo for very old, dirty, or debris-covered snow.

---

### 6.3 Surface Energy Balance

**File:** `clm_thermal.F90`

**Purpose:** Solve the energy budget to determine surface temperature and fluxes.

#### Energy Balance Equation

For bare ground or snow surface, energy conservation requires:

$$R_{net} - H - LE - G = 0$$

Where:
- $R_{net}$ = Net radiation (shortwave + longwave) [W/m²]
- $H$ = Sensible heat flux [W/m²]
- $LE$ = Latent heat flux [W/m²]
- $G$ = Ground heat flux [W/m²]

#### Net Radiation

$$R_{net} = (1 - \alpha) \cdot S_{down} + \epsilon \cdot L_{down} - \epsilon \cdot \sigma \cdot T_s^4$$

For snow surfaces:
- $\alpha$ = snow albedo (band-weighted)
- $\epsilon$ = 0.97 (snow emissivity)
- $\sigma$ = 5.67 × 10⁻⁸ W/(m²·K⁴) (Stefan-Boltzmann)

#### Turbulent Fluxes

Sensible heat:
$$H = \rho_{air} \cdot c_p \cdot \frac{(T_s - T_{air})}{r_a}$$

Latent heat:
$$LE = \rho_{air} \cdot \frac{(q_s - q_{air})}{r_a} \cdot L_v$$

Where:
- $r_a$ = aerodynamic resistance [s/m]
- $L_v$ = latent heat of sublimation ($2.844 \times 10^6$ J/kg for snow) or evaporation ($2.501 \times 10^6$ J/kg)
- $q_s$ = saturation specific humidity at surface temperature

#### Heat Conduction (Fourier's Law)

Heat flux between layers:

$$F_j = k_j \cdot \frac{T_{j+1} - T_j}{z_{j+1} - z_j}$$

Where $k_j$ is thermal conductivity (see Section 11).

#### Crank-Nicholson Time Discretization

The temperature equation is solved implicitly:

$$C_j \cdot \frac{T_j^{n+1} - T_j^n}{\Delta t} = (1-\theta) \cdot \left(\frac{F_{j-1}^n - F_j^n}{\Delta z_j}\right) + \theta \cdot \left(\frac{F_{j-1}^{n+1} - F_j^{n+1}}{\Delta z_j}\right)$$

Where:
- $C_j$ = volumetric heat capacity [J/(m³·K)]
- $\theta$ = `cnfac` = 0.5 (Crank-Nicholson weighting)
- Superscript $n$ = current timestep, $n+1$ = next timestep

This forms a tridiagonal matrix system $\mathbf{A} \cdot \mathbf{T}^{n+1} = \mathbf{b}$ solved by Thomas algorithm (`clm_tridia`).

---

### 6.4 Phase Change (Melt/Freeze)

**File:** `clm_meltfreeze.F90`

**Purpose:** Handle melting and freezing within snow and soil layers.

#### Phase Change Identification

Phase change occurs when temperature exceeds/falls below freezing while the appropriate phase is present:

**Melting condition:** $m_{ice} > 0$ and $T > T_{frz}$
**Freezing condition:** $m_{liq} > 0$ and $T < T_{frz}$

When phase change begins, temperature is clamped to $T_{frz} = 273.16$ K.

#### Energy Available for Phase Change

The energy residual after heat conduction:

$$H_m = H_{cond} - C \cdot \frac{(T_{frz} - T^n)}{\Delta t}$$

Where:
- $H_{cond}$ = heat flux from conduction [W/m²]
- $C$ = heat capacity [J/(m²·K)]
- $T^n$ = temperature before phase change

Positive $H_m$ indicates energy available for melting; negative for freezing.

#### Mass Transfer

Mass melted (positive) or frozen (negative):

$$\Delta m = \frac{H_m \cdot \Delta t}{L_f}$$

Where $L_f = 3.336 \times 10^5$ J/kg is the latent heat of fusion.

Update ice and liquid:
$$m_{ice}^{n+1} = \max(0, m_{ice}^n - \Delta m)$$
$$m_{liq}^{n+1} = m_{total} - m_{ice}^{n+1}$$

#### NEW: Combined Damping Mechanism (feature/clm_snow)

**Code location:** `clm_meltfreeze.F90:127-174`

Two damping factors reduce melt energy to correct model biases:

##### Thin Snow Damping

$$D_{depth} = \begin{cases}
D_{min} & SWE \leq 0 \\
D_{min} + (1 - D_{min}) \cdot \frac{SWE}{SWE_{thresh}} & 0 < SWE < SWE_{thresh} \\
1 & SWE \geq SWE_{thresh}
\end{cases}$$

Where:
- $D_{min}$ = `thin_snow_damping` (minimum damping factor, e.g., 0.3)
- $SWE_{thresh}$ = `thin_snow_threshold` (typically 50 kg/m²)

**Physics basis:** Thin snowpacks exhibit unrealistic melt rates because CLM's ground heat flux assumes equilibrium between soil and snow. In reality, shallow snow buffers this exchange.

##### SZA-Based Damping

$$D_{SZA} = \begin{cases}
1 & \cos\theta_z \geq \cos\theta_{ref} \\
D_{min} + (1 - D_{min}) \cdot \frac{\cos\theta_z - \cos\theta_{min}}{\cos\theta_{ref} - \cos\theta_{min}} & \cos\theta_{min} < \cos\theta_z < \cos\theta_{ref} \\
D_{min} & \cos\theta_z \leq \cos\theta_{min}
\end{cases}$$

Default parameters:
- $\cos\theta_{ref} = 0.5$ (reference SZA ~60°)
- $\cos\theta_{min} = 0.1$ (maximum damping SZA ~84°)
- $D_{min}$ = `sza_snow_damping`

**Physics basis:** At high SZA, incident radiation is predominantly diffuse with longer optical paths through snow, increasing effective albedo beyond CLM's prediction. Important at high latitudes.

##### Combined Application

$$D_{total} = D_{depth} \cdot D_{SZA}$$

$$H_m^{damped} = H_m \cdot D_{total}$$

#### Residual Energy

If excess energy remains after complete phase change (all ice melted or all liquid frozen):

$$T^{n+1} = T_{frz} + \frac{H_{residual}}{C}$$

This allows temperature to depart from freezing when phase change is complete.

---

### 6.5 Snow Hydrology

**File:** `clm_hydro_snow.F90`

**Purpose:** Route liquid water through the snowpack.

#### Percolation Scheme: Simple Bucket Overflow (Not Physically Based)

The code explicitly states this is a simplified, non-physical scheme (lines 85-91):

```fortran
! Capillary forces within snow are usually two or more orders of magnitude
! less than those of gravity. Only gravity terms are considered.
! the genernal expression for water flow is "K * ss**3", however,
! no effective parameterization for "K".  Thus, a very simple consideration
! (not physically based) is introduced:
! when the liquid water of layer exceeds the layer's holding
! capacity, the excess meltwater adds to the underlying neighbor layer.
```

**Key simplifications:**
- 1D gravity drainage only (no lateral flow)
- No Darcy flow or Richards equation
- No hydraulic conductivity parameterization
- Instantaneous drainage when holding capacity exceeded

#### No Hydraulic Conductivity

Unlike typical porous media models, CLM snow hydrology does **not** compute hydraulic conductivity. The code acknowledges "no effective parameterization for K" exists. The general form $q = K \cdot S^3$ is mentioned but not implemented.

#### Volume Fractions and Effective Porosity

**File:** `clm_hydro_snow.F90:79-83`

**Ice volume fraction:**
$$\phi_{ice} = \min\left(1, \frac{m_{ice}}{\Delta z \cdot \rho_{ice}}\right)$$

**Effective porosity** (pore space not occupied by ice):
$$\phi_{eff} = 1 - \phi_{ice}$$

**Liquid volume fraction:**
$$\phi_{liq} = \min\left(\phi_{eff}, \frac{m_{liq}}{\Delta z \cdot \rho_{liq}}\right)$$

```fortran
do j = clm%snl+1, 0
   vol_ice(j) = min(dble(1.), clm%h2osoi_ice(j)/(clm%dz(j)*denice))
   eff_porosity(j) = 1. - vol_ice(j)
   vol_liq(j) = min(eff_porosity(j), clm%h2osoi_liq(j)/(clm%dz(j)*denh2o))
enddo
```

#### Holding Capacity and Drainage

Water drains when liquid content exceeds the irreducible saturation:

$$\theta_{hold} = S_{ir} \cdot \phi_{eff}$$

**Drainage condition:** $\phi_{liq} > S_{ir} \cdot \phi_{eff}$

**Drainage flux:**
$$q_{out} = \max\left(0, (\phi_{liq} - S_{ir} \cdot \phi_{eff}) \cdot \Delta z\right) \cdot \rho_{liq}$$

**File:** `clm_hydro_snow.F90:101-106`

```fortran
qout = max(dble(0.), (vol_liq(j) - clm%ssi*eff_porosity(j)) * clm%dz(j))
qout = min(qout, (1.-vol_ice(j+1)-vol_liq(j+1)) * clm%dz(j+1))  ! limit by receiver
```

The drainage is limited by the receiving layer's available capacity to prevent overfilling.

#### Impermeable Layer Check

**File:** `clm_hydro_snow.F90:98-103`

If effective porosity of either the source or receiving layer falls below `wimp` (default 0.05), flow is blocked:

```fortran
if (eff_porosity(j) < clm%wimp .OR. eff_porosity(j+1) < clm%wimp) then
   qout = 0.
else
   qout = max(dble(0.), (vol_liq(j) - clm%ssi*eff_porosity(j)) * clm%dz(j))
   qout = min(qout, (1.-vol_ice(j+1)-vol_liq(j+1)) * clm%dz(j+1))
endif
```

**Physics basis:** High ice content creates impermeable ice lenses that block vertical flow. This is the only mechanism for perched water tables in the snow model.

#### Key Parameters

| Parameter | Variable | Default | Description |
|-----------|----------|---------|-------------|
| Irreducible saturation | `ssi` | 0.033 | Fraction of porosity held by capillary forces |
| Impermeable threshold | `wimp` | 0.05 | Minimum porosity for flow to occur |

These are set in `drv_clmin.dat` and can be spatially variable.

#### Flux to Soil

**File:** `clm_hydro_snow.F90:112-113`

Water percolating through the bottom snow layer becomes input to the soil:

```fortran
qout_snowb = qout/clm%dtime
clm%qflx_top_soil = qout_snowb
```

When no snow layers exist (`snl >= 1`):
$$q_{top,soil} = q_{rain,grnd} + q_{melt} + q_{dew,grnd}$$

#### Soil Condition Does NOT Affect Snow Percolation

**Important limitation:** The snow percolation scheme does **not** consider soil conditions:
- No check for frozen soil
- No check for soil ice fraction
- No check for soil infiltration capacity

The flux `qflx_top_soil` is passed to soil regardless of soil state. What happens next depends on:
- `clm_hydro_soil.F90` (CLM's soil hydrology, largely bypassed in ParFlow coupling)
- ParFlow's Richards solver (which does handle frozen soil via ice saturation)

#### Rain-on-Snow Events

When precipitation is mixed (both rain and snow), the two fractions are handled separately:

##### Snow Fraction: Added in `clm_hydro_canopy`

**File:** `clm_hydro_canopy.F90:307-310`

The snow fraction is added directly to the **top snow layer's ice content** and increases layer thickness:

```fortran
! only ice part of snowfall is added here, the liquid part will be added later
if (clm%snl < 0 .AND. newnode == 0) then
   clm%h2osoi_ice(clm%snl+1) = clm%h2osoi_ice(clm%snl+1) + clm%dtime*clm%qflx_snow_grnd
   clm%dz(clm%snl+1) = clm%dz(clm%snl+1) + dz_snowf*clm%dtime
endif
```

$$m_{ice,top}^{n+1} = m_{ice,top}^n + q_{snow} \cdot \Delta t$$
$$\Delta z_{top}^{n+1} = \Delta z_{top}^n + \frac{q_{snow}}{\rho_{snow,new}} \cdot \Delta t$$

##### Rain Fraction: Added in `clm_hydro_snow`

**File:** `clm_hydro_snow.F90:73-75`

The rain fraction is added to the **top snow layer's liquid water content** (no thickness change):

```fortran
clm%h2osoi_liq(clm%snl+1) = clm%h2osoi_liq(clm%snl+1) +  &
     (clm%qflx_rain_grnd + clm%qflx_dew_grnd - clm%qflx_evap_grnd)*clm%dtime
clm%h2osoi_liq(clm%snl+1) = max(dble(0.), clm%h2osoi_liq(clm%snl+1))
```

$$m_{liq,top}^{n+1} = \max\left(0, m_{liq,top}^n + (q_{rain} + q_{dew} - q_{evap}) \cdot \Delta t\right)$$

##### Subsequent Processes

After rain is added to the top snow layer:

1. **Percolation** - If liquid content exceeds holding capacity ($\theta_{liq} > S_{ir} \cdot \phi_{eff}$), excess drains to lower layers (same timestep)

2. **Refreezing** - In `clm_meltfreeze`, if the snow layer is below freezing and liquid water exists:
   $$T < T_{frz} \text{ and } m_{liq} > 0 \Rightarrow \text{freeze}$$
   Refreezing releases latent heat $L_f$, warming the snow layer.

##### Rain-on-Snow Flow Diagram

```
Mixed Precipitation (rain + snow)
         │
         ▼
┌─────────────────────────────────┐
│   clm_hydro_canopy              │
│   Partition: flfall determines  │
│   qflx_rain_grnd, qflx_snow_grnd│
└─────────────────────────────────┘
         │
         ├──── Snow fraction ────► h2osoi_ice(snl+1) += snow*dt
         │                         dz(snl+1) += dz_snowf*dt
         │
         ▼
┌─────────────────────────────────┐
│   clm_hydro_snow                │
│   Rain added to top layer:      │
│   h2osoi_liq(snl+1) += rain*dt  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Percolation (same routine)    │
│   If θ_liq > ssi*φ_eff:         │
│   excess drains to layer below  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   clm_meltfreeze                │
│   If T < Tfrz and liquid exists:│
│   liquid refreezes, releases Lf │
└─────────────────────────────────┘
```

##### No Explicit Rain Energy Accounting

**Important limitation:** There is **no explicit sensible heat flux** from warm rain falling on cold snow. The rain is simply added as liquid mass; thermal equilibration happens implicitly through the heat conduction solver over subsequent timesteps.

In reality, rain-on-snow events can contribute significant sensible heat that accelerates melt - this advective heat flux is not explicitly captured in CLM.

##### Summary: Rain vs Snow Handling

| Component | Variable Updated | Thickness Change? | File:Lines |
|-----------|------------------|-------------------|------------|
| Snow fraction | `h2osoi_ice(snl+1)` | Yes | `clm_hydro_canopy.F90:308-309` |
| Rain fraction | `h2osoi_liq(snl+1)` | No | `clm_hydro_snow.F90:73-74` |

#### Summary: What This Scheme Can and Cannot Represent

| Feature | Supported? | Notes |
|---------|------------|-------|
| 1D gravity drainage | ✓ | Bucket overflow |
| Capillary retention | ✓ | Via `ssi` parameter |
| Ice lens blocking | ✓ | Via `wimp` threshold |
| Rain-on-snow mass | ✓ | Added as liquid to top layer |
| Rain-on-snow energy | ✗ | No advective heat from warm rain |
| Hydraulic conductivity | ✗ | No K parameterization |
| Preferential flow | ✗ | Homogeneous layers assumed |
| Lateral flow | ✗ | 1D vertical only |
| Frozen soil feedback | ✗ | Handled by ParFlow, not CLM |

**Physics limitation:** Real snowpacks can have complex preferential flow paths, ice lenses at multiple depths, and significant lateral flow on slopes. This scheme is a first-order approximation suitable for large-scale modeling but may miss important dynamics in detailed process studies.

---

### 6.6 Snow Compaction

**File:** `clm_compact.F90`

**Purpose:** Simulate snow densification from three metamorphism processes (based on SNTHERM, Jordan 1991).

#### 1. Destructive Metamorphism (Crystal Growth)

Sintering and grain rounding:

$$\frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{1} = -c_3 \cdot \exp\left(-c_4 \cdot (T_{frz} - T)\right) \cdot f_{\rho} \cdot f_{liq}$$

Temperature factor: Rate increases near melting point.

Density factor (high-density snow compacts slower):
$$f_{\rho} = \begin{cases}
1 & \rho < \rho_{dm} \\
\exp(-46 \times 10^{-3} \cdot (\rho - \rho_{dm})) & \rho \geq \rho_{dm}
\end{cases}$$

Liquid water factor (wet snow metamorphoses faster):
$$f_{liq} = \begin{cases}
c_5 = 2 & \theta_{liq} > 0.01 \\
1 & \text{otherwise}
\end{cases}$$

Constants:
- $c_3 = 2.777 \times 10^{-6}$ s⁻¹
- $c_4 = 0.04$ K⁻¹
- $\rho_{dm} = 100$ kg/m³

**Physics basis:** Vapor diffusion drives crystal growth and bonding (sintering). Rate increases exponentially with temperature.

#### 2. Overburden Compaction

Snow behaves as a viscous material under load:

$$\frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{2} = -\frac{P_s}{\eta_0} \cdot \exp\left(-0.08 \cdot T_d - c_2 \cdot \rho\right)$$

Where:
- $P_s$ = overburden pressure (sum of mass above layer) [kg/m²]
- $\eta_0 = 9 \times 10^5$ kg·s/m² (reference viscosity)
- $T_d = T_{frz} - T$ [K]
- $c_2 = 0.023$ m³/kg

**Physics basis:** Snow deforms plastically under its own weight. Viscosity decreases exponentially with temperature (ice creep is thermally activated) and increases with density.

#### 3. Melt Compaction

Structure collapse when ice melts:

$$\frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{3} = \begin{cases}
-\frac{1}{\Delta t} \cdot \max\left(0, \frac{f_{ice}^{old} - f_{ice}^{new}}{f_{ice}^{old}}\right) & \text{if melting} \\
0 & \text{otherwise}
\end{cases}$$

**Physics basis:** Ice provides structural support. When ice melts, grains settle and porosity decreases.

#### Total Compaction Rate

$$\frac{d\Delta z}{dt} = \Delta z \cdot \left(\frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{1} + \frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{2} + \frac{1}{\Delta z}\frac{d\Delta z}{dt}\bigg|_{3}\right)$$

Update:
$$\Delta z^{n+1} = \Delta z^n \cdot \left(1 + \frac{d\Delta z}{dt} \cdot \Delta t\right)$$

---

### 6.7 Layer Management

#### Combining Thin Layers (`clm_combin.F90`)

Minimum thickness thresholds:
```fortran
dzmin = [0.010, 0.015, 0.025, 0.055, 0.115]  ! meters for layers 1-5
```

If a layer is below threshold or has < 0.1 kg/m2 ice:
1. Merge with neighboring layer (prefer thinner neighbor)
2. Conserve mass and energy using `clm_combo()`
3. Shift overlying layers down
4. Increment `snl`

If total `snowdp < 0.01 m`, all snow is removed:
```fortran
snl = 0
h2osno = zwice
h2osoi_liq(1) = h2osoi_liq(1) + zwliq   ! Liquid ponds on soil
```

#### Subdividing Thick Layers (`clm_subdiv.F90`)

Maximum thickness thresholds:
```fortran
! Layer 1: split if > 0.03 m → 0.02 m + remainder
! Layer 2: split if > 0.07 m (2 layers) or > 0.05 m (3+ layers)
! Layer 3: split if > 0.18 m (3 layers) or > 0.11 m (4+ layers)
! Layer 4: split if > 0.41 m (4 layers) or > 0.23 m (5 layers)
```

**Physics basis:** Multiple layers resolve the temperature gradient and allow realistic melt-refreeze cycles. Maximum 5 layers balances accuracy and computational cost.

---

### 6.8 Snow Age Evolution

**File:** `clm_snowage.F90`

**Purpose:** Track non-dimensional snow age for albedo decay calculations (BATS scheme, Dickinson et al. 1993).

#### Age Increase (Three Terms)

$$\frac{d\tau}{dt} = 10^{-6} \cdot (r_1 + r_2 + r_3)$$

**1. Crystal growth (temperature-dependent Arrhenius):**
$$r_1 = \exp\left(5000 \cdot \left(\frac{1}{T_{frz}} - \frac{1}{T_{grnd}}\right)\right)$$

The 5000 K activation energy represents vapor diffusion in ice.

**2. Near-melting growth (enhanced near 0°C):**
$$r_2 = \exp\left(\min(0, 10 \cdot (T_{frz} - T_{grnd}))\right)$$

Capped to prevent overflow at warm temperatures.

**3. Soot/dirt aging (constant):**
$$r_3 = 0.3$$

Represents contamination that darkens snow over time.

#### Age Reset on Fresh Snow

Fresh snowfall reduces the effective age:

$$\Delta S = 0.1 \cdot \max(0, SWE^{n+1} - SWE^n)$$

$$\tau^{n+1} = \max\left(0, (\tau^n + \Delta\tau) \cdot (1 - \Delta S)\right)$$

**Physics basis:** Fresh snow has small grains and high albedo. The 0.1 factor means 10 mm of new SWE would reduce age by 100% (complete reset).

#### Typical Values

| Condition | $\tau$ | Physical Interpretation |
|-----------|--------|------------------------|
| Fresh snow | 0 | Small grains, high albedo |
| 1 day old | ~0.1 | Beginning grain growth |
| 1 week old | ~1 | Moderate aging |
| Old firn | >10 | Large grains, low albedo |

---

## 7. Layer Indexing Convention

Understanding the indexing is critical:

```
           ATMOSPHERE
              ↓
┌─────────────────────────┐  zi(-4)
│  Snow Layer  snl+1=-4   │
├─────────────────────────┤  zi(-3)
│  Snow Layer  snl+2=-3   │
├─────────────────────────┤  zi(-2)
│  Snow Layer  snl+3=-2   │
├─────────────────────────┤  zi(-1)
│  Snow Layer  snl+4=-1   │
├─────────────────────────┤  zi(0) = 0 (surface)
│  Snow/Soil Interface  0 │
├─────────────────────────┤  zi(1)
│  Soil Layer 1           │
├─────────────────────────┤  zi(2)
│  Soil Layer 2           │
└─────────────────────────┘
```

- `snl` = Number of snow layers (always <= 0)
- Snow indices: `snl+1` to `0` (negative indices)
- Soil indices: `1` to `nlevsoi` (positive indices)
- Example: If `snl = -3`, active snow layers are `-2`, `-1`, `0`

---

## 8. Physical Constants

From `clm_varcon.F90`:

| Constant | Value | Description |
|----------|-------|-------------|
| `tfrz` | 273.16 K | Freezing temperature |
| `hfus` | 3.336e5 J/kg | Latent heat of fusion |
| `hsub` | 2.844e6 J/kg | Latent heat of sublimation |
| `hvap` | 2.510e6 J/kg | Latent heat of evaporation |
| `denice` | 917 kg/m3 | Density of ice |
| `denh2o` | 1000 kg/m3 | Density of liquid water |
| `sb` | 5.67e-8 W/m2/K4 | Stefan-Boltzmann constant |
| `cpice` | 2117.27 J/kg/K | Specific heat of ice |
| `tkice` | 2.29 W/m/K | Thermal conductivity of ice |

---

## 9. New Parameterizations (feature/clm_snow)

### Summary of Additions

| Feature | Keys | Purpose |
|---------|------|---------|
| Rain-snow partitioning | `SnowPartition` (CLM/WetbulbThreshold/WetbulbLinear/Dai/Jennings) | 5 methods for rain-snow separation |
| Configurable thresholds | `SnowTCrit`, `SnowTLow`, `SnowTHigh`, `SnowTransitionWidth` | Tune existing methods |
| Dai coefficients | `DaiCoeffA/B/C/D` | Calibrate sigmoidal method |
| Jennings coefficients | `JenningsCoeffA/B/G` | Calibrate bivariate logistic |
| Thin snow damping | `ThinSnowDamping`, `ThinSnowThreshold` | Prevent premature melt |
| SZA melt damping | `SZASnowDamping`, coszen bounds | Correct high-latitude bias |
| Albedo schemes | `AlbedoScheme` (CLM/VIC/Tarboton) | Flexible albedo decay |
| Fractional snow cover | `FracSnoScheme`, `FracSnoRoughness` | Configurable frac_sno |

### Rain-Snow Partitioning Methods

| Method | Key Value | Best For | Reference |
|--------|-----------|----------|-----------|
| CLM | `CLM` | Default, backward compatible | CLM documentation |
| Wetbulb Threshold | `WetbulbThreshold` | Dry mountain climates | Wang et al. (2019) GRL |
| Wetbulb Linear | `WetbulbLinear` | Smooth transition needed | Wang et al. (2019) GRL |
| Dai Sigmoidal | `Dai` | Global applications | Dai (2008) JAMC |
| Jennings Bivariate | `Jennings` | Humidity-sensitive regions | Jennings et al. (2018) Nat Commun |

### Configuration Examples

**Conservative snow (reduce melt bias):**
```
Solver.CLM.ThinSnowDamping = 0.3
Solver.CLM.SZASnowDamping = 0.5
Solver.CLM.AlbedoScheme = VIC
```

**Aggressive melt (warm climate):**
```
Solver.CLM.ThinSnowDamping = 0.0
Solver.CLM.SZASnowDamping = 1.0
Solver.CLM.AlbedoScheme = CLM
```

**Humidity-aware partitioning (dry continental):**
```
Solver.CLM.SnowPartition = Jennings
```

**Custom Dai coefficients for regional calibration:**
```
Solver.CLM.SnowPartition = Dai
Solver.CLM.DaiCoeffA = -48.2292
Solver.CLM.DaiCoeffB = 0.7205
Solver.CLM.DaiCoeffC = 1.1662
Solver.CLM.DaiCoeffD = 1.0223
```

---

---

## 10. Sensitivity Analysis (Ryken et al. 2020)

The sensitivity of PF-CLM snow processes was analyzed in detail by Ryken et al. (2020) using the active subspaces method at the Niwot Ridge AmeriFlux site.

### Key Findings

#### Most Sensitive Parameters by Land Cover

| Land Cover | Most Sensitive Parameters | Least Sensitive |
|------------|--------------------------|-----------------|
| **Bare Ground** | Shortwave/longwave radiation, albedo coefficients (snal0, snal1, cons, conn) | Wind, pressure |
| **Evergreen Needleleaf** | Longwave radiation, air temperature, humidity | Snow parameters (dampened by canopy) |

#### Vegetation Dampening Effect

Trees with fractional vegetation cover of 0.8 (evergreen needleleaf) cause the model to become **insensitive to snow albedo parameters** because the canopy intercepts most incoming shortwave radiation. This means:

- **Bare ground**: Snow parameters dominate SWE sensitivity
- **Forested areas**: Meteorological forcing dominates

#### Sensitivity by Season

- **Accumulation season** (Oct-Feb): Sensitive to precipitation, temperature, humidity
- **Melt season** (Mar-Jun): Most sensitive overall; radiation and albedo parameters critical
- **September**: Least sensitive (lowest average SWE)

#### Output Ranges from Perturbation Analysis

| Output Metric | Evergreen Needleleaf | Bare Ground |
|---------------|---------------------|-------------|
| Peak SWE range | 31.2 mm | 320.2 mm |
| Time of melt range | 163 hours | 2013 hours |

The ~10× larger ranges for bare ground highlight greater sensitivity to parameter uncertainty.

### Parameter Ranges from Literature

From Ryken et al. (2020), Table 1:

| Parameter | Default | Min | Max | Description |
|-----------|---------|-----|-----|-------------|
| `snal0` | 0.95 | 0.70 | 0.99 | VIS albedo of new snow |
| `snal1` | 0.65 | 0.00 | 0.70 | NIR albedo of new snow |
| `cons` | 0.2 | 0.10 | 0.30 | VIS decay constant |
| `conn` | 0.5 | 0.40 | 0.60 | NIR decay constant |
| `age3` | 0.3 | 0.10 | 0.50 | Soot/dirt aging factor |
| `sl` | 2.0 | 1.0 | 3.0 | Zenith angle factor |

### Implications for Model Calibration

1. **For bare ground/alpine**: Focus calibration on albedo parameters
2. **For forested areas**: Focus on meteorological forcing accuracy
3. **Melt timing**: Most sensitive to radiation and albedo during melt season
4. **Peak SWE**: Sensitive to humidity (affects sublimation) and temperature (affects partitioning)

---

## 11. Snow Thermal Conductivity

**Source:** SNTHERM.89 (Jordan, 1991), as documented in Ryken et al. (2020) Eq. 6.

The thermal conductivity of snow:

$$k_{snow} = k_a + (7.75 \times 10^{-5} \rho_{iw} + 1.105 \times 10^{-6} \rho_{iw}^2)(k_i - k_a)$$

Where:
- $k_a = 0.023$ W/(m·K) (thermal conductivity of air)
- $k_i = 2.29$ W/(m·K) (thermal conductivity of ice)
- $\rho_{iw}$ = partial density of ice and liquid water [kg/m³]

$$\rho_{iw} = \frac{m_{ice} + m_{liq}}{\Delta z}$$

**Physics basis:** Snow thermal conductivity increases nonlinearly with density because denser snow has more ice-to-ice contacts (grain bonds) for heat conduction.

| Snow Type | $\rho$ (kg/m³) | $k$ (W/(m·K)) |
|-----------|---------------|---------------|
| Fresh powder | 50 | 0.03 |
| Settled snow | 200 | 0.15 |
| Wind-packed | 350 | 0.45 |
| Dense firn | 500 | 0.85 |
| Ice | 917 | 2.29 |

**Effective conductivity for layered snowpack:**

For heat conduction between layers $j$ and $j+1$:

$$k_{eff} = \frac{k_j \cdot k_{j+1} \cdot (\Delta z_j + \Delta z_{j+1})}{k_j \cdot \Delta z_{j+1} + k_{j+1} \cdot \Delta z_j}$$

This is the harmonic mean weighted by layer thickness, appropriate for series heat flow.

---

## References

- Anderson, E.A. (1976). A Point Energy and Mass Balance Model of a Snow Cover. NOAA Technical Report NWS 19.
- Andreadis, K.M., et al. (2009). Modeling snow accumulation and ablation processes in forested environments. Water Resources Research.
- **Dai, A. (2008). Temperature and pressure dependence of the rain-snow phase transition over land and ocean. J. Applied Meteorology and Climatology, 47, 2686-2697.** https://doi.org/10.1175/2008JAMC1860.1
- Dickinson, R.E., et al. (1993). Biosphere-Atmosphere Transfer Scheme (BATS) Version 1e. NCAR Technical Note.
- **Jennings, K.S., Winchell, T.S., Livneh, B., & Molotch, N.P. (2018). Spatial variation of the rain-snow temperature threshold across the Northern Hemisphere. Nature Communications, 9, 2831.** https://doi.org/10.1038/s41467-018-03629-7
- Jordan, R. (1991). A One-Dimensional Temperature Model for a Snow Cover. CRREL Special Report 91-16.
- Jordan, R., Andreas, E.L., & Makshtas, A.P. (1999). Heat budget of snow-covered sea ice at North Pole 4. J. Geophys. Res., 104(C4), 7785-7806.
- **Ryken, A., Bearup, L.A., Jefferson, J.L., Constantine, P., & Maxwell, R.M. (2020). Sensitivity and model reduction of simulated snow processes: Contrasting observational and parameter uncertainty to improve prediction. Advances in Water Resources, 135, 103473.** https://doi.org/10.1016/j.advwatres.2019.103473
- Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air Temperature. J. Applied Meteorology and Climatology.
- Tarboton, D.G. & Luce, C.H. (1996). Utah Energy Balance Snow Accumulation and Melt Model (UEB).
- Wang, Y.H., Broxton, P., Fang, Y., Behrangi, A., Barlage, M., Zeng, X., & Niu, G.Y. (2019). A Wet-Bulb Temperature-Based Rain-Snow Partitioning Scheme Improves Snowpack Prediction Over the Drier Western United States. Geophysical Research Letters, 46, 13825-13835. https://doi.org/10.1029/2019GL085722
- Warren, S.G. & Wiscombe, W.J. (1980). A Model for the Spectral Albedo of Snow. J. Atmos. Sci., 37, 2734-2745.

---

*Document generated January 2025 for ParFlow CLM snow model analysis.*
*Cross-checked against Ryken et al. (2020) sensitivity analysis.*
