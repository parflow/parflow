# Richards Solver Approach

This document describes the numerical solver infrastructure for ParFlow's Richards equation solver, including the nonlinear solver, linear solvers, preconditioners, and configuration options.

## Overview

ParFlow uses a **Newton-Krylov-multigrid** approach to solve the nonlinear Richards equation, as described in [Jones & Woodward (2001)](#references). Improvements for coupled surface-subsurface problems are documented in [Osei-Kuffuor et al. (2014)](#references).

```
Outer: Newton iteration (KINSOL)
  └── Inner: Krylov linear solve (GMRES/SPGMR)
        └── Preconditioner: Multigrid (MGSemi/PFMG/SMG)
```

---

## Solver Hierarchy

```
SolverRichards (top-level)
│
├── Time-stepping loop
│   └── NonlinSolverInvoke (KINSOL)
│       │
│       ├── NlFunctionEval (residual F(p))
│       ├── RichardsJacobianEval (Jacobian ∂F/∂p) [optional]
│       │
│       └── Linear Solve (SPGMR)
│           └── Preconditioner
│               ├── MGSemi (internal multigrid)
│               ├── PFMG (HYPRE parallel multigrid)
│               ├── SMG (HYPRE semi-coarsening)
│               └── PFMGOctree (PFMG with inactive cells)
│
└── Post-solve updates
```

---

## Nonlinear Solver: KINSOL

**Location:** [`kinsol_nonlin_solver.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_nonlin_solver.c)

ParFlow uses KINSOL from the [SUNDIALS library](https://computing.llnl.gov/projects/sundials) (or an embedded version) for the nonlinear solve.

### Newton-Krylov Method

The Newton iteration solves:

$$\mathbf{J}(p^k) \delta p^k = -\mathbf{F}(p^k)$$
$$p^{k+1} = p^k + \alpha \delta p^k$$

where:
- $\mathbf{F}(p)$ = nonlinear residual (Richards equation)
- $\mathbf{J}(p) = \partial \mathbf{F}/\partial p$ = Jacobian matrix
- $\alpha$ = line search parameter (0 < α ≤ 1)

### Inexact Newton

The linear system is solved inexactly using an iterative Krylov method (GMRES). The tolerance is adapted based on nonlinear progress:

$$\|\mathbf{J} \delta p + \mathbf{F}\| \leq \eta_k \|\mathbf{F}\|$$

where $\eta_k$ is the forcing parameter.

### Configuration Keys

See [Richards Solver keys](https://parflow.readthedocs.io/en/latest/keys.html#richards-equation-solver-parameters) in the ParFlow manual.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.Nonlinear.ResidualTol` | double | 1e-7 | Convergence tolerance for ||F(p)|| |
| `Solver.Nonlinear.StepTol` | double | 1e-7 | Convergence tolerance for step size ||δp|| |
| `Solver.Nonlinear.MaxIter` | int | 15 | Maximum Newton iterations per timestep |
| `Solver.Nonlinear.PrintFlag` | string | HighVerbosity | Output level: NoVerbosity, LowVerbosity, NormalVerbosity, HighVerbosity |

### Globalization Strategy

| Key | Value | Description |
|-----|-------|-------------|
| `Solver.Nonlinear.Globalization` | `LineSearch` | Backtracking line search (default, more robust) |
| | `InexactNewton` | No globalization (faster when converging) |

**Line search:** Reduces step size $\alpha$ if full Newton step increases residual.

### Eta (Forcing) Parameters

Control how accurately the linear system is solved:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.Nonlinear.EtaChoice` | string | Walker2 | Method for computing η |
| `Solver.Nonlinear.EtaValue` | double | 1e-4 | Constant η (for EtaConstant) |
| `Solver.Nonlinear.EtaAlpha` | double | 2.0 | Walker2 parameter α |
| `Solver.Nonlinear.EtaGamma` | double | 0.9 | Walker2 parameter γ |

**EtaChoice options:**
- `EtaConstant` - Fixed tolerance (η = EtaValue)
- `Walker1` - Adaptive based on residual ratio
- `Walker2` - Adaptive with safeguards (recommended)

---

## Jacobian Computation

ParFlow supports both analytical and matrix-free Jacobian approaches.

### Analytical Jacobian

**Location:** [`richards_jacobian_eval.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/richards_jacobian_eval.c)

When enabled, the full Jacobian matrix is assembled explicitly.

**Configuration:**
```python
run.Solver.Nonlinear.UseJacobian = True
```

**Structure:** 7-point stencil (3D)

$$J_{i,j} = \frac{\partial F_i}{\partial p_j}$$

For cell $i$, non-zero entries exist for:
- $j = i$ (center)
- $j = i \pm 1$ (x-neighbors)
- $j = i \pm n_x$ (y-neighbors)
- $j = i \pm n_x n_y$ (z-neighbors)

### Matrix-Free (Default)

When `UseJacobian = False`, KINSOL uses finite difference approximations:

$$\mathbf{J} \mathbf{v} \approx \frac{\mathbf{F}(p + \epsilon \mathbf{v}) - \mathbf{F}(p)}{\epsilon}$$

**Configuration:**
```python
run.Solver.Nonlinear.UseJacobian = False
run.Solver.Nonlinear.DerivativeEpsilon = 1e-7
```

**Trade-offs:**
- Matrix-free: Lower memory, no Jacobian assembly cost
- Analytical: Better convergence, required for some preconditioners

### Symmetric Jacobian Option

For preconditioning, can use only the symmetric part:

```python
run.Solver.Linear.Preconditioner.SymmetricMat = 'Symmetric'
run.Solver.Linear.Preconditioner.PCMatrixType = 'PFSymmetric'
```

**PCMatrixType options:**
| Value | Description |
|-------|-------------|
| `FullJacobian` (0) | Full Jacobian (robust) |
| `PFSymmetric` (1) | Symmetric part (default, faster) |
| `SymmetricPart` (2) | Alternative symmetric extraction |
| `Picard` (3) | Linearization around current state |

---

## Linear Solver: SPGMR

**Location:** KINSOL's internal SPGMR (Scaled Preconditioned GMRES)

GMRES iteratively solves the linear system using Krylov subspace projection.

### Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Solver.Linear.KrylovDimension` | int | 10 | Krylov subspace size (GMRES restart) |
| `Solver.Linear.MaxRestarts` | int | 0 | Number of GMRES restarts |

**Krylov dimension trade-offs:**
- Larger: Better convergence, more memory
- Smaller: Less memory, may need more restarts
- Typical range: 10-50

---

## Preconditioners

**Location:** [`kinsol_pc.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_pc.c), various `pf_*.c` files

Preconditioning accelerates linear solver convergence by approximately solving:

$$\mathbf{M}^{-1} \mathbf{J} \delta p = -\mathbf{M}^{-1} \mathbf{F}$$

where $\mathbf{M} \approx \mathbf{J}$ is the preconditioner.

### Selection

```python
# Options: 'NoPC', 'MGSemi', 'SMG', 'PFMG', 'PFMGOctree'
run.Solver.Linear.Preconditioner = 'PFMG'
```

### NoPC (No Preconditioning)

No preconditioning applied. Only useful for testing or very well-conditioned problems.

### MGSemi (Internal Semi-Coarsening Multigrid)

**Location:** [`mg_semi.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/mg_semi.c)

Built-in multigrid without external dependencies.

**Properties:**
- Semi-coarsening (alternates coarsening direction)
- Point relaxation (Gauss-Seidel)
- Lower memory than HYPRE options
- Only works with symmetric matrices

**Configuration:**
```python
run.Solver.Linear.Preconditioner = 'MGSemi'
run.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
run.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10
```

### PFMG (ParFlow MultiGrid)

**Location:** [`pf_pfmg.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_pfmg.c)

From the [HYPRE library](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods). Best general-purpose option.

**Properties:**
- Full coarsening in all directions
- Plane/line relaxations
- Excellent parallel scalability
- Robust for heterogeneous problems

**Configuration:**
```python
run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
run.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
run.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1
run.Solver.Linear.Preconditioner.PFMG.Smoother = 'RedBlackGaussSeidel'
run.Solver.Linear.Preconditioner.PFMG.RAPType = 'NonGalerkin'
```

**RAPType:**
- `Galerkin` - Variational coarse grid operator (more accurate)
- `NonGalerkin` - Direct discretization (faster, default)

### SMG (Semi-Coarsening Multigrid from HYPRE)

**Location:** [`pf_smg.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_smg.c)

**Properties:**
- Semi-coarsening like MGSemi
- Plane relaxations (more robust than point)
- Better for anisotropic problems
- Higher memory than PFMG

**Configuration:**
```python
run.Solver.Linear.Preconditioner = 'SMG'
run.Solver.Linear.Preconditioner.SMG.MaxIter = 1
run.Solver.Linear.Preconditioner.SMG.NumPreRelax = 1
run.Solver.Linear.Preconditioner.SMG.NumPostRelax = 1
```

### PFMGOctree

**Location:** [`pf_pfmg_octree.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_pfmg_octree.c)

PFMG variant optimized for domains with many inactive cells.

**Configuration:**
```python
run.Solver.Linear.Preconditioner = 'PFMGOctree'
# Same options as PFMG
```

---

## Preconditioner Comparison

| Preconditioner | External Library | Memory | Robustness | Speed | Best For |
|----------------|-----------------|--------|------------|-------|----------|
| NoPC | None | Minimal | Poor | Slowest | Testing only |
| MGSemi | None | Low | Moderate | Moderate | Small problems, symmetric matrices |
| PFMG | [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) | Medium | Good | Fast | **General use (recommended)** |
| SMG | [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) | Higher | Best | Moderate | Anisotropic, challenging problems |
| PFMGOctree | [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods) | Medium | Good | Fast | Large inactive regions |

**Note:** PFMG = ParFlow MultiGrid (structured multigrid optimized for ParFlow's grid structure).

---

## HYPRE Integration

**Location:** [`pf_hypre.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_hypre.c)

When using PFMG, SMG, or PFMGOctree, ParFlow interfaces with the [HYPRE library](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods).

### Build Requirement

```bash
cmake .. -DHYPRE_ROOT=/path/to/hypre
```

### Data Transfer

```c
// ParFlow → HYPRE
CopyParFlowVectorToHypreVector(pf_vector, hypre_vector);

// HYPRE → ParFlow
CopyHypreVectorToParFlowVector(hypre_vector, pf_vector);
```

---

## Complete Configuration Example

All examples use the ParFlow Python interface:

```python
from parflow import Run

run = Run("my_simulation", __file__)
```

### Robust Configuration (Challenging Problems)

```python
# Nonlinear solver
run.Solver.Nonlinear.ResidualTol = 1e-6
run.Solver.Nonlinear.StepTol = 1e-6
run.Solver.Nonlinear.MaxIter = 25
run.Solver.Nonlinear.PrintFlag = 'HighVerbosity'
run.Solver.Nonlinear.EtaChoice = 'Walker2'
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Nonlinear.UseJacobian = True

# Linear solver
run.Solver.Linear.KrylovDimension = 30
run.Solver.Linear.MaxRestarts = 2

# Preconditioner
run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
run.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 2
run.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 2
run.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'
```

### Fast Configuration (Well-Posed Problems)

```python
# Nonlinear solver
run.Solver.Nonlinear.ResidualTol = 1e-5
run.Solver.Nonlinear.StepTol = 1e-5
run.Solver.Nonlinear.MaxIter = 10
run.Solver.Nonlinear.EtaChoice = 'Walker2'
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Nonlinear.UseJacobian = False

# Linear solver
run.Solver.Linear.KrylovDimension = 10

# Preconditioner
run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
run.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
run.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1
```

### Low-Memory Configuration

```python
# Matrix-free (no Jacobian storage)
run.Solver.Nonlinear.UseJacobian = False
run.Solver.Nonlinear.DerivativeEpsilon = 1e-7

# Small Krylov subspace
run.Solver.Linear.KrylovDimension = 5

# Internal multigrid (no HYPRE)
run.Solver.Linear.Preconditioner = 'MGSemi'
run.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
run.Solver.Linear.Preconditioner.SymmetricMat = 'Symmetric'
```

---

## Convergence Diagnostics

### Nonlinear Solver Output

With `PrintFlag = HighVerbosity`:

```
KINSOL: iter =  1, ||F|| = 1.234e-02, step = 1.000
KINSOL: iter =  2, ||F|| = 5.678e-04, step = 1.000
KINSOL: iter =  3, ||F|| = 2.345e-06, step = 1.000
KINSOL: iter =  4, ||F|| = 8.901e-08, step = 1.000
KINSOL: converged, final ||F|| = 8.901e-08
```

### Convergence Failure

Common causes and solutions:

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Diverging ||F|| | Time step too large | Reduce initial dt |
| Stagnating ||F|| | Poor initial guess | Better IC, spinup |
| Very slow convergence | Ill-conditioned | Use PFMG, increase Krylov dim |
| Linear solver fails | Singular matrix | Check BCs, permeability |

### Checking Linear Solver

Enable HYPRE diagnostics:

```python
run.Solver.Linear.Preconditioner.PFMG.PrintLevel = 1
```

---

## Time Step Control

### Adaptive Time Stepping

ParFlow can adapt the time step based on convergence:

```python
run.TimeStep.Type = 'Growth'
run.TimeStep.InitialStep = 0.001
run.TimeStep.GrowthFactor = 1.5
run.TimeStep.ReductionFactor = 0.5
run.TimeStep.MaxStep = 1.0
run.TimeStep.MinStep = 1e-8
```

**Behavior:**
- Increase dt by `GrowthFactor` after successful steps
- Decrease dt by `ReductionFactor` after failure
- Constrain to [MinStep, MaxStep]

### Fixed Time Stepping

```python
run.TimeStep.Type = 'Constant'
run.TimeStep.Value = 1.0
```

---

## Parallel Performance

### Strong Scaling

For fixed problem size, optimal processor count depends on:
- Grid size per processor (aim for ~10³-10⁴ cells)
- Communication overhead vs. computation
- Preconditioner parallel efficiency

### Weak Scaling

PFMG and SMG show excellent weak scaling to thousands of processors.

### Load Balancing

For heterogeneous domains (many inactive cells):
- Use `PFMGOctree` preconditioner
- Consider domain decomposition strategy

---

## Algorithm Summary

```
For each timestep:
  1. Compute forcing/BCs at current time

  2. Newton iteration (KINSOL):
     a. Evaluate residual F(p) via NlFunctionEval
        - Compute density ρ(p)
        - Compute saturation S(p)
        - Compute relative permeability kr(p)
        - Assemble finite volume residual

     b. Check convergence: ||F|| < tol?
        - Yes: Accept solution, exit Newton
        - No: Continue to step c

     c. Solve linear system J·δp = -F via SPGMR
        - Apply preconditioner (MGSemi/PFMG/SMG)
        - GMRES iterations until ||r|| < η·||F||

     d. Update solution: p ← p + α·δp
        - Line search to find α if needed

     e. Return to step a (next Newton iteration)

  3. Update auxiliary variables
     - Saturation, density, velocities

  4. Accumulate water balance terms

  5. Write output (if dump interval reached)

  6. Advance to next timestep
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [`solver_richards.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/solver_richards.c) | Top-level solver, time stepping |
| [`kinsol_nonlin_solver.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_nonlin_solver.c) | KINSOL wrapper, Newton iteration |
| [`kinsol_pc.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/kinsol_pc.c) | Preconditioner selection and setup |
| [`nl_function_eval.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/nl_function_eval.c) | Residual F(p) computation |
| [`richards_jacobian_eval.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/richards_jacobian_eval.c) | Jacobian ∂F/∂p assembly |
| [`mg_semi.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/mg_semi.c) | Internal multigrid preconditioner |
| [`pf_pfmg.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_pfmg.c) | HYPRE PFMG wrapper |
| [`pf_smg.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_smg.c) | HYPRE SMG wrapper |
| [`pf_pfmg_octree.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_pfmg_octree.c) | HYPRE PFMG octree variant |
| [`pf_hypre.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pf_hypre.c) | HYPRE integration utilities |
| [`pcg.c`](https://github.com/parflow/parflow/blob/master/pfsimulator/parflow_lib/pcg.c) | Preconditioned CG (for IMPES) |

---

## Troubleshooting Guide

### Problem: Nonlinear solver not converging

**Diagnosis:**
```python
run.Solver.Nonlinear.PrintFlag = 'HighVerbosity'
```

**Solutions:**
1. Reduce initial timestep
2. Use line search: `Globalization = LineSearch`
3. Increase `MaxIter`
4. Check boundary conditions are consistent
5. Verify initial condition is physical

### Problem: Linear solver iterations too high

**Diagnosis:**
Check GMRES iteration count in output.

**Solutions:**
1. Switch to PFMG if using MGSemi
2. Increase `KrylovDimension` (try 20-30)
3. Increase preconditioner iterations: `PFMG.MaxIter 2`
4. Use analytical Jacobian: `UseJacobian True`

### Problem: Memory usage too high

**Solutions:**
1. Use matrix-free: `UseJacobian False`
2. Reduce `KrylovDimension`
3. Use MGSemi instead of HYPRE preconditioners
4. Reduce grid resolution

### Problem: Slow parallel performance

**Solutions:**
1. Increase grid cells per processor
2. Use PFMG (better parallel efficiency than SMG)
3. Optimize processor topology (P×Q×R)
4. Profile with timing: `run.Solver.Timing = True`

---

## References

### ParFlow Solver Development

1. **Jones, J.E., & Woodward, C.S. (2001)**. Newton–Krylov-multigrid solvers for large-scale, highly heterogeneous, variably saturated flow problems. *Advances in Water Resources*, 24(7), 763-774. [doi:10.1016/S0309-1708(00)00075-0](https://doi.org/10.1016/S0309-1708(00)00075-0)
   - Introduces the Newton-Krylov-multigrid approach used in ParFlow
   - Demonstrates scalability on heterogeneous variably saturated problems

2. **Osei-Kuffuor, D., Maxwell, R.M., & Woodward, C.S. (2014)**. Improved numerical solvers for implicit coupling of subsurface and overland flow. *Advances in Water Resources*, 74, 185-195. [doi:10.1016/j.advwatres.2014.09.006](https://doi.org/10.1016/j.advwatres.2014.09.006)
   - Preconditioner improvements for coupled surface-subsurface flow
   - Analysis of solver performance with overland flow boundary conditions

### External Libraries

3. **SUNDIALS** - Suite of Nonlinear and Differential/Algebraic Equation Solvers
   - Website: [https://computing.llnl.gov/projects/sundials](https://computing.llnl.gov/projects/sundials)
   - Hindmarsh, A.C., et al. (2005). SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers. *ACM Transactions on Mathematical Software*, 31(3), 363-396.

4. **HYPRE** - Scalable Linear Solvers and Multigrid Methods
   - Website: [https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)
   - Falgout, R.D., & Yang, U.M. (2002). hypre: A library of high performance preconditioners. *International Conference on Computational Science*, 632-641.

### Background

5. Brown, P.N., & Saad, Y. (1990). Hybrid Krylov methods for nonlinear systems of equations. *SIAM Journal on Scientific and Statistical Computing*, 11(3), 450-481.

6. Ashby, S.F., & Falgout, R.D. (1996). A parallel multigrid preconditioned conjugate gradient algorithm for groundwater flow simulations. *Nuclear Science and Engineering*, 124(1), 145-159.
