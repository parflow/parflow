# -----------------------------------------------------------------------------
# Validation test for qx_overland and qy_overland outputs - ROTATED GEOMETRY
# This test rotates the tilted-V geometry 90 degrees to test if the PFTools
# OverlandFlow calculation has the same issue in Y-direction
# -----------------------------------------------------------------------------

import sys
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb
from parflow.tools.hydrology import calculate_overland_fluxes

def create_tiltedv_rotated_run(flow_method='OverlandFlow'):
    """Create a TiltedV run configuration rotated 90 degrees"""
    overland = Run("overland_flux_validation_rotated", __file__)

    overland.FileVersion = 4

    overland.Process.Topology.P = 1
    overland.Process.Topology.Q = 1
    overland.Process.Topology.R = 1

    overland.ComputationalGrid.Lower.X = 0.0
    overland.ComputationalGrid.Lower.Y = 0.0
    overland.ComputationalGrid.Lower.Z = 0.0

    overland.ComputationalGrid.NX = 5
    overland.ComputationalGrid.NY = 5
    overland.ComputationalGrid.NZ = 1

    overland.ComputationalGrid.DX = 10.0
    overland.ComputationalGrid.DY = 10.0
    overland.ComputationalGrid.DZ = 0.05

    overland.GeomInput.Names = "domaininput leftinput rightinput channelinput"

    overland.GeomInput.domaininput.GeomName = "domain"
    overland.GeomInput.leftinput.GeomName = "left"
    overland.GeomInput.rightinput.GeomName = "right"
    overland.GeomInput.channelinput.GeomName = "channel"

    overland.GeomInput.domaininput.InputType = "Box"
    overland.GeomInput.leftinput.InputType = "Box"
    overland.GeomInput.rightinput.InputType = "Box"
    overland.GeomInput.channelinput.InputType = "Box"

    overland.Geom.domain.Lower.X = 0.0
    overland.Geom.domain.Lower.Y = 0.0
    overland.Geom.domain.Lower.Z = 0.0

    overland.Geom.domain.Upper.X = 50.0
    overland.Geom.domain.Upper.Y = 50.0
    overland.Geom.domain.Upper.Z = 0.05
    overland.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

    # ROTATED: Left/right slopes now in Y-direction
    # Bottom slope
    overland.Geom.left.Lower.X = 0.0
    overland.Geom.left.Lower.Y = 0.0
    overland.Geom.left.Lower.Z = 0.0

    overland.Geom.left.Upper.X = 50.0
    overland.Geom.left.Upper.Y = 20.0
    overland.Geom.left.Upper.Z = 0.05

    # Top slope
    overland.Geom.right.Lower.X = 0.0
    overland.Geom.right.Lower.Y = 30.0
    overland.Geom.right.Lower.Z = 0.0

    overland.Geom.right.Upper.X = 50.0
    overland.Geom.right.Upper.Y = 50.0
    overland.Geom.right.Upper.Z = 0.05

    # Channel in middle
    overland.Geom.channel.Lower.X = 0.0
    overland.Geom.channel.Lower.Y = 20.0
    overland.Geom.channel.Lower.Z = 0.0

    overland.Geom.channel.Upper.X = 50.0
    overland.Geom.channel.Upper.Y = 30.0
    overland.Geom.channel.Upper.Z = 0.05

    overland.Geom.Perm.Names = "domain"
    overland.Geom.domain.Perm.Type = "Constant"
    overland.Geom.domain.Perm.Value = 0.0000001

    overland.Perm.TensorType = "TensorByGeom"

    overland.Geom.Perm.TensorByGeom.Names = "domain"

    overland.Geom.domain.Perm.TensorValX = 1.0
    overland.Geom.domain.Perm.TensorValY = 1.0
    overland.Geom.domain.Perm.TensorValZ = 1.0

    overland.SpecificStorage.Type = "Constant"
    overland.SpecificStorage.GeomNames = "domain"
    overland.Geom.domain.SpecificStorage.Value = 1.0e-4

    overland.Phase.Names = "water"

    overland.Phase.water.Density.Type = "Constant"
    overland.Phase.water.Density.Value = 1.0

    overland.Phase.water.Viscosity.Type = "Constant"
    overland.Phase.water.Viscosity.Value = 1.0

    overland.Contaminants.Names = ""

    overland.Geom.Retardation.GeomNames = ""

    overland.Gravity = 1.0

    overland.TimingInfo.BaseUnit = 0.05
    overland.TimingInfo.StartCount = 0
    overland.TimingInfo.StartTime = 0.0
    overland.TimingInfo.StopTime = 0.4
    overland.TimingInfo.DumpInterval = -2
    overland.TimeStep.Type = "Constant"
    overland.TimeStep.Value = 0.05

    overland.Geom.Porosity.GeomNames = "domain"
    overland.Geom.domain.Porosity.Type = "Constant"
    overland.Geom.domain.Porosity.Value = 0.01

    overland.Domain.GeomName = "domain"

    overland.Phase.RelPerm.Type = "VanGenuchten"
    overland.Phase.RelPerm.GeomNames = "domain"

    overland.Geom.domain.RelPerm.Alpha = 6.0
    overland.Geom.domain.RelPerm.N = 2.0

    overland.Phase.Saturation.Type = "VanGenuchten"
    overland.Phase.Saturation.GeomNames = "domain"

    overland.Geom.domain.Saturation.Alpha = 6.0
    overland.Geom.domain.Saturation.N = 2.0
    overland.Geom.domain.Saturation.SRes = 0.2
    overland.Geom.domain.Saturation.SSat = 1.0

    overland.Wells.Names = ""

    overland.Cycle.Names = "constant rainrec"
    overland.Cycle.constant.Names = "alltime"
    overland.Cycle.constant.alltime.Length = 1
    overland.Cycle.constant.Repeat = -1

    overland.Cycle.rainrec.Names = "rain rec"
    overland.Cycle.rainrec.rain.Length = 2
    overland.Cycle.rainrec.rec.Length = 300
    overland.Cycle.rainrec.Repeat = -1

    overland.BCPressure.PatchNames = overland.Geom.domain.Patches

    overland.Patch.x_lower.BCPressure.Type = "FluxConst"
    overland.Patch.x_lower.BCPressure.Cycle = "constant"
    overland.Patch.x_lower.BCPressure.alltime.Value = 0.0

    overland.Patch.y_lower.BCPressure.Type = "FluxConst"
    overland.Patch.y_lower.BCPressure.Cycle = "constant"
    overland.Patch.y_lower.BCPressure.alltime.Value = 0.0

    overland.Patch.z_lower.BCPressure.Type = "FluxConst"
    overland.Patch.z_lower.BCPressure.Cycle = "constant"
    overland.Patch.z_lower.BCPressure.alltime.Value = 0.0

    overland.Patch.x_upper.BCPressure.Type = "FluxConst"
    overland.Patch.x_upper.BCPressure.Cycle = "constant"
    overland.Patch.x_upper.BCPressure.alltime.Value = 0.0

    overland.Patch.y_upper.BCPressure.Type = "FluxConst"
    overland.Patch.y_upper.BCPressure.Cycle = "constant"
    overland.Patch.y_upper.BCPressure.alltime.Value = 0.0

    overland.Patch.z_upper.BCPressure.Type = flow_method
    overland.Patch.z_upper.BCPressure.Cycle = "rainrec"
    overland.Patch.z_upper.BCPressure.rain.Value = -0.01
    overland.Patch.z_upper.BCPressure.rec.Value = 0.0000

    overland.Mannings.Type = "Constant"
    overland.Mannings.GeomNames = "domain"
    overland.Mannings.Geom.domain.Value = 3.0e-6

    overland.PhaseSources.water.Type = "Constant"
    overland.PhaseSources.water.GeomNames = "domain"
    overland.PhaseSources.water.Geom.domain.Value = 0.0

    overland.KnownSolution = "NoKnownSolution"

    overland.Solver = "Richards"
    overland.Solver.MaxIter = 2500

    overland.Solver.Nonlinear.MaxIter = 100
    overland.Solver.Nonlinear.ResidualTol = 1e-9
    overland.Solver.Nonlinear.EtaChoice = "EtaConstant"
    overland.Solver.Nonlinear.EtaValue = 0.01
    overland.Solver.Nonlinear.UseJacobian = False
    overland.Solver.Nonlinear.DerivativeEpsilon = 1e-15
    overland.Solver.Nonlinear.StepTol = 1e-20
    overland.Solver.Nonlinear.Globalization = "LineSearch"
    overland.Solver.Linear.KrylovDimension = 50
    overland.Solver.Linear.MaxRestart = 2
    overland.Solver.OverlandKinematic.Epsilon = 1e-5

    overland.Solver.Linear.Preconditioner = "PFMG"
    overland.Solver.PrintSubsurf = False
    overland.Solver.Drop = 1e-20
    overland.Solver.AbsTol = 1e-10

    overland.Solver.WriteSiloSubsurfData = False
    overland.Solver.WriteSiloPressure = False
    overland.Solver.WriteSiloSlopes = False
    overland.Solver.WriteSiloSaturation = False
    overland.Solver.WriteSiloConcentration = False

    overland.Solver.PrintQxOverland = True
    overland.Solver.PrintQyOverland = True
    overland.Solver.PrintSlopes = True
    overland.Solver.PrintMannings = True
    overland.Solver.PrintMask = True

    overland.ICPressure.Type = "HydroStaticPatch"
    overland.ICPressure.GeomNames = "domain"
    overland.Geom.domain.ICPressure.Value = -3.0

    overland.Geom.domain.ICPressure.RefGeom = "domain"
    overland.Geom.domain.ICPressure.RefPatch = "z_upper"

    # ROTATED: Slopes now in Y-direction with channel having zero slope
    # X-direction has uniform slope
    overland.TopoSlopesX.Type = "Constant"
    overland.TopoSlopesX.GeomNames = "domain"
    overland.TopoSlopesX.Geom.domain.Value = 0.01

    overland.TopoSlopesY.Type = "Constant"
    overland.TopoSlopesY.GeomNames = "left right channel"
    overland.TopoSlopesY.Geom.left.Value = -0.01
    overland.TopoSlopesY.Geom.right.Value = 0.01
    
    if flow_method == 'OverlandFlow':
        overland.TopoSlopesY.Geom.channel.Value = 0.00
    else:
        overland.TopoSlopesY.Geom.channel.Value = 0.01

    overland.Solver.Nonlinear.UseJacobian = False
    overland.Solver.Linear.Preconditioner.PCMatrixType = "PFSymmetric"

    return overland


def validate_fluxes(output_dir, run_name, flow_method, dx, dy, dt, timestep=5):
    """
    Validate qx_overland and qy_overland outputs against independent calculation
    
    Parameters:
    -----------
    output_dir : str
        Directory containing ParFlow outputs
    run_name : str
        Name of the run
    flow_method : str
        'OverlandFlow' or 'OverlandKinematic'
    dx, dy : float
        Grid spacing in x and y directions (m)
    dt : float
        Timestep (hours)
    timestep : int
        Which timestep to validate
    
    Returns:
    --------
    bool : True if validation passes
    """
    
    print(f"\n{'='*60}")
    print(f"Validating {flow_method} ROTATED GEOMETRY - Timestep {timestep}")
    print(f"{'='*60}\n")
    
    timestep_str = str(timestep).rjust(5, "0")
    
    press_file = f"{output_dir}/{run_name}.out.press.{timestep_str}.pfb"
    qx_file = f"{output_dir}/{run_name}.out.qx_overland.{timestep_str}.pfb"
    qy_file = f"{output_dir}/{run_name}.out.qy_overland.{timestep_str}.pfb"
    slope_x_file = f"{output_dir}/{run_name}.out.slope_x.pfb"
    slope_y_file = f"{output_dir}/{run_name}.out.slope_y.pfb"
    mannings_file = f"{output_dir}/{run_name}.out.mannings.pfb"
    mask_file = f"{output_dir}/{run_name}.out.mask.pfb"
    
    pressure = read_pfb(press_file)
    qx_pf = read_pfb(qx_file)
    qy_pf = read_pfb(qy_file)
    slope_x = read_pfb(slope_x_file)
    slope_y = read_pfb(slope_y_file)
    mannings = read_pfb(mannings_file)
    mask = read_pfb(mask_file)
    
    print(f"Loaded ParFlow outputs:")
    print(f"  Pressure shape: {pressure.shape}")
    print(f"  qx_overland shape: {qx_pf.shape}")
    print(f"  qy_overland shape: {qy_pf.shape}")
    
    if len(qx_pf.shape) == 3:
        qx_pf_2d = qx_pf[0, :, :] * dy
        qy_pf_2d = qy_pf[0, :, :] * dx
    else:
        qx_pf_2d = qx_pf * dy
        qy_pf_2d = qy_pf * dx
    
    if len(slope_x.shape) == 3:
        slope_x_2d = slope_x[0, :, :]
        slope_y_2d = slope_y[0, :, :]
        mannings_2d = mannings[0, :, :]
    else:
        slope_x_2d = slope_x
        slope_y_2d = slope_y
        mannings_2d = mannings
    
    print(f"\nCalculating independent fluxes using ParFlow hydrology tools...")
    qx_calc, qy_calc = calculate_overland_fluxes(
        pressure,
        slope_x_2d,
        slope_y_2d,
        mannings_2d,
        dx,
        dy,
        flow_method=flow_method,
        epsilon=1e-5,
        mask=mask
    )
    
    print(f"  Raw calculated qx shape: {qx_calc.shape}")
    print(f"  Raw calculated qy shape: {qy_calc.shape}")
    
    if flow_method == 'OverlandFlow':
        if len(qx_calc.shape) == 3:
            qx_calc_2d = qx_calc[0, :, :-1]
            qy_calc_2d = qy_calc[0, :-1, :]
        else:
            qx_calc_2d = qx_calc[:, :-1]
            qy_calc_2d = qy_calc[:-1, :]
    else:
        if len(qx_calc.shape) == 3:
            qx_calc_2d = qx_calc[0, :, 1:]
            qy_calc_2d = qy_calc[0, 1:, :]
        else:
            qx_calc_2d = qx_calc[:, 1:]
            qy_calc_2d = qy_calc[1:, :]
    
    print(f"  Subset qx shape: {qx_calc_2d.shape}")
    print(f"  Subset qy shape: {qy_calc_2d.shape}")
    print(f"  ParFlow qx shape: {qx_pf_2d.shape}")
    print(f"  ParFlow qy shape: {qy_pf_2d.shape}")
    
    print(f"\n{'='*60}")
    print(f"Comparing ParFlow outputs vs Independent calculation")
    print(f"{'='*60}")
    
    print(f"\nX-Direction Fluxes (qx):")
    print(f"  ParFlow:    min={qx_pf_2d.min():.6e}, max={qx_pf_2d.max():.6e}, mean={qx_pf_2d.mean():.6e}")
    print(f"  Calculated: min={qx_calc_2d.min():.6e}, max={qx_calc_2d.max():.6e}, mean={qx_calc_2d.mean():.6e}")
    
    print(f"\nY-Direction Fluxes (qy):")
    print(f"  ParFlow:    min={qy_pf_2d.min():.6e}, max={qy_pf_2d.max():.6e}, mean={qy_pf_2d.mean():.6e}")
    print(f"  Calculated: min={qy_calc_2d.min():.6e}, max={qy_calc_2d.max():.6e}, mean={qy_calc_2d.mean():.6e}")
    
    diff_x = qx_pf_2d - qx_calc_2d
    diff_y = qy_pf_2d - qy_calc_2d
    
    print(f"\nDifferences (ParFlow - Calculated):")
    print(f"  X: min={diff_x.min():.6e}, max={diff_x.max():.6e}, mean={diff_x.mean():.6e}, std={diff_x.std():.6e}")
    print(f"  Y: min={diff_y.min():.6e}, max={diff_y.max():.6e}, mean={diff_y.mean():.6e}, std={diff_y.std():.6e}")
    
    threshold = 1e-10
    mask_x = np.abs(qx_calc_2d) > threshold
    mask_y = np.abs(qy_calc_2d) > threshold
    
    print(f"\nRelative errors (where |flux| > {threshold}):")
    
    if mask_x.sum() > 0:
        rel_err_x = np.abs(diff_x[mask_x] / qx_calc_2d[mask_x])
        print(f"  X: mean={rel_err_x.mean()*100:.6f}%, max={rel_err_x.max()*100:.6f}%, median={np.median(rel_err_x)*100:.6f}%")
        print(f"     ({mask_x.sum()} cells, {100*mask_x.sum()/mask_x.size:.1f}% of domain)")
    else:
        print(f"  X: No significant fluxes")
    
    if mask_y.sum() > 0:
        rel_err_y = np.abs(diff_y[mask_y] / qy_calc_2d[mask_y])
        print(f"  Y: mean={rel_err_y.mean()*100:.6f}%, max={rel_err_y.max()*100:.6f}%, median={np.median(rel_err_y)*100:.6f}%")
        print(f"     ({mask_y.sum()} cells, {100*mask_y.sum()/mask_y.size:.1f}% of domain)")
    else:
        print(f"  Y: No significant fluxes")
    
    qx_match = np.allclose(qx_pf_2d, qx_calc_2d, rtol=1e-5, atol=1e-10)
    qy_match = np.allclose(qy_pf_2d, qy_calc_2d, rtol=1e-5, atol=1e-10)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULT")
    print(f"{'='*60}")
    print(f"qx_overland matches: {qx_match}")
    print(f"qy_overland matches: {qy_match}")
    
    if qx_match and qy_match:
        print(f"\n✓ SUCCESS: {flow_method} flux outputs are correct!")
        return True
    else:
        print(f"\n✗ FAILED: {flow_method} flux outputs do not match expected values")
        if not qx_match:
            print(f"   → X-direction has issues")
        if not qy_match:
            print(f"   → Y-direction has issues (SAME AS ORIGINAL X!)")
        return False


if __name__ == "__main__":
    dx = 10.0
    dy = 10.0
    dt = 0.05
    
    # Only test OverlandFlow with rotated geometry
    test_configs = [
        ('OverlandFlow', 'FluxValidation_OverlandFlow_Rotated'),
    ]
    
    all_passed = True
    
    for flow_method, run_name in test_configs:
        print(f"\n{'#'*60}")
        print(f"Testing {flow_method} - ROTATED GEOMETRY")
        print(f"{'#'*60}")
        print(f"\nGeometry: Tilted-V rotated 90 degrees")
        print(f"  - Channel now in Y-direction (y indices 2-3)")
        print(f"  - Y-slopes: -0.01 (bottom), 0.0 (channel), +0.01 (top)")
        print(f"  - X-slope: +0.01 (uniform)")
        print(f"\nThis tests if PFTools Y-direction has same issue as X-direction")
        print(f"in the original geometry.\n")
        
        overland = create_tiltedv_rotated_run(flow_method=flow_method)
        overland.set_name(run_name)
        
        output_dir = get_absolute_path(f"test_output/{run_name}")
        mkdir(output_dir)
        
        print(f"\nRunning ParFlow simulation...")
        overland.run(working_directory=output_dir)
        print(f"Simulation complete")
        
        passed = validate_fluxes(
            output_dir=output_dir,
            run_name=run_name,
            flow_method=flow_method,
            dx=dx,
            dy=dy,
            dt=dt,
            timestep=5
        )
        
        if not passed:
            all_passed = False
    
    print(f"\n{'#'*60}")
    print(f"FINAL RESULT - ROTATED GEOMETRY TEST")
    print(f"{'#'*60}")
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("PFTools calculation matches ParFlow in rotated geometry")
        sys.exit(0)
    else:
        print("\n✗ TESTS FAILED")
        print("If Y-direction now fails (like X did originally), this confirms")
        print("PFTools has a bug handling zero-slope channels in OverlandFlow")
        sys.exit(1)
