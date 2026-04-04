"""
Compare overland flow results: standard kinematic vs diffusion correction.

Reads ParFlow output and plots:
1. Pressure (ponding depth) time series at selected cells
2. Spatial maps of ponding depth at final timestep
3. Overland flux comparison (if qx/qy output available)
4. Hydrology-package flux comparison: kinematic vs kinematic+diffusive
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from parflow.tools.io import read_pfb

# Also use the hydrology module to compute fluxes from pressure
try:
    from parflow.tools.hydrology import (
        calculate_overland_fluxes,
        calculate_overland_flow,
    )

    HAS_HYDROLOGY = True
except ImportError:
    HAS_HYDROLOGY = False

from parflow.tools.fs import get_absolute_path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

test_output_dir = get_absolute_path("test_output")
n_timesteps = 11
dx = 10.0
dy = 10.0
nx = 5
ny = 5

# Runs to compare
runs = {
    "Kinematic (no correction)": {
        "dir": "TiltedV_OverlandKin",
        "name": "TiltedV_OverlandKin",
        "existing": True,
    },
    "DiffCorr Picard (alpha=1.0)": {
        "dir": "TiltedV_OverlandKin_DiffCorr_Picard",
        "name": "TiltedV_OverlandKin_DiffCorr_Picard",
    },
    "DiffCorr FullNewton (alpha=1.0)": {
        "dir": "TiltedV_OverlandKin_DiffCorr_FullNewton",
        "name": "TiltedV_OverlandKin_DiffCorr_FullNewton",
    },
    "DiffCorr Picard (alpha=0.5)": {
        "dir": "TiltedV_OverlandKin_DiffCorr_Alpha05",
        "name": "TiltedV_OverlandKin_DiffCorr_Alpha05",
    },
}

# Slopes and Manning's for hydrology module comparison
slopex = np.zeros((ny, nx))
slopex[:, :2] = -0.01  # left slope
slopex[:, 2] = 0.01  # channel
slopex[:, 3:] = 0.01  # right slope
slopey = np.full((ny, nx), 0.01)
mannings = np.full((ny, nx), 3.0e-6)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_run(run_info):
    """Load pressure time series for a run."""
    run_dir = os.path.join(test_output_dir, run_info["dir"])
    run_name = run_info["name"]

    if not os.path.exists(run_dir):
        print(f"  Skipping {run_name}: directory not found")
        return None

    pressures = []
    for t in range(n_timesteps):
        ts = str(t).rjust(5, "0")
        fname = os.path.join(run_dir, f"{run_name}.out.press.{ts}.pfb")
        if os.path.exists(fname):
            pressures.append(read_pfb(fname))
        else:
            print(f"  Missing: {fname}")
            return None

    # Try to load qx/qy output
    qx_data = []
    qy_data = []
    for t in range(n_timesteps):
        ts = str(t).rjust(5, "0")
        qx_fname = os.path.join(run_dir, f"{run_name}.out.overlandsumx.{ts}.pfb")
        qy_fname = os.path.join(run_dir, f"{run_name}.out.overlandsumy.{ts}.pfb")
        if os.path.exists(qx_fname):
            qx_data.append(read_pfb(qx_fname))
        if os.path.exists(qy_fname):
            qy_data.append(read_pfb(qy_fname))

    return {
        "pressures": pressures,
        "qx": qx_data if qx_data else None,
        "qy": qy_data if qy_data else None,
    }


print("Loading runs...")
data = {}
for label, info in runs.items():
    print(f"  {label}")
    d = load_run(info)
    if d is not None:
        data[label] = d

if not data:
    print("No data found. Run the tests first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Plot 1: Ponding depth time series at selected cells
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
cells = [
    ("Left slope (1,2)", 2, 1),
    ("Channel (2,2)", 2, 2),
    ("Right slope (3,2)", 2, 3),
]

times = np.arange(n_timesteps) * 0.1  # BaseUnit * DumpInterval steps

for ax, (cell_label, row, col) in zip(axes, cells):
    for label, d in data.items():
        p_ts = [np.maximum(p[-1, row, col], 0.0) for p in d["pressures"]]
        ax.plot(times, p_ts, marker="o", markersize=3, label=label)
    ax.set_title(cell_label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Ponding depth")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Ponding Depth Time Series — Tilted-V Catchment")
plt.tight_layout()
plt.savefig(os.path.join(test_output_dir, "diffcorr_timeseries.png"), dpi=150)
print(f"Saved: {test_output_dir}/diffcorr_timeseries.png")

# ---------------------------------------------------------------------------
# Plot 2: Spatial maps at final timestep
# ---------------------------------------------------------------------------

n_runs = len(data)
fig, axes = plt.subplots(1, n_runs, figsize=(4 * n_runs, 3.5))
if n_runs == 1:
    axes = [axes]

for ax, (label, d) in zip(axes, data.items()):
    p_final = np.maximum(d["pressures"][-1][-1, :, :], 0.0)
    im = ax.imshow(
        p_final, origin="lower", aspect="equal", extent=[0, nx * dx, 0, ny * dy]
    )
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.colorbar(im, ax=ax, label="Ponding (m)")

fig.suptitle("Final Timestep Ponding Depth")
plt.tight_layout()
plt.savefig(os.path.join(test_output_dir, "diffcorr_spatial.png"), dpi=150)
print(f"Saved: {test_output_dir}/diffcorr_spatial.png")

# ---------------------------------------------------------------------------
# Plot 3: Picard vs FullNewton difference (should be ~0)
# ---------------------------------------------------------------------------

picard_key = "DiffCorr Picard (alpha=1.0)"
newton_key = "DiffCorr FullNewton (alpha=1.0)"
if picard_key in data and newton_key in data:
    fig, ax = plt.subplots(figsize=(6, 4))
    diffs = []
    for t in range(n_timesteps):
        p_pic = data[picard_key]["pressures"][t]
        p_new = data[newton_key]["pressures"][t]
        diffs.append(np.max(np.abs(p_pic - p_new)))
    ax.plot(times, diffs, "k-o")
    ax.set_xlabel("Time")
    ax.set_ylabel("Max |Picard - FullNewton|")
    ax.set_title("Picard vs FullNewton Convergence Check")
    ax.set_yscale("log" if max(diffs) > 0 else "linear")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(test_output_dir, "diffcorr_picard_vs_newton.png"), dpi=150)
    print(f"Saved: {test_output_dir}/diffcorr_picard_vs_newton.png")
else:
    print("Skipping Picard vs FullNewton comparison (missing data)")

# ---------------------------------------------------------------------------
# Plot 4: Hydrology module flux comparison
# ---------------------------------------------------------------------------

if HAS_HYDROLOGY:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Use final timestep pressure from each available run
    methods = []
    if "Kinematic (no correction)" in data:
        methods.append(("Kinematic", "Kinematic (no correction)", "OverlandKinematic"))
    if picard_key in data:
        methods.append(("Kin+Diffusive", picard_key, "OverlandKinematicDiffusive"))

    for col, (method_label, data_key, flow_method) in enumerate(methods):
        p_final = data[data_key]["pressures"][-1]
        mask = np.ones_like(p_final)

        qeast, qnorth = calculate_overland_fluxes(
            p_final,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method=flow_method,
            epsilon=1e-5,
            mask=mask,
        )

        im0 = axes[0, col].imshow(
            qeast[:, 1:-1], origin="lower", aspect="equal", cmap="RdBu_r"
        )
        axes[0, col].set_title(f"qeast — {method_label}")
        plt.colorbar(im0, ax=axes[0, col])

        im1 = axes[1, col].imshow(
            qnorth[1:-1, :], origin="lower", aspect="equal", cmap="RdBu_r"
        )
        axes[1, col].set_title(f"qnorth — {method_label}")
        plt.colorbar(im1, ax=axes[1, col])

    # Difference if both exist
    if len(methods) >= 2:
        p_kin = data[methods[0][1]]["pressures"][-1]
        p_diff = data[methods[1][1]]["pressures"][-1]
        mask = np.ones_like(p_kin)

        qe_kin, qn_kin = calculate_overland_fluxes(
            p_kin,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method="OverlandKinematic",
            epsilon=1e-5,
            mask=mask,
        )
        qe_dif, qn_dif = calculate_overland_fluxes(
            p_diff,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method="OverlandKinematicDiffusive",
            epsilon=1e-5,
            mask=mask,
        )

        im2 = axes[0, 2].imshow(
            qe_dif[:, 1:-1] - qe_kin[:, 1:-1],
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
        )
        axes[0, 2].set_title("qeast difference")
        plt.colorbar(im2, ax=axes[0, 2])

        im3 = axes[1, 2].imshow(
            qn_dif[1:-1, :] - qn_kin[1:-1, :],
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
        )
        axes[1, 2].set_title("qnorth difference")
        plt.colorbar(im3, ax=axes[1, 2])

    fig.suptitle("Hydrology Module: Overland Flux Comparison (Final Timestep)")
    plt.tight_layout()
    plt.savefig(os.path.join(test_output_dir, "diffcorr_hydrology_fluxes.png"), dpi=150)
    print(f"Saved: {test_output_dir}/diffcorr_hydrology_fluxes.png")

# ---------------------------------------------------------------------------
# Plot 5: Outlet flow time series from hydrology module
# ---------------------------------------------------------------------------

if HAS_HYDROLOGY:
    # The tilted-V drains in the +y direction (slopey=0.01).
    # The outlet is at the y_upper boundary (last row, max y).
    # Total outflow = sum of qnorth at the southern boundary of the domain
    # (qnorth[-1, :] = flux leaving through the y_upper face).

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Kinematic flow method for all runs
    for label, d in data.items():
        outflow_kin = []
        for t in range(n_timesteps):
            p = d["pressures"][t]
            mask = np.ones_like(p)
            qeast, qnorth = calculate_overland_fluxes(
                p,
                slopex,
                slopey,
                mannings,
                dx,
                dy,
                flow_method="OverlandKinematic",
                epsilon=1e-5,
                mask=mask,
            )
            # Outflow at y_upper: positive qnorth at the last row
            outlet_q = np.sum(np.maximum(0, qnorth[-1, :]))
            outflow_kin.append(outlet_q)
        ax1.plot(times, outflow_kin, marker="o", markersize=3, label=label)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Outlet flow (kinematic calc)")
    ax1.set_title("Outlet Hydrograph — Kinematic Hydrology")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Now compare kinematic vs diffusive hydrology calc for each run
    for label, d in data.items():
        outflow_diff = []
        for t in range(n_timesteps):
            p = d["pressures"][t]
            mask = np.ones_like(p)
            qeast, qnorth = calculate_overland_fluxes(
                p,
                slopex,
                slopey,
                mannings,
                dx,
                dy,
                flow_method="OverlandKinematicDiffusive",
                epsilon=1e-5,
                mask=mask,
            )
            outlet_q = np.sum(np.maximum(0, qnorth[-1, :]))
            outflow_diff.append(outlet_q)
        ax2.plot(times, outflow_diff, marker="o", markersize=3, label=label)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Outlet flow (diffusive calc)")
    ax2.set_title("Outlet Hydrograph — Diffusive Hydrology")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Outlet Flow Time Series (Hydrology Module)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(test_output_dir, "diffcorr_outlet_hydrograph.png"), dpi=150
    )
    print(f"Saved: {test_output_dir}/diffcorr_outlet_hydrograph.png")

print("\nDone.")
