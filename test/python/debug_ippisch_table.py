#!/usr/bin/env python3
"""
Diagnostic: compare Ippisch saturation and rel perm table values
against direct evaluation to isolate interpolation errors.
Mirrors the C code logic in SatComputeTable/VanGComputeTable.
"""

import numpy as np

alpha = 1.0
n = 1.5
m = 1.0 - 1.0 / n  # 1/3
h_s = 1.0 / alpha  # InverseAlpha mode
s_res = 0.2643
s_sat = 1.0
s_dif = s_sat - s_res
min_pressure_head = -300.0
num_sample_points = 20000

# Ippisch constants
Scs = (1.0 + (alpha * h_s) ** n) ** (-m)
opahn_s = 1.0 + (alpha * h_s) ** n
ahnm1_s = (alpha * h_s) ** (n - 1)
F_denom = 1.0 - ahnm1_s / opahn_s**m
ippisch_factor = np.sqrt(Scs) * F_denom**2

print(f"Ippisch constants:")
print(f"  h_s = {h_s}")
print(f"  Scs = {Scs:.10f}")
print(f"  F_denom = {F_denom:.10f}")
print(f"  ippisch_factor = {ippisch_factor:.10f}")
print()

# ============================================================
# Build saturation table (mirrors SatComputeTable)
# ============================================================
table_range = abs(min_pressure_head) - h_s
interval = table_range / (num_sample_points - 1)
x = np.array([h_s + i * interval for i in range(num_sample_points + 1)])
a_sat = np.zeros(num_sample_points + 1)  # S(h)
a_sat_der = np.zeros(num_sample_points + 1)  # dS/dh (positive mag)

for idx in range(num_sample_points + 1):
    Sc = (1.0 + (alpha * x[idx]) ** n) ** (-m)
    a_sat[idx] = s_dif * (Sc / Scs) + s_res
    a_sat_der[idx] = (
        (m * n * alpha * (alpha * x[idx]) ** (n - 1))
        * s_dif
        / (Scs * (1.0 + (alpha * x[idx]) ** n) ** (m + 1))
    )

# ============================================================
# Build rel perm table (mirrors VanGComputeTable with Ippisch)
# ============================================================
a_kr = np.zeros(num_sample_points + 1)
a_kr_der = np.zeros(num_sample_points + 1)

for idx in range(num_sample_points + 1):
    opahn = 1.0 + (alpha * x[idx]) ** n
    ahnm1 = (alpha * x[idx]) ** (n - 1)
    coeff = 1.0 - ahnm1 * opahn ** (-m)

    # Standard Kr
    kr_std = coeff**2 / opahn ** (m / 2)
    # Standard dKr/dh
    dkr_std = (
        2.0
        * (coeff / opahn ** (m / 2))
        * (
            (n - 1) * (alpha * x[idx]) ** (n - 2) * alpha * opahn ** (-m)
            - ahnm1 * m * opahn ** (-(m + 1)) * n * alpha * ahnm1
        )
        + coeff**2 * (m / 2) * opahn ** (-(m + 2) / 2) * n * alpha * ahnm1
    )

    a_kr[idx] = kr_std / ippisch_factor
    a_kr_der[idx] = dkr_std / ippisch_factor


# ============================================================
# Direct evaluation functions
# ============================================================
def sat_direct(head):
    if head <= h_s:
        return s_dif + s_res
    Sc = (1.0 + (alpha * head) ** n) ** (-m)
    return s_dif * (Sc / Scs) + s_res


def sat_der_direct(head):
    if head <= h_s:
        return 0.0
    return (
        (m * n * alpha * (alpha * head) ** (n - 1))
        * s_dif
        / (Scs * (1.0 + (alpha * head) ** n) ** (m + 1))
    )


def kr_direct(head):
    if head <= h_s:
        return 1.0
    opahn = 1.0 + (alpha * head) ** n
    ahnm1 = (alpha * head) ** (n - 1)
    kr_std = (1.0 - ahnm1 / opahn**m) ** 2 / opahn ** (m / 2)
    return kr_std / ippisch_factor


def kr_der_direct(head):
    if head <= h_s:
        return 0.0
    opahn = 1.0 + (alpha * head) ** n
    ahnm1 = (alpha * head) ** (n - 1)
    coeff = 1.0 - ahnm1 * opahn ** (-m)
    dkr_std = (
        2.0
        * (coeff / opahn ** (m / 2))
        * (
            (n - 1) * (alpha * head) ** (n - 2) * alpha * opahn ** (-m)
            - ahnm1 * m * opahn ** (-(m + 1)) * n * alpha * ahnm1
        )
        + coeff**2 * (m / 2) * opahn ** (-(m + 2) / 2) * n * alpha * ahnm1
    )
    return dkr_std / ippisch_factor


# ============================================================
# Compare table sample points vs direct eval
# ============================================================
print("=" * 70)
print("Table sample points vs direct eval (should be exact match)")
print("=" * 70)

max_sat_err = 0
max_sat_der_err = 0
max_kr_err = 0
max_kr_der_err = 0

for idx in [0, 1, 2, 10, 100, 1000, 5000, 10000, 15000, 19999, 20000]:
    if idx > num_sample_points:
        continue
    h = x[idx]
    s_tbl = a_sat[idx]
    s_dir = sat_direct(h)
    sd_tbl = a_sat_der[idx]
    sd_dir = sat_der_direct(h)
    kr_tbl = a_kr[idx]
    kr_dir = kr_direct(h)
    krd_tbl = a_kr_der[idx]
    krd_dir = kr_der_direct(h)

    max_sat_err = max(max_sat_err, abs(s_tbl - s_dir))
    max_sat_der_err = max(max_sat_der_err, abs(sd_tbl - sd_dir))
    max_kr_err = max(max_kr_err, abs(kr_tbl - kr_dir))
    max_kr_der_err = max(max_kr_der_err, abs(krd_tbl - krd_dir))

    if idx <= 2 or idx == 10:
        print(
            f"  h={h:10.4f}: S_tbl={s_tbl:.8f} S_dir={s_dir:.8f} "
            f"dS_tbl={sd_tbl:.8e} dS_dir={sd_dir:.8e}"
        )
        print(
            f"             Kr_tbl={kr_tbl:.8f} Kr_dir={kr_dir:.8f} "
            f"dKr_tbl={krd_tbl:.8e} dKr_dir={krd_dir:.8e}"
        )

print(f"\n  Max errors at sample points:")
print(f"    S:   {max_sat_err:.2e}")
print(f"    dS:  {max_sat_der_err:.2e}")
print(f"    Kr:  {max_kr_err:.2e}")
print(f"    dKr: {max_kr_der_err:.2e}")

# ============================================================
# Check boundary values
# ============================================================
print(f"\n{'='*70}")
print("Boundary checks")
print(f"{'='*70}")
print(f"  At h=h_s={h_s}:")
print(f"    S  = {a_sat[0]:.10f}  (should be {s_sat})")
print(f"    Kr = {a_kr[0]:.10f}  (should be 1.0)")
print(f"    dS = {a_sat_der[0]:.10e}  (should be finite)")
print(f"    dKr= {a_kr_der[0]:.10e}  (should be finite)")
print(f"  At h=h_s+eps (index 1, h={x[1]:.6f}):")
print(f"    S  = {a_sat[1]:.10f}")
print(f"    Kr = {a_kr[1]:.10f}")

# ============================================================
# Hermite spline interpolation test at midpoints
# ============================================================
print(f"\n{'='*70}")
print("Spline interpolation at midpoints vs direct eval")
print(f"{'='*70}")


# Build Fritsch-Carlson spline slopes (same as C code)
def build_fc_spline(x, a, num_pts):
    """Fritsch-Carlson monotonic Hermite spline slopes."""
    d = np.zeros(num_pts + 1)
    h = np.zeros(num_pts)
    delta = np.zeros(num_pts)

    for i in range(num_pts):
        h[i] = x[i + 1] - x[i]
        delta[i] = (a[i + 1] - a[i]) / h[i]

    d[0] = delta[0]
    d[num_pts] = delta[num_pts - 1]
    for i in range(1, num_pts):
        d[i] = (delta[i - 1] + delta[i]) / 2

    for i in range(num_pts):
        if delta[i] == 0:
            d[i] = 0
            d[i + 1] = 0
        else:
            al = d[i] / delta[i]
            be = d[i + 1] / delta[i]
            mag = al**2 + be**2
            if mag > 9.0:
                d[i] = 3 * al * delta[i] / mag
                d[i + 1] = 3 * be * delta[i] / mag
    return d


def hermite_eval(xv, x, a, d, interval, h_s_val, num_pts):
    """Evaluate Hermite cubic at xv (mirrors SatLookupSpline)."""
    if xv <= h_s_val:
        return None  # air-entry zone
    pt = int(np.floor((xv - h_s_val) / interval))
    if pt >= num_pts:
        pt = num_pts - 1
    x0 = x[pt]
    x1 = x[pt + 1]
    t = (xv - x0) / (x1 - x0)
    return (
        (2 * t**3 - 3 * t**2 + 1) * a[pt]
        + (t**3 - 2 * t**2 + t) * (x1 - x0) * d[pt]
        + (-2 * t**3 + 3 * t**2) * a[pt + 1]
        + (t**3 - t**2) * (x1 - x0) * d[pt + 1]
    )


# Build splines for S and dS
d_sat = build_fc_spline(x, a_sat, num_sample_points)
d_sat_der = build_fc_spline(x, a_sat_der, num_sample_points)
d_kr = build_fc_spline(x, a_kr, num_sample_points)
d_kr_der = build_fc_spline(x, a_kr_der, num_sample_points)

# Test at many random points
test_heads = np.concatenate(
    [
        np.linspace(h_s + 0.001, h_s + 1.0, 100),  # near air-entry
        np.linspace(2.0, 10.0, 100),  # moderate
        np.linspace(10.0, 100.0, 100),  # mid-range
        np.linspace(100.0, 299.0, 100),  # dry end
    ]
)

max_errs = {"S": 0, "dS": 0, "Kr": 0, "dKr": 0}
worst = {"S": None, "dS": None, "Kr": None, "dKr": None}

for head in test_heads:
    s_spline = hermite_eval(head, x, a_sat, d_sat, interval, h_s, num_sample_points)
    sd_spline = hermite_eval(
        head, x, a_sat_der, d_sat_der, interval, h_s, num_sample_points
    )
    kr_spline = hermite_eval(head, x, a_kr, d_kr, interval, h_s, num_sample_points)
    krd_spline = hermite_eval(
        head, x, a_kr_der, d_kr_der, interval, h_s, num_sample_points
    )

    if s_spline is None:
        continue

    s_dir = sat_direct(head)
    sd_dir = sat_der_direct(head)
    kr_dir = kr_direct(head)
    krd_dir = kr_der_direct(head)

    for key, spl, dr in [
        ("S", s_spline, s_dir),
        ("dS", sd_spline, sd_dir),
        ("Kr", kr_spline, kr_dir),
        ("dKr", krd_spline, krd_dir),
    ]:
        err = abs(spl - dr)
        if err > max_errs[key]:
            max_errs[key] = err
            worst[key] = (head, spl, dr)

print(f"  Max spline interpolation errors (400 test points):")
for key in ["S", "dS", "Kr", "dKr"]:
    h, spl, dr = worst[key]
    rel_err = abs(spl - dr) / max(abs(dr), 1e-30)
    print(
        f"    {key:3s}: abs={max_errs[key]:.2e}  rel={rel_err:.2e}  "
        f"at h={h:.4f}  (spline={spl:.8e}, direct={dr:.8e})"
    )

# ============================================================
# Check specific problematic region: near h_s
# ============================================================
print(f"\n{'='*70}")
print("Detailed near-air-entry comparison (h_s to h_s + 0.1)")
print(f"{'='*70}")
for head in np.linspace(h_s, h_s + 0.1, 11):
    s_dir = sat_direct(head)
    sd_dir = sat_der_direct(head)
    kr_dir = kr_direct(head)

    if head > h_s:
        s_spl = hermite_eval(head, x, a_sat, d_sat, interval, h_s, num_sample_points)
        kr_spl = hermite_eval(head, x, a_kr, d_kr, interval, h_s, num_sample_points)
        s_err = abs(s_spl - s_dir)
        kr_err = abs(kr_spl - kr_dir)
    else:
        s_spl = s_sat
        kr_spl = 1.0
        s_err = 0
        kr_err = 0

    print(
        f"  h={head:.4f}: S_dir={s_dir:.8f} S_spl={s_spl:.8f} err={s_err:.2e}  "
        f"Kr_dir={kr_dir:.8f} Kr_spl={kr_spl:.8f} err={kr_err:.2e}"
    )
