# -----------------------------------------------------------------------------
# Gate for adaptive-dt Layer 3a (Solver.AdaptiveDt.RateControl).
#
# Reuses the Layer-4 gate scenario (adaptive_dt_onset.py): a flat domain,
# quiet for 4 forcing intervals, then a strong storm arriving at a forcing
# boundary through transient evap_trans, entered at a base dt of one interval.
#
# RateControl is the a-priori counterpart of ErrorControl: it bounds dt from
# the last accepted step's tolerance-weighted state-change rate, BEFORE the
# solve.  It is deliberately blind to incoming forcing (entry protection is
# OnsetControl's one-shot job), so the gate runs the intended pairing,
# RateControl + OnsetControl, and asserts:
#   (a) quiet when the state is quiet -- no 'r' bound and full dt before the
#       storm (the pre-storm column is static, so the rate is ~0);
#   (b) the storm entry is capped by the onset layer ('o'), and from the step
#       after it the rate limiter takes over ('r' steps) and stays engaged
#       through the event;
#   (c) regulation: with the pair on, no solve after the onset-capped entry
#       costs more than half the blind entry (which sits at the MaxIter
#       cliff), and there are no convergence failures.
# -----------------------------------------------------------------------------

import sys
from parflow.tools.fs import mkdir, get_absolute_path

# reuse the onset gate's scenario, forcing writer, and log parsers
from adaptive_dt_onset import (build_run, write_forcing, parse_kinsol,
                               parse_outlog, STORM_FILE, MAX_ITER)


def run_case(name, rate=False):
    outdir = get_absolute_path(f"test_output/{name}")
    mkdir(outdir)
    write_forcing(outdir)
    r = build_run(name)
    if rate:
        r.Solver.AdaptiveDt = True
        r.Solver.AdaptiveDt.RateControl = True
        r.Solver.AdaptiveDt.OnsetControl = True
        r.Solver.AdaptiveDt.OnsetControl.FillHorizon = 1.0
        # economy weights for the rate bound (shared with ErrorControl)
        r.Solver.AdaptiveDt.ErrorControl.RelTol = 1.0e-2
        r.Solver.AdaptiveDt.ErrorControl.AbsTol = 1.0e-3
    r.run(working_directory=outdir)
    nonlin, attempts, per_att = parse_kinsol(f"{outdir}/{name}.out.kinsol.log")
    rows = parse_outlog(f"{outdir}/{name}.out.log")
    steps = [row for row in rows if row[2] != "i"]
    failures = attempts - len(steps)
    storm_atts = [(t, its) for (t, its) in per_att if t > STORM_FILE]
    entry_its = storm_atts[0][1]
    # max solve cost in the rate-limited regime (everything after the
    # onset-capped entry step, which is Layer 4's business)
    max_rest = max((its for (_, its) in storm_atts[1:]), default=0)
    return nonlin, failures, entry_its, max_rest, rows


def main():
    b_nl, b_fail, b_entry, b_max, b_rows = run_case("adapt_rate_off")
    r_nl, r_fail, r_entry, r_max, r_rows = run_case("adapt_rate_on", rate=True)

    pre = [row for row in r_rows if row[0] <= STORM_FILE
           and row[2] != "i"]
    r_pre = [row for row in pre if row[2] == "r"]
    r_storm = [row for row in r_rows if row[0] > STORM_FILE and row[2] == "r"]
    first_storm_row = next(row for row in r_rows if row[0] > STORM_FILE)

    print(f"blind : {b_nl} Newton, {b_fail} failures, "
          f"entry {b_entry}/{MAX_ITER} its")
    print(f"rate+onset : {r_nl} Newton in {len(r_rows)} steps, "
          f"{r_fail} failures, max post-entry step {r_max} its, "
          f"{len(r_storm)} 'r' steps, first storm step "
          f"dt={first_storm_row[1]:.3f} '{first_storm_row[2]}'")

    quiet_when_quiet = len(r_pre) == 0 and all(
        abs(row[1] - 1.0) < 1e-9 for row in pre)
    entry_capped = first_storm_row[2] == "o"
    stays_engaged = len(r_storm) >= 3
    blind_at_cliff = b_entry >= MAX_ITER - 1
    regulated = r_max <= b_entry // 2
    clean = r_fail == 0

    print(f"no 'r' and full dt before the storm : {quiet_when_quiet}")
    print(f"entry step capped by onset ('o')    : {entry_capped}")
    print(f"rate limiter engaged in the storm   : {stays_engaged}")
    print(f"blind entry at the MaxIter cliff    : {blind_at_cliff}")
    print(f"post-entry solves <= half blind entry : {regulated}")
    print(f"no convergence failures             : {clean}")

    if (quiet_when_quiet and entry_capped and stays_engaged
            and blind_at_cliff and regulated and clean):
        print("adaptive_dt_rate : PASSED")
    else:
        print("adaptive_dt_rate : FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
