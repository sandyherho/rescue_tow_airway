"""Run the whole pipeline: simulations, figures, reports, animations.

Usage::

    python scripts/run_all.py            # everything
    python scripts/run_all.py --no-anim  # skip the animations

Cached simulation results in ``outputs/cache`` are reused by the figure
and report scripts; the M1 risk maps skip environments already cached.
Delete ``outputs/cache`` to recompute from scratch.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

COMPUTE = ["compute_statics", "compute_riskmaps_m0", "compute_verification",
           "compute_riskmaps_m1", "compute_breathing",
           "compute_environments", "compute_buoyancy",
           "compute_sensitivity"]
FIGURES = ["fig01_model", "fig02_verification", "fig03_riskmaps",
           "fig04_heading", "fig05_buoyancy", "fig06_breathing",
           "fig07_coastal", "fig08_sensitivity", "fig09_decision"]
REPORTS = ["make_reports"]
ANIMS = ["anim01_tow", "anim02_oblique", "anim03_riskradar", "anim04_hull",
         "anim05_breathing"]


def run(name):
    t0 = time.time()
    print("== %s" % name, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, os.path.join(HERE, name + ".py")],
                   check=True, env=env)
    print("   %.0f s" % (time.time() - t0), flush=True)


def main():
    steps = COMPUTE + FIGURES + REPORTS
    if "--no-anim" not in sys.argv:
        steps += ANIMS
    for s in steps:
        run(s)


if __name__ == "__main__":
    main()
