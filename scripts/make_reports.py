"""Plain-text reports of every number quoted from the simulations."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rescue_tow_airway.body import Rescue, build_body  # noqa: E402
from rescue_tow_airway.environments import (e5_beach, e6_hull,  # noqa
                                            suite)
from rescue_tow_airway.io import has_cache, load_cache, write_report  # noqa
from rescue_tow_airway.linear import LinearModel  # noqa: E402

STRATS = ("fixed", "clear", "crest", "oracle")


def report_model(lines):
    body = build_body()
    lines += ["MODEL DIVER (default rescue configuration)",
              "  total mass with equipment     %.1f kg" % body.total_mass,
              "  primitives / strips           %d / %d"
              % (body.xi.size, body.strip_xi.size)]
    st = load_cache("statics")
    v = st["volumes"]
    for tow in ("collar", "valve"):
        i12 = int(np.argmin(np.abs(v - 12)))
        lines.append("  %-6s tow, 12 L, 0.5 m/s: trim %.1f deg, airway "
                     "clearance %.1f cm" % (tow, np.degrees(
                         st["pitch_%s_05" % tow][i12]),
                         100 * st["clear_%s_05" % tow][i12]))
    lm = LinearModel(body, 0.5)
    wn, z, _ = lm.eigen()
    lines.append("  natural periods (s): " + ", ".join(
        "%.2f (zeta %.2f)" % (2 * np.pi / w, zz) for w, zz in zip(wn, z)))
    for k, lab in enumerate(st["release_cases"]):
        fin = np.degrees(st["release_roll"][k, -1] + st["release_head"][k,
                                                                        -1])
        lines.append("  still-water release, %-22s final face roll %6.1f "
                     "deg, clearance %+.3f m" % (lab, fin,
                                                 st["release_clear"][k,
                                                                     -1]))
    try:
        build_body(Rescue(bcd_liters=0.0))
        from rescue_tow_airway.dynamics import equilibrium
        equilibrium(build_body(Rescue(bcd_liters=0.0)))
        lines.append("  empty BCD: floats")
    except RuntimeError:
        lines.append("  empty BCD: no floating equilibrium (sinks)")
    lines.append("")


def report_verification(lines):
    if not has_cache("verification"):
        return
    v = load_cache("verification")
    lines += ["VERIFICATION (E2 spectrum, long-crested, 0.5 m/s)",
              "  Hs    heading  sigma_r M0  sigma_r M1  ratio   "
              "P_awash M0  P_awash M1"]
    for i, hs in enumerate(v["hs"]):
        for j, h in enumerate(v["headings"]):
            m0, m1 = v["m0"][i, j], v["m1"][i, j]
            lines.append("  %.2f  %5.0f    %.4f      %.4f      %.3f   "
                         "%.4f      %.4f" % (hs, h, m0[1], m1[1],
                                             m1[1] / m0[1], m0[3], m1[3]))
    lines.append("  time-step convergence (dt, sigma_r, P_awash, rms "
                 "difference to dt=0.02):")
    for dt, c, e in zip(v["dts"], v["conv"], v["conv_err"]):
        lines.append("    %.2f  %.5f  %.4f  %.2e" % (dt, c[0], c[1], e))
    lines.append("")


def report_maps(lines):
    lines += ["RISK MAPS (probability that a breath meets water)"]
    for sea in suite():
        tag = sea.name.replace("-", "m").replace("+", "p")
        g = sea.spectral_grid()
        hs_local = 4 * np.sqrt(0.5 * g["a2"].sum())
        m = load_cache("m0_map_" + tag)
        lines.append("  %s %s: local Hs %.2f m" % (sea.name, sea.label,
                                                   hs_local))
        i = int(np.argmin(np.abs(m["speeds"] - 0.5)))
        for h in (0.0, 45.0, 90.0, 135.0, 180.0):
            j = int(np.argmin(np.abs(m["headings"] - h)))
            lines.append("    M0 0.5 m/s heading %5.1f: P_breath %.4f "
                         "P_awash %.4f T_enc %.2f s" % (
                             h, m["p_breath"][i, j], m["p_awash"][i, j],
                             m["tz_enc"][i, j]))
        if has_cache("m1_map_" + tag):
            m1 = load_cache("m1_map_" + tag)
            for a, u in enumerate(m1["speeds"]):
                for b, h in enumerate(m1["headings"]):
                    lines.append("    M1 %.1f m/s heading %5.1f: P_breath "
                                 "%.4f P_awash %.4f max face roll %.1f deg"
                                 % (u, h, m1["p_breath"][a, b],
                                    m1["p_awash"][a, b],
                                    np.degrees(m1["roll_max"][a, b])))
    lines.append("")


def report_breathing(lines):
    lines += ["BREATH TIMING (M1, 32 tows of 300 s; failure probability, "
              "mean interval between breaths)"]
    for tag in ("E1", "E2", "E3", "E7", "E4m"):
        for sc in ("tow", "hold"):
            name = "breath_%s_%s" % (tag, sc)
            if not has_cache(name):
                continue
            d = load_cache(name)
            parts = ["%s %.3f/%.1fs" % (s, d["fail_" + s],
                                        d["interval_" + s])
                     for s in STRATS]
            lines.append("  %-4s %-4s P_awash %.4f rate %.4f/s  Fano(10,"
                         "20,30,60 s) %s  D %.2f bits" % (
                             tag, sc, d["p_awash"], d["rate"],
                             " ".join("%.2f" % f for f in d["fano"]),
                             d["info"]))
            lines.append("            " + "  ".join(parts))
            if np.all(np.isfinite(d["gev"])):
                lines.append("            longest immersion per tow: median "
                             "%.2f s, max %.2f s; GEV shape %.2f" % (
                                 np.median(d["longest"]), d["longest"].max(),
                                 d["gev"][0]))
    lines.append("")


def report_buoyancy(lines):
    if not has_cache("buoyancy"):
        return
    b = load_cache("buoyancy")
    lines += ["BCD AIR, TOW POINT, HEAD SUPPORT (E2, 0.5 m/s, M0 P_breath "
              "at 180 / 90 deg)"]
    for tow in ("collar", "valve"):
        for kn in (2, 40):
            arr = b["m0_%s_k%02d" % (tow, kn)]
            for vv, row in zip(b["vol_m0"], arr):
                if vv in (8, 10, 12, 16, 20, 24):
                    lines.append("  %-6s k=%2d  %4.0f L: %.4f / %.4f" % (
                        tow, kn, vv, row[2, 1], row[1, 1]))
    for kn in (2, 40):
        arr = b["m1_k%02d" % kn]
        for vv, row in zip(b["vol_m1"], arr):
            lines.append("  M1 collar k=%2d %4.0f L: P_breath 90 deg %.4f, "
                         "180 deg %.4f" % (kn, vv, row[0, 1], row[1, 1]))
    lines.append("")


def report_coastal(lines):
    if not has_cache("environments"):
        return
    e = load_cache("environments")
    lines += ["BEACH EXIT (E5, 0.5 m/s shoreward)"]
    for d, row in zip(e["e5_depth_m0"], e["e5_m0"]):
        lines.append("  M0 h %5.1f m: Hs %.2f m, P_breath %.4f" % (
            d, row[0], row[3]))
    for d, row in zip(e["e5_depth_m1"], e["e5_m1"]):
        lines.append("  M1 h %5.1f m: P_awash %.4f, P_breath %.4f" % (
            d, row[0], row[2]))
    sat = e5_beach(2.0)
    sat.spectral_grid()
    lines.append("  depth-limited saturation active at 2 m: %s" %
                 bool(sat.extra.get("saturated", False)))
    lines += ["BOAT HULL (E6, stopped, R = %.1f)" % e6_hull().wall.refl]
    ratio = e["e6_var_eta"] / e["e6_var_open"]
    dist = e["e6_dist_m0"]
    k = np.argmin(ratio[dist > 2.0]) + np.sum(dist <= 2.0)
    lines.append("  surface variance ratio at the hull %.2f; first minimum "
                 "%.2f at %.2f m" % (ratio[0], ratio[k], dist[k]))
    xa = build_body().airway[0]
    for name in ("normal", "parallel"):
        m1 = e["e6_m1_" + name]
        for d, row in zip(e["e6_dist_m1"], m1):
            if name == "normal" and d - xa < 0.15:
                lines.append("  M1 %-8s d %.1f m: excluded (airway within "
                             "0.15 m of the hull)" % (name, d))
                continue
            lines.append("  M1 %-8s d %.1f m: P_awash %.4f P_breath %.4f" % (
                name, d, row[0], row[2]))
    lines.append("")


def report_sensitivity(lines):
    if not has_cache("sensitivity"):
        return
    s = load_cache("sensitivity")
    lines += ["SOBOL INDICES (M0, E2, 0.5 m/s; outputs: P_breath 180 deg, "
              "P_breath 90 deg, log10 ratio)"]
    for i, n in enumerate(s["names"]):
        lines.append("  %-9s S1 %s  ST %s" % (n, np.round(s["s1"][i], 3),
                                              np.round(s["st"][i], 3)))
    y = s["sample_y"]
    lines.append("  across beats into in %.1f%% of %d random coefficient "
                 "sets" % (100 * np.mean(y[:, 2] > 0), y.shape[0]))
    lines.append("")


def main():
    lines = ["Airway submersion during a surface tow: numerical results",
             "All results are from idealized models without observational "
             "data.", ""]
    report_model(lines)
    report_verification(lines)
    report_maps(lines)
    report_breathing(lines)
    report_buoyancy(lines)
    report_coastal(lines)
    report_sensitivity(lines)
    write_report("results", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
