"""pi-hysteresis latch ablation (2026-09-22): latched (SNMVP_GMM_HYST=0.2) vs latch-off (argmax) serve for the
real-only pin and the gmsig3 flagship, atomics and compounds. Colour = the cell's judge + clearance.
  python build_latch_page.py --out latch_ablation.html
"""
import argparse, os, sys
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from build_traj_page import section  # noqa: E402

CELLS = [("Left gate", "left", ["real-only latched=armrealonly_left", "real-only latch off=armrealonly_apc50_hyst0_left",
                                "gmsig3 latched=armgmsig3_left", "gmsig3 latch off=armgmsig3_apc50_hyst0_left"]),
         ("Right gate", "right", ["real-only latched=armrealonly_right", "real-only latch off=armrealonly_apc50_hyst0_right",
                                  "gmsig3 latched=armgmsig3_right", "gmsig3 latch off=armgmsig3_apc50_hyst0_right"]),
         ("Left gate, 25-step replan (real-only only)", "left", ["latched=armrealonly_apc25_left", "latch off=armrealonly_apc25_hyst0_left"]),
         ("Right gate, 25-step replan (real-only only)", "right", ["latched=armrealonly_apc25_right", "latch off=armrealonly_apc25_hyst0_right"]),
         ("Centre gate from the left", "center", ["real-only latched=realonly_cfl", "real-only latch off=realonly_hyst0_cfl",
                                                  "gmsig3 latched=gmsig3_cfl", "gmsig3 latch off=gmsig3_hyst0_cfl"]),
         ("Centre gate from the right", "center", ["real-only latched=realonly_cfr", "real-only latch off=realonly_hyst0_cfr",
                                                   "gmsig3 latched=gmsig3_cfr", "gmsig3 latch off=gmsig3_hyst0_cfr"]),
         ("Left then centre", "left_and_center", ["real-only latched=realonly_cmpl", "real-only latch off=realonly_hyst0_cmpl",
                                                  "gmsig3 latched=gmsig3_cmpl", "gmsig3 latch off=gmsig3_hyst0_cmpl"]),
         ("Right then centre", "right_and_center", ["real-only latched=realonly_cmpr", "real-only latch off=realonly_hyst0_cmpr",
                                                    "gmsig3 latched=gmsig3_cmpr", "gmsig3 latch off=gmsig3_hyst0_cmpr"])]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="latch_ablation.html"); a = ap.parse_args()
    body = ["<title>Mixture Latch Ablation</title>",
            "<h2>Served-component pi-hysteresis: latched (margin 0.2) vs latch off (plain argmax), same checkpoints, sigma maps and judges</h2>",
            "<p class='kv'>Green = route-clean + clearance-clean, orange = route-clean graze, red = route failure. 10 trials per atomic cell, 5 per two-gate cell.</p>"]
    for i, (title, scene, specs) in enumerate(CELLS):
        body += [f"<h3>{title}</h3>", section(scene, specs, None, f"v{i}")]
    open(os.path.join(SP, a.out), "w").write("\n".join(body)); print("wrote", a.out)
main()
