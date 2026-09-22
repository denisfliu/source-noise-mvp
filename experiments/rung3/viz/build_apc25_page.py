"""Real-only pin at a 50-step vs 25-step replan, left and right cells (2026-09-22).
  python build_apc25_page.py --out realonly_apc25.html
"""
import argparse, os, sys
SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
from build_traj_page import section  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="realonly_apc25.html"); a = ap.parse_args()
    body = ["<title>Real-Only Pin, Replan Interval</title>",
            "<h2>Real-only pin: replan every 50 steps vs every 25 steps (10 trials per cell, 400 steps each)</h2>",
            "<p class='kv'>Same checkpoint, server, sigma map and scorer. Green = route-clean + clearance-clean, orange = route-clean graze, "
            "red = route failure. The judge's goal box (yellow) is entered by the pilot's own demonstrations only 23/50 (left) and 9/50 (right).</p>",
            "<h3>Left gate</h3>",
            section("left", ["APC 50=armrealonly_left", "APC 25=armrealonly_apc25_left"], None, "vL"),
            "<h3>Right gate</h3>",
            section("right", ["APC 50=armrealonly_right", "APC 25=armrealonly_apc25_right"], None, "vR")]
    open(os.path.join(SP, a.out), "w").write("\n".join(body)); print("wrote", a.out)
main()
