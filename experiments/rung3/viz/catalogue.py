"""Shared catalogue of paper-relevant rollout cells and the automatic-verdict parser
(used by build_gradebook.py and build_sketchreview.py)."""
import glob
import os
import re

RD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = "/home/dfliu/ctxrun"


# ---------------------------------------------------------------- auto verdicts
def parse_all_scores():
    """traj stem -> {'judge': line, 'clear': line, 'scene': str, 'header': str,
    'minclr': float, 'clrstep': int}. Files processed oldest-first so re-scores win."""
    out = {}
    files = sorted(glob.glob(f"{RUN}/*_scores.txt"), key=os.path.getmtime)
    rx_traj = re.compile(r"^(traj_\S+?)\.npy\s+(.*)$")
    rx_clr = re.compile(r"min-clearance ([\d.]+) m @step\s+(\d+)")
    for f in files:
        header, scene = "", None
        for raw in open(f, errors="replace"):
            line = raw.rstrip()
            if line.startswith("== ") and not re.match(r"== \d+/\d+ ", line):
                header = line[3:]
                continue
            if line.startswith("scene="):
                scene = line.split("=", 1)[1].split()[0]
                continue
            m = rx_traj.match(line)
            if not m:
                continue
            stem, rest = m.group(1), m.group(2).strip()
            d = out.setdefault(stem, {})
            if "min-clearance" in rest:
                d["clear"] = rest
                if scene:
                    d["scene"] = scene
                mc = rx_clr.search(rest)
                if mc:
                    d["minclr"], d["clrstep"] = float(mc.group(1)), int(mc.group(2))
            else:
                d["judge"] = rest
                d["header"] = header
    return out


AUTO = parse_all_scores()

# ---------------------------------------------------------------- cell catalogue
ARMS = [("xswap", "Full method (xswap), seed 42", "armxswap", "xswap"),
        ("xswaps7", "Full method (xswap), seed 7", "armxswaps7", "xswaps7"),
        ("scr3", "pi0 baseline (scratch3), seed 42", "armscr3", "scr3"),
        ("scr3s7", "pi0 baseline (scratch3), seed 7", "armscr3s7", "scr3s7"),
        ("gmsig3", "w/o cross-domain (gmsig3), seed 42", "armgmsig3", "gmsig3"),
        ("gmsig3s7", "w/o cross-domain (gmsig3), seed 7", "armgmsig3s7", "gmsig3s7"),
        ("synthonly", "sim-only training (synthonly), seed 42", "armsynthonly", "synthonly"),
        ("nosig", "w/o uncertainty (nosig), seed 42", "armnosig", "nosig")]
TASK = {"left": ("Left gate", "left"), "right": ("Right gate", "right"),
        "cfl": ("Center from left", "center"), "cfr": ("Center from right", "center"),
        "cmpl": ("Compound L->C", "left_and_center"), "cmpr": ("Compound R->C", "right_and_center")}

CELLS = []  # dicts: id, label, arm, campaign, scene, prefix, sketch
for arm, arm_label, ap, cp in ARMS:
    for t in ("left", "right", "cfl", "cfr"):
        pre = f"{ap}_{t}" if t in ("left", "right") else f"{cp}_{t}"
        CELLS.append(dict(id=pre, label=TASK[t][0], arm=arm_label, campaign="Atomic tasks",
                          scene=TASK[t][1], prefix=pre, sketch=None))
    for t in ("cmpl", "cmpr"):
        pre = f"{cp}_{t}"
        CELLS.append(dict(id=pre, label=TASK[t][0] + " (unguided)", arm=arm_label,
                          campaign="Compounds, unguided", scene=TASK[t][1], prefix=pre, sketch=None))

SK = lambda n: f"{RD}/{n}.json"
SKETCH_ROWS = [
    # prefix, label, checkpoint, scene, sketch json
    ("skd_cmpl", "Hand-drawn L->C", "gmsig3 s42", "left_and_center", "sketch_cmpl_denis"),
    ("skdns1_cmpl", "Hand-drawn L->C, rollout seed 1", "gmsig3 s42", "left_and_center", "sketch_cmpl_denis"),
    ("skd_cmpr", "Hand-drawn R->C round 1", "gmsig3 s42", "right_and_center", "sketch_cmpr_denis"),
    ("skdns1_cmpr", "Hand-drawn R->C round 1, rollout seed 1", "gmsig3 s42", "right_and_center", "sketch_cmpr_denis"),
    ("skdr1_cmpr", "Hand-drawn R->C r1", "gmsig3 s42", "right_and_center", "sketch_cmpr_denis_r1"),
    ("skcmpl", "Corrective (machine) sketch L->C", "gmsig3 s42", "left_and_center", "sketch_cmpl"),
    ("skm4_cmpl", "4-click L->C, sigma 0", "gmsig3 s42", "left_and_center", "sketch_cmpl_min4"),
    ("skm4s_cmpl", "4-click L->C, sigma 0.5", "gmsig3 s42", "left_and_center", "sketch_cmpl_min4s"),
    ("s7m4_cmpl", "4-click L->C, sigma 0", "gmsig3 s7", "left_and_center", "sketch_cmpl_min4"),
    ("s7m4s_cmpl", "4-click L->C, sigma 0.5", "gmsig3 s7", "left_and_center", "sketch_cmpl_min4s"),
    ("skm4_cmpr", "4-click R->C, sigma 0", "gmsig3 s42", "right_and_center", "sketch_cmpr_min4"),
    ("skm4s_cmpr", "4-click R->C, sigma 0.5", "gmsig3 s42", "right_and_center", "sketch_cmpr_min4s"),
    ("skm5_cmpr", "5-click R->C", "gmsig3 s42", "right_and_center", "sketch_cmpr_min5"),
    ("skm5ns1_cmpr", "5-click R->C, rollout seed 1", "gmsig3 s42", "right_and_center", "sketch_cmpr_min5"),
    ("s7m5_cmpr", "5-click R->C", "gmsig3 s7", "right_and_center", "sketch_cmpr_min5"),
    ("m5f42_cmpr", "5-click R->C, corrected waypoint", "gmsig3 s42", "right_and_center", "sketch_cmpr_min5f"),
    ("m5fs7_cmpr", "5-click R->C, corrected waypoint", "gmsig3 s7", "right_and_center", "sketch_cmpr_min5f"),
    ("xsk42_cmpl_denis", "Hand-drawn L->C", "xswap s42", "left_and_center", "sketch_cmpl_denis"),
    ("xsk42_cmpr_r1", "Hand-drawn R->C r1", "xswap s42", "right_and_center", "sketch_cmpr_denis_r1"),
    ("xsk42_cmpl_min4", "4-click L->C, sigma 0", "xswap s42", "left_and_center", "sketch_cmpl_min4"),
    ("xsk42_cmpl_min4s", "4-click L->C, sigma 0.5", "xswap s42", "left_and_center", "sketch_cmpl_min4s"),
    ("xsk42_cmpr_min5f", "5-click R->C, corrected waypoint", "xswap s42", "right_and_center", "sketch_cmpr_min5f"),
    ("xsks7_cmpl_denis", "Hand-drawn L->C", "xswap s7", "left_and_center", "sketch_cmpl_denis"),
    ("xsks7_cmpr_r1", "Hand-drawn R->C r1", "xswap s7", "right_and_center", "sketch_cmpr_denis_r1"),
    ("xsks7_cmpl_min4", "4-click L->C, sigma 0", "xswap s7", "left_and_center", "sketch_cmpl_min4"),
    ("xsks7_cmpl_min4s", "4-click L->C, sigma 0.5", "xswap s7", "left_and_center", "sketch_cmpl_min4s"),
    ("xsks7_cmpr_min5f", "5-click R->C, corrected waypoint", "xswap s7", "right_and_center", "sketch_cmpr_min5f"),
    ("xsk42_cmpr_min4", "4-click R->C, sigma 0", "xswap s42", "right_and_center", "sketch_cmpr_min4"),
    ("xsk42_cmpr_min4s", "4-click R->C, sigma 0.5", "xswap s42", "right_and_center", "sketch_cmpr_min4s"),
    ("xsk42_cmpr_min5", "5-click R->C", "xswap s42", "right_and_center", "sketch_cmpr_min5"),
    ("xsk42_cmpr_denis", "Hand-drawn R->C round 1", "xswap s42", "right_and_center", "sketch_cmpr_denis"),
    ("xsks7_cmpr_min4", "4-click R->C, sigma 0", "xswap s7", "right_and_center", "sketch_cmpr_min4"),
    ("xsks7_cmpr_min4s", "4-click R->C, sigma 0.5", "xswap s7", "right_and_center", "sketch_cmpr_min4s"),
    ("xsks7_cmpr_min5", "5-click R->C", "xswap s7", "right_and_center", "sketch_cmpr_min5"),
    ("xsks7_cmpr_denis", "Hand-drawn R->C round 1", "xswap s7", "right_and_center", "sketch_cmpr_denis"),
    ("xsk42_cmpl_min4v2", "4-click L->C v2 (pt 3 moved), sigma 0", "xswap s42", "left_and_center", "sketch_cmpl_min4v2"),
    ("xsk42_cmpl_min4sv2", "4-click L->C v2 (pt 3 moved), sigma 0.5", "xswap s42", "left_and_center", "sketch_cmpl_min4sv2"),
    ("xsk42_cmpr_r2", "Hand-drawn R->C r2 (pts 2,4,5 moved)", "xswap s42", "right_and_center", "sketch_cmpr_denis_r2"),
    ("xsks7_cmpl_min4v2", "4-click L->C v2 (pt 3 moved), sigma 0", "xswap s7", "left_and_center", "sketch_cmpl_min4v2"),
    ("xsks7_cmpl_min4sv2", "4-click L->C v2 (pt 3 moved), sigma 0.5", "xswap s7", "left_and_center", "sketch_cmpl_min4sv2"),
    ("xsks7_cmpr_r2", "Hand-drawn R->C r2 (pts 2,4,5 moved)", "xswap s7", "right_and_center", "sketch_cmpr_denis_r2"),
]
for pre, label, ck, scene, sk in SKETCH_ROWS:
    CELLS.append(dict(id=pre, label=label, arm=ck, campaign="Sketch rows", scene=scene,
                      prefix=pre, sketch=SK(sk)))
APPS = [("app_orbit", "Orbit, 1.5 loops r=0.9", "xswap s42", "right", "sketch_orbit"),
        ("app_fig8", "Figure-eight", "xswap s42", "right", "sketch_fig8"),
        ("app_tempo06", "Tempo 0.6x", "xswap s42", "right", "sketch_tempo06"),
        ("app_tempo10", "Tempo 1.0x", "xswap s42", "right", "sketch_tempo10"),
        ("app_tempo15", "Tempo 1.5x", "xswap s42", "right", "sketch_tempo15"),
        ("scrsk_cmpl", "Hand-drawn L->C through UNPINNED pi0", "scratch3 s42", "left_and_center", "sketch_cmpl_denis"),
        ("scrsk_orbit", "Orbit through UNPINNED pi0", "scratch3 s42", "right", "sketch_orbit"),
        ("sde_cmpl_t03", "Hand-drawn L->C, SDEdit t0=0.3", "scratch3 s42 + SDEdit", "left_and_center", "sketch_cmpl_denis"),
        ("sde_cmpl_t05", "Hand-drawn L->C, SDEdit t0=0.5", "scratch3 s42 + SDEdit", "left_and_center", "sketch_cmpl_denis"),
        ("sde_cmpl_t07", "Hand-drawn L->C, SDEdit t0=0.7", "scratch3 s42 + SDEdit", "left_and_center", "sketch_cmpl_denis"),
        ("sde_cmpl_t09", "Hand-drawn L->C, SDEdit t0=0.9", "scratch3 s42 + SDEdit", "left_and_center", "sketch_cmpl_denis"),
        ("sde_orbit_t03", "Orbit, SDEdit t0=0.3", "scratch3 s42 + SDEdit", "right", "sketch_orbit"),
        ("sde_orbit_t05", "Orbit, SDEdit t0=0.5", "scratch3 s42 + SDEdit", "right", "sketch_orbit"),
        ("sde_orbit_t07", "Orbit, SDEdit t0=0.7", "scratch3 s42 + SDEdit", "right", "sketch_orbit"),
        ("kx_pin_c0", "Hand-drawn L->C + 0.4 m kick, carrot 0", "xswap s42 pin", "left_and_center", "sketch_cmpl_denis"),
        ("kx_pin_c20", "Hand-drawn L->C + 0.4 m kick, carrot 20", "xswap s42 pin", "left_and_center", "sketch_cmpl_denis_c20"),
        ("kx_sde03_c0", "Hand-drawn L->C + 0.4 m kick, carrot 0", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", "sketch_cmpl_denis"),
        ("kx_sde03_c20", "Hand-drawn L->C + 0.4 m kick, carrot 20", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", "sketch_cmpl_denis_c20"),
        ("kx_sde05_c20", "Hand-drawn L->C + 0.4 m kick, carrot 20", "scratch3 s42 + SDEdit t0=0.5", "left_and_center", "sketch_cmpl_denis_c20"),
        ("bs_sde03_m4L", "4-click L->C (original line)", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", "sketch_cmpl_min4"),
        ("bs_sde05_m4L", "4-click L->C (original line)", "scratch3 s42 + SDEdit t0=0.5", "left_and_center", "sketch_cmpl_min4"),
        ("bs_sde03_m4R", "4-click R->C (original line)", "scratch3 s42 + SDEdit t0=0.3", "right_and_center", "sketch_cmpr_min4"),
        ("bs_sde05_m4R", "4-click R->C (original line)", "scratch3 s42 + SDEdit t0=0.5", "right_and_center", "sketch_cmpr_min4"),
        ("fs_pin", "Hand-drawn L->C at 2.5x pace", "xswap s42 pin", "left_and_center", "sketch_cmpl_denis_fast"),
        ("fs_sde03", "Hand-drawn L->C at 2.5x pace", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", "sketch_cmpl_denis_fast"),
        ("adv_none", "Advice L->C: prompt swap only after gate 1", "xswap s42 pin", "left_and_center", None),
        ("adv_coarse", "Advice L->C: one target, coarse x,y words only", "xswap s42 pin", "left_and_center", None),
        ("adv_h50", "Advice L->C: one target, single coarsest x,y word", "xswap s42 pin", "left_and_center", None),
        ("adv_all", "Advice L->C: one target, all 16 words (2-click pursuit)", "xswap s42 pin", "left_and_center", None),
        ("adv_2t", "Advice L->C: staging + exit targets, coarse x,y words", "xswap s42 pin", "left_and_center", None),
        ("adv_sde03", "Advice L->C: one-target pursuit chunk as SDEdit guide", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", None),
        ("ng_2t_coarse", "Nudge L->C (no trigger): 2 targets, coarse x,y words", "xswap s42 pin", "left_and_center", None),
        ("ng_2t_h50", "Nudge L->C (no trigger): 2 targets, single coarsest x,y word", "xswap s42 pin", "left_and_center", None),
        ("ng_2t_all", "Nudge L->C (no trigger): 2 targets, all 16 words", "xswap s42 pin", "left_and_center", None),
        ("ng_2t_coarse_swap", "Nudge L->C (no trigger): 2 targets, coarse words, prompt swap at last target", "xswap s42 pin", "left_and_center", None),
        ("ng_1t_coarse", "Nudge L->C (no trigger): 1 target (center exit) from the start, coarse words", "xswap s42 pin", "left_and_center", None),
        ("ng_1t_idle", "Nudge L->C (no trigger): 1 target, coarse words only while the head is parking", "xswap s42 pin", "left_and_center", None),
        ("ng_sde03_2t", "Nudge L->C (no trigger): 2-target pursuit chunk as SDEdit guide", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", None),
        ("ng_3t_coarse", "Nudge L->C (no trigger): 3 targets incl. center staging, coarse words", "xswap s42 pin", "left_and_center", None),
        ("ng_3t_h50", "Nudge L->C (no trigger): 3 targets, single coarsest word", "xswap s42 pin", "left_and_center", None),
        ("ng_3t_all", "Nudge L->C (no trigger): 3 targets, all 16 words", "xswap s42 pin", "left_and_center", None),
        ("ng_sde03_3t", "Nudge L->C (no trigger): 3-target pursuit chunk as SDEdit guide", "scratch3 s42 + SDEdit t0=0.3", "left_and_center", None),
        ("rs_coarse_xyz_left", "VLM reasoner coarse words: Left gate", "xswap s42 + Qwen2.5-VL-3B", "left", None),
        ("rs_coarse_xyz_right", "VLM reasoner coarse words: Right gate", "xswap s42 + Qwen2.5-VL-3B", "right", None),
        ("rs_coarse_xyz_cfl", "VLM reasoner coarse words: Center from left", "xswap s42 + Qwen2.5-VL-3B", "center", None),
        ("rs_coarse_xyz_cfr", "VLM reasoner coarse words: Center from right", "xswap s42 + Qwen2.5-VL-3B", "center", None),
        ("rs_coarse_xyz_cmpl", "VLM reasoner coarse words: Compound L->C", "xswap s42 + Qwen2.5-VL-3B", "left_and_center", None),
        ("pinoff_left", "Pin OFF at inference (plain noise, sigma 1.5): Left gate", "xswap s42, no command", "left", None),
        ("pinoff_right", "Pin OFF at inference: Right gate", "xswap s42, no command", "right", None),
        ("pinoff_cfl", "Pin OFF at inference: Center from left", "xswap s42, no command", "center", None),
        ("pinoff_cfr", "Pin OFF at inference: Center from right", "xswap s42, no command", "center", None),
        ("sdehead_t03_cfr", "SDEdit t0=0.3 guided by our head's U c: Center from right", "scratch3 + xswap head", "center", None),
        ("sdehead_t05_cfr", "SDEdit t0=0.5 guided by our head's U c: Center from right", "scratch3 + xswap head", "center", None),
        ("sdehead_t03_left", "SDEdit t0=0.3 guided by our head's U c: Left gate", "scratch3 + xswap head", "left", None),
        ("sdehead_t03_right", "SDEdit t0=0.3 guided by our head's U c: Right gate", "scratch3 + xswap head", "right", None),
        ("sdehead_t03_cfl", "SDEdit t0=0.3 guided by our head's U c: Center from left", "scratch3 + xswap head", "center", None),
        ("sdehead_t07_cfr", "SDEdit t0=0.7 guided by our head's U c: Center from right", "scratch3 + xswap head", "center", None),
        ("sdehead_t09_cfr", "SDEdit t0=0.9 guided by our head's U c: Center from right", "scratch3 + xswap head", "center", None),
        ("dec_cfr", "DECODE ONLY (execute U c, no flow): Center from right", "xswap s42 head, no denoising", "center", None),
        ("dec_left", "DECODE ONLY (execute U c, no flow): Left gate", "xswap s42 head, no denoising", "left", None),
        ("dec_cmpl", "DECODE ONLY: hand-drawn L->C sketch's U c, no flow", "sketch command, no denoising", "left_and_center", "sketch_cmpl_denis"),
        ("src_cfr", "SOURCE SAMPLE executed (z = g_perp + U c), no flow: Center from right", "xswap s42 head, no denoising", "center", None),
        ("src_cmpl", "SOURCE SAMPLE executed, no flow: hand-drawn L->C sketch", "sketch command, no denoising", "left_and_center", "sketch_cmpl_denis")]
for pre, label, ck, scene, sk in APPS:
    CELLS.append(dict(id=pre, label=label, arm=ck, campaign="Pin applications & control",
                      scene=scene, prefix=pre, sketch=SK(sk) if sk else None))


def trial_files(prefix):
    fs = glob.glob(f"{RUN}/traj_{prefix}_*.npy")
    fs = [f for f in fs if re.fullmatch(rf"traj_{re.escape(prefix)}_\d+\.npy", os.path.basename(f))]
    return sorted(fs, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))


CAMPAIGN_ORDER = ["Atomic tasks", "Compounds, unguided", "Sketch rows", "Pin applications & control"]
CELLS.sort(key=lambda c: CAMPAIGN_ORDER.index(c["campaign"]))
