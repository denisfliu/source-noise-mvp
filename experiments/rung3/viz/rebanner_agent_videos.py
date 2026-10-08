"""Rebuild the agent-flight videos with a banner that shows all of its text (2026-10-08).

The banner drawn during the flights (gate_rollout_batch._banner) never wrapped its first line (task · arm · tag), so long
tasks ran off the frame, and it silently dropped the agent's note after six lines. The image area is fine and every
decision's full note is archived (agentflight/<tag>_decisions/cmds/<k>.json), so each video is rebuilt from its own
frames: the scene part is kept, the banner is redrawn below it with every line wrapped and nothing dropped, at one
banner height per video (tall enough for its longest note).

Frames map to decisions by the old banner itself: its decision line changes exactly when a new decision starts, so a
jump in that band marks a boundary. The script checks that the number of segments equals the number of decisions that
flew (a stop decision ends the flight without frames) and refuses a video where they disagree.

  python3 rebanner_agent_videos.py --out ~/ctxrun/agent_videos_fixed      # the 90 Table 2 evaluation flights
"""
import argparse, json, os, subprocess, sys, textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = "/home/dfliu/ctxrun"; RD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = {"mannequin": "find the mannequin and hover in front of it", "orbit": "fly one full circle around the center gate",
         "left_mannequin": "go through the gate on the left, then find the mannequin and hover in front of it"}
CELLS = [("Sonnet", "ours", "_ours", range(11, 16)), ("Sonnet", "decoded U c (old waypoint baseline)", "_waypoints", range(11, 16)),
         ("Sonnet", "smooth waypoints", "_smoothwp_claude-sonnet-5", range(11, 16)),
         ("Opus", "ours", "_ours_opus", range(21, 26)), ("Opus", "decoded U c (old waypoint baseline)", "_waypoints_claude-opus-5", range(21, 26)),
         ("Opus", "smooth waypoints", "_smoothwp_claude-opus-5", range(21, 26))]
W, H_IN = 512, 560                  # the flight videos: 512 x 554 rendered, scaled to 560 by the encoder
SCENE = round(384 * H_IN / 554)     # rows of scene (image + overlays) above the old banner
BANNER = slice(SCENE + 4, H_IN)      # the old banner: static text within a decision
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H_IN, W, 3)


def segments(F):
    """Start index of each decision's frames: the old banner's text is static within a decision, so a boundary is a frame
    pair where banner pixels change sharply (consecutive holds repeat their note, so only the decision number may change)."""
    d = (np.abs(F[1:, BANNER].astype(np.int16) - F[:-1, BANNER].astype(np.int16)).max(axis=3) > 60).sum(axis=(1, 2))
    return [0] + [i + 1 for i in np.where(d > 5)[0]]    # a one-digit change ("decision 16" -> "17") moves ~10-40 pixels; noise moves 0


def wrap(draw, text, font, width):
    lines, line = [], ""
    for w in text.split():
        cand = (line + " " + w).strip()
        if draw.textlength(cand, font=font) > width and line:
            lines.append(line); line = w
        else:
            line = cand
    return lines + ([line] if line else [])


def banner_lines(draw, head, k, verdict, why, f, fb):
    out = [(l, fb, (200, 205, 215)) for l in wrap(draw, head, fb, W - 20)]
    out.append((f"decision {k}  ·  {verdict.upper()}", fb, (120, 220, 150) if verdict == "approve" else (250, 180, 90)))
    out += [(l, f, (225, 230, 238)) for l in wrap(draw, why, f, W - 20)]
    return out


def rebuild(src, dst, head, decisions):
    F = frames(src); S = segments(F)
    flown = [c for c in decisions if not c.get("stop")]
    if len(S) != len(flown):
        return f"SKIP {os.path.basename(src)}: {len(S)} banner segments vs {len(flown)} decisions that flew"
    f, fb = ImageFont.truetype(FONT, 14), ImageFont.truetype(BOLD, 14)
    probe = ImageDraw.Draw(Image.new("RGB", (W, 10)))
    per = [banner_lines(probe, head, c["k"], c.get("verdict", "approve"), c.get("why", "") or "", f, fb) for c in flown]
    bh = 12 + max(len(p) for p in per) * 18; bh += (SCENE + bh) % 2      # even frame height for yuv420
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{SCENE + bh}", "-r", "9",
                            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst], stdin=subprocess.PIPE)
    bounds = S + [len(F)]
    for j, lines in enumerate(per):
        im = Image.new("RGB", (W, SCENE + bh), (14, 16, 20)); d = ImageDraw.Draw(im); y = SCENE + 7
        for text, font, col in lines:
            d.text((10, y), text, fill=col, font=font); y += 18
        canvas = np.asarray(im).copy()
        for i in range(bounds[j], bounds[j + 1]):
            canvas[:SCENE] = F[i, :SCENE]
            enc.stdin.write(canvas.tobytes())
    enc.stdin.close(); enc.wait()
    return f"ok {os.path.basename(dst)}: {len(F)} frames, {len(flown)} decisions, banner {bh}px"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    bad = 0
    for agent, arm, suf, trials in CELLS:
        sub = os.path.join(a.out, f"{agent} - {arm}"); os.makedirs(sub, exist_ok=True)
        for task, prompt in TASKS.items():
            for t in trials:
                tag = f"{task}{suf}_t{t}"; cd = f"{RD}/agentflight/{tag}_decisions/cmds"
                dec = [json.load(open(os.path.join(cd, n))) for n in sorted(os.listdir(cd))]
                head = f"TASK: {prompt}   ·   {agent} + {arm}   ·   trial {t}"
                msg = rebuild(f"{RUN}/agent_{tag}.mp4", os.path.join(sub, f"{task}_t{t}.mp4"), head, dec)
                bad += msg.startswith("SKIP"); print(msg, flush=True)
    print(f"done, {bad} skipped"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
