"""Build the APC=50 grounded-prior review page: per-group videos, 3D trajectory viewers over the
scene point clouds, and the offline command-accuracy diagnostic that contradicts the flights."""
import base64
import glob
import os
import re

import numpy as np

import cloudviewer

SP = os.path.dirname(os.path.abspath(__file__))
RUN = "/home/ubuntu/ctxrun"
OUT = f"{SP}/apc50_review.html"

SCORES = open(f"{RUN}/apc50full_scores.txt").read()


def parse(tag):
    """(transits, clearance-clean or None, judge rows, per-trial min clearances) for one group.

    Blocks are delimited by the '== <tag> (judge: ...)' headers the score file writes; the judge
    and clearance sections both emit lines beginning 'traj_', so they are told apart by field."""
    lines = SCORES.splitlines()
    heads = [i for i, l in enumerate(lines) if l.startswith("== ") and "(judge" in l]
    at = next(i for i in heads if lines[i].startswith(f"== {tag} (judge"))
    end = next((i for i in heads if i > at), len(lines))
    seg = "\n".join(lines[at:end])
    tr = re.search(r"== (\d+)/(\d+) success", seg)
    cl = re.search(r"== (\d+)/(\d+) clearance-clean", seg)
    rows = [l for l in seg.splitlines() if l.startswith("traj_") and "SUCCESS=" in l]
    mins = [float(x) for x in re.findall(r"min-clearance ([0-9.]+) m", seg)]
    return (int(tr.group(1)), int(tr.group(2))), (int(cl.group(1)) if cl else None), rows, mins


GROUPS = [
    dict(tag="sing_left", title="Left gate", scene="left", judge="left",
         prompt="go through the gate on the left and hover over the stuffed animal", nch=8),
    dict(tag="sing_right", title="Right gate", scene="right", judge="right",
         prompt="go through the gate on the right and hover over the stuffed animal", nch=8),
    dict(tag="ctr_cfl", title="Centre gate, from the left", scene="center", judge="center_from_left",
         prompt="go through the center gate from the left and hover over the stuffed animal", nch=10),
    dict(tag="ctr_cfr", title="Centre gate, from the right", scene="center", judge="center_from_right",
         prompt="go through the center gate from the right and hover over the stuffed animal", nch=10),
    dict(tag="cmp_left", title="Compositional: left, then centre", scene="left_and_center",
         judge="left_and_center",
         prompt="go through the gate on the left, then through the center gate and hover over the stuffed animal",
         nch=14),
    dict(tag="cmp_right", title="Compositional: right, then centre", scene="right_and_center",
         judge="right_and_center",
         prompt="go through the gate on the right, then through the center gate and hover over the stuffed animal",
         nch=14),
]

for g in GROUPS:
    (a, b), cl, rows, mins = parse(g["tag"])
    g.update(transit=a, ntr=b, clean=cl, rows=rows, mins=mins)
    g["trajs"] = [np.load(f)[:, :3] for f in sorted(glob.glob(f"{RUN}/traj_a50_{g['tag']}_*.npy"))]
    g["ok"] = [("SUCCESS=True" in r) for r in rows] if rows else [False] * len(g["trajs"])
    g["vids"] = [f"{SP}/v_{g['tag']}_{i}.mp4" for i in (1, 2)]


def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def chip(g):
    if g["transit"] == 0:
        return '<span class="chip fail">no transit</span>'
    if g["clean"] == 0:
        return f'<span class="chip graze">{g["transit"]}/{g["ntr"]} transit · clips gate</span>'
    if g["clean"] is None:
        return f'<span class="chip graze">{g["transit"]}/{g["ntr"]} judge</span>'
    return f'<span class="chip pass">{g["transit"]}/{g["ntr"]} clean</span>'


OFFLINE = [("left", 0.9589, "+0.22 −0.12 −0.08", "+0.20 −0.09 −0.09", "0.08 0.04 0.05", "0.04 0.03 0.04"),
           ("right", 0.9469, "+0.24 −0.05 −0.08", "+0.23 −0.05 −0.10", "0.07 0.08 0.06", "0.07 0.07 0.04"),
           ("centre from left", 0.9387, "+0.28 −0.20 −0.09", "+0.26 −0.17 −0.09", "0.09 0.07 0.05", "0.05 0.07 0.02"),
           ("centre from right", 0.9027, "+0.31 −0.07 −0.11", "+0.25 −0.04 −0.12", "0.09 0.07 0.04", "0.17 0.05 0.05")]

REACH = [("Left gate", "grounded", "2.41 2.35 2.38", "0 / 5"), ("Left gate", "one-hot", "1.84 1.77 1.77", "5 / 5"),
         ("Right gate", "grounded", "2.76 2.85 2.76", "0 / 5"), ("Right gate", "one-hot", "1.52 1.52 1.52", "5 / 5")]

css = """
:root{
  --paper:#f6f7f9; --card:#ffffff; --ink:#111820; --ink2:#48545f; --line:#dde2e8;
  --teal:#0e7c7b; --pass:#2c7f4f; --graze:#a8701a; --fail:#a83a3a; --shade:#eef1f5;
}
@media (prefers-color-scheme: dark){
  :root{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1; --ink2:#94a2b0; --line:#242e39;
         --teal:#4fd1c9; --pass:#5fbe86; --graze:#d7a352; --fail:#e0787a; --shade:#1a222b; }
}
:root[data-theme="dark"]{ --paper:#0d1116; --card:#141a21; --ink:#e6ebf1; --ink2:#94a2b0; --line:#242e39;
  --teal:#4fd1c9; --pass:#5fbe86; --graze:#d7a352; --fail:#e0787a; --shade:#1a222b; }
:root[data-theme="light"]{ --paper:#f6f7f9; --card:#ffffff; --ink:#111820; --ink2:#48545f; --line:#dde2e8;
  --teal:#0e7c7b; --pass:#2c7f4f; --graze:#a8701a; --fail:#a83a3a; --shade:#eef1f5; }
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
main{max-width:1080px;margin:0 auto;padding:48px 22px 96px;display:flex;flex-direction:column;gap:44px}
header{display:flex;flex-direction:column;gap:12px;border-bottom:1px solid var(--line);padding-bottom:26px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--teal)}
h1{margin:0;font-size:clamp(27px,4vw,40px);line-height:1.14;letter-spacing:-.02em;text-wrap:balance;font-weight:650}
.sub{margin:0;color:var(--ink2);max-width:66ch}
h2{margin:0 0 4px;font-size:22px;letter-spacing:-.01em;font-weight:640}
h3{margin:0;font-size:17px;font-weight:620}
p{margin:0 0 12px;max-width:72ch}
section{display:flex;flex-direction:column;gap:18px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.sc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 15px;
  display:flex;flex-direction:column;gap:9px}
.sc .t{font-size:14px;font-weight:600;line-height:1.3}
.chip{display:inline-block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
  padding:3px 8px;border-radius:5px;border:1px solid currentColor;white-space:nowrap}
.chip.pass{color:var(--pass)} .chip.graze{color:var(--graze)} .chip.fail{color:var(--fail)}
.sc .n{font-size:12.5px;color:var(--ink2)}
.grp{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.ghead{padding:16px 18px;display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;
  justify-content:space-between;border-bottom:1px solid var(--line)}
.prompt{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:var(--ink2);
  background:var(--shade);padding:8px 11px;border-radius:6px;border-left:2px solid var(--teal);
  margin:14px 18px 0;overflow-x:auto}
.vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;padding:14px 18px}
.vids figure{margin:0;display:flex;flex-direction:column;gap:7px}
video{width:100%;border-radius:8px;background:#000;display:block}
figcaption{font-size:12.5px;color:var(--ink2)}
.tbl{overflow-x:auto;padding:0 18px 18px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);font-weight:600}
tbody tr:last-child td{border-bottom:none}
.good{color:var(--pass)} .bad{color:var(--fail)} .warn{color:var(--graze)}
.v3dwrap{padding:0 18px 18px;display:flex;flex-direction:column;gap:9px}
.v3dwrap canvas{width:100%;display:block;border-radius:8px;background:#05080b;border:1px solid var(--line)}
.v3dui{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:12.5px;color:var(--ink2)}
.lg{display:inline-flex;gap:6px;align-items:center;cursor:pointer}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.ct{color:var(--ink2);opacity:.75} .hint{opacity:.7}
.v3dnote{font-size:12.5px;color:var(--ink2);margin:0}
.callout{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--teal);
  border-radius:10px;padding:16px 18px;display:flex;flex-direction:column;gap:8px}
.callout h3{color:var(--teal)}
figure.dia{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 18px 14px}
figure.dia svg{width:100%;height:auto;max-width:660px;margin:0 auto;display:block}
footer{color:var(--ink2);font-size:12.5px;border-top:1px solid var(--line);padding-top:18px}
"""

DIA = """
<figure class="dia">
<svg viewBox="0 0 660 226" role="img" aria-label="A 50-step action chunk with only the first 8 steps
executed before replanning, versus all 50 executed; the pinned command spans the whole chunk.">
 <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
   orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>
 <text x="0" y="14" font-size="12" fill="currentColor" opacity=".72">one inference = a 50-step chunk; the pin commands its coarse displacement</text>
 <rect x="0" y="26" width="600" height="26" rx="4" fill="none" stroke="currentColor" opacity=".45"/>
 <line x1="0" y1="62" x2="600" y2="62" stroke="currentColor" opacity=".35" marker-end="url(#ar)"/>
 <text x="604" y="66" font-size="11" fill="currentColor" opacity=".6">step 50</text>
 <text x="0" y="98" font-size="12" fill="currentColor">APC = 8</text>
 <rect x="0" y="106" width="96" height="26" rx="4" fill="currentColor" opacity=".8"/>
 <rect x="96" y="106" width="504" height="26" rx="4" fill="none" stroke="currentColor"
   stroke-dasharray="4 4" opacity=".4"/>
 <text x="106" y="123" font-size="11.5" fill="currentColor" opacity=".62">discarded — replan from the new state</text>
 <text x="0" y="166" font-size="12" fill="currentColor">APC = 50</text>
 <rect x="0" y="174" width="600" height="26" rx="4" fill="currentColor" opacity=".8"/>
 <text x="0" y="218" font-size="11.5" fill="currentColor" opacity=".72">
   0.16 of the commanded displacement flown  ·  vs  ·  all of it flown, with no feedback inside the chunk</text>
</svg>
<figcaption>The execution fraction. Raising it from 8 to 50 steps took the one-hot scaffold from 7/20 to
10/10 completions with no retraining — and is what exposes a command error that frequent replanning
would have papered over.</figcaption>
</figure>"""


def group_html(g, i):
    trs = [t for t, ok in zip(g["trajs"], g["ok"]) if ok]
    frs = [t for t, ok in zip(g["trajs"], g["ok"]) if not ok]
    grp = []
    if trs:
        grp.append({"label": "judged transit", "color": [95, 190, 134], "trajs": trs})
    if frs:
        grp.append({"label": "failed", "color": [224, 120, 122], "trajs": frs})
    note = ("Every run starts at the origin at 1.5 m. Orbit to see how far past the gate plane the "
            "flights travel before turning back toward the goal.")
    view = cloudviewer.viewer_html(g["scene"], grp, note=note, height=430, elem_id=f"v{i}", max_pts=14000)
    rows = ""
    for j, ok in enumerate(g["ok"]):
        if j < len(g["mins"]):
            cval, ccls = f"{g['mins'][j]:.3f} m", "good" if g["mins"][j] >= 0.18 else "warn"
        else:
            cval, ccls = "not measured", ""
        rows += (f'<tr><td class="mono">{j + 1}</td>'
                 f'<td class="mono {"good" if ok else "bad"}">{"transit" if ok else "no transit"}</td>'
                 f'<td class="mono {ccls}">{cval}</td>'
                 f'<td class="mono">{np.max(g["trajs"][j][:, 0]):.2f}</td>'
                 f'<td class="mono">{np.round(g["trajs"][j][-1], 2).tolist()}</td></tr>')
    vids = "".join(
        f'<figure><video controls preload="metadata" playsinline '
        f'src="data:video/mp4;base64,{b64(v)}"></video>'
        f'<figcaption>trial {k+1} — {"transit" if g["ok"][k] else "no transit"}</figcaption></figure>'
        for k, v in enumerate(g["vids"]) if os.path.exists(v))
    return f"""
<div class="grp">
 <div class="ghead"><h3>{g['title']}</h3>{chip(g)}</div>
 <div class="prompt">{g['prompt']}</div>
 <div class="vids">{vids}</div>
 <div class="tbl"><table><thead><tr><th>trial</th><th>transit judge</th><th>min gate clearance</th>
  <th>max x reached</th><th>end position</th></tr></thead><tbody>{rows}</tbody></table></div>
 {view}
</div>"""


strip = "".join(
    f'<div class="sc"><div class="t">{g["title"]}</div>{chip(g)}'
    f'<div class="n">{g["nch"]} chunks · {g["nch"]*50} steps allowed</div></div>' for g in GROUPS)

off = "".join(
    f'<tr><td>{n}</td><td class="mono">{r:+.4f}</td><td class="mono">{t}</td><td class="mono">{p}</td>'
    f'<td class="mono">{e}</td><td class="mono">{x}</td></tr>' for n, r, t, p, e, x in OFFLINE)

reach = "".join(
    f'<tr><td>{a}</td><td>{b}</td><td class="mono">{c}</td>'
    f'<td class="mono {"good" if d.startswith("5") else "bad"}">{d}</td></tr>' for a, b, c, d in REACH)

html = f"""<title>APC=50 with the grounded command source</title>
<style>{css}</style>
<main>
<header>
 <div class="eyebrow">drone gate navigation · flight review · 2026-08-11</div>
 <h1>Full-chunk execution with an enumeration-free command source</h1>
 <p class="sub">The same pinned flow that flies 10/10 clean under the one-hot scaffold, re-served with a
 language-grounded prior that has never seen a task list. Five trials per group, strict transit judge plus
 gate-clearance, every trajectory shown.</p>
</header>

<section>
 <h2>Where it stands</h2>
 <div class="strip">{strip}</div>
 <div class="callout">
  <h3>No group is a clean success</h3>
  <p>The single gates never cross the aperture — the flights reach the gate, pass it off-centre by
  7–15&nbsp;cm on the left and miss by 20–25&nbsp;cm on the right, then continue out to
  <span class="mono">x&nbsp;≈&nbsp;2.4–2.85&nbsp;m</span> before returning to hover near the goal box.
  The centre-gate groups do transit, 5/5 and 4/5, but every one of them clips the gate
  (min clearance 0.003–0.062&nbsp;m against a 0.18&nbsp;m body threshold), so under our own rule —
  transit judge <em>and</em> clearance <em>and</em> video — they are grazes, not successes. The
  compositional prompts latch zero of two gates.</p>
 </div>
</section>

<section>
 <h2>What the execution fraction changed</h2>
 {DIA}
 <p>Raising the executed steps per inference is a serving-side change, so this ran on an existing
 checkpoint. Under the one-hot scaffold it was decisive. Under the grounded prior it is not: the flights
 travel roughly a metre further downrange, which is exactly what open-loop execution of a slightly
 over-long command looks like.</p>
 <div class="tbl" style="padding:0">
  <table><thead><tr><th>scene</th><th>command source</th><th>max x reached (m, 3 trials)</th>
  <th>clean successes</th></tr></thead><tbody>{reach}</tbody></table></div>
</section>

<section>
 <h2>The offline metrics do not predict this</h2>
 <p>On held-out episodes the grounded prior's commands are accurate per task, and accurate in physical
 units: 4–17&nbsp;cm of error per axis on a chunk whose commanded displacement is 20–30&nbsp;cm. Both
 numbers are as good as the one-hot scaffold's. Whatever breaks the flights is not visible here.</p>
 <div class="tbl" style="padding:0">
  <table><thead><tr><th>task</th><th>held c-R²</th><th>true [dx dy dz] m</th><th>predicted [dx dy dz] m</th>
  <th>MAE early (m)</th><th>MAE transit (m)</th></tr></thead><tbody>{off}</tbody></table></div>
 <p style="margin-top:6px;color:var(--ink2);font-size:13.5px">Per task, not pooled. Pooled R² is
 inflated by between-task variance and would read +0.94 for a predictor that only knew which task it
 was on.</p>
</section>

<section>
 <h2>Every flight</h2>
 <p>Two of five trials per group as video, all five as trajectories in the point cloud. The clouds are
 the same Gaussian-splat geometry the clearance scorer measures against, with the scene's gate edits
 applied.</p>
 {"".join(group_html(g, i) for i, g in enumerate(GROUPS))}
</section>

<footer>
 Flow <span class="mono">gate_pin_zeropad/4999</span> · basis <span class="mono">pin_U_gate_rrr_k5</span> ·
 command source <span class="mono">langprior_zeropad.pt</span>, an MLP over the model state and a PCA-64
 projection of the post-fusion language-token embedding — no task list, no classifier, live embeddings
 every inference. Judge <span class="mono">rung3/gate_success.py</span>, clearance
 <span class="mono">rung3/gate_clearance.py</span> at a 0.18 m body radius. Scene clouds and gate
 geometry are simulator ground truth, used here for scoring and visualisation only.
</footer>
</main>
<script>
document.querySelectorAll('video').forEach(v=>{{v.addEventListener('play',()=>{{
  document.querySelectorAll('video').forEach(o=>{{if(o!==v)o.pause()}});}});}});
</script>
"""
open(OUT, "w").write(html)
print(OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB")
