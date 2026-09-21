"""Head uncertainty through a flight: atomic left gate vs the two-gate compositional task (2026-09-20).

Reads the per-replan command logs written by serve_gate_pin_joint.py (no sketch, so the GMM head authors
every command) and draws sigma* -- the norm of the served mixture component's own standard deviation --
against flight time, with the gate crossings marked. Cloud sections below show the flights themselves.

  python build_uncert_page.py            -> uncert.html
"""
import json, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer
from build_traj_page import marks
import gate_success as G

RUN = os.path.expanduser("~/ctxrun")
NCH, APC, DT = 14, 50, 0.1
CS, CC = [122, 190, 255], [245, 160, 95]          # left task, compound task
GATE = {"left": np.array([0.91, 0.75]), "center": np.array([2.75, -0.33])}


def load(tag, nch=NCH):
    a = np.load(f"{RUN}/clog_{tag}.npy").astype(np.float64)
    n = a.shape[1]; k = n - 3 - 16 - 4                       # pi block width
    rows = {"pos": a[:, :3], "c": a[:, 3:19], "pi": a[:, 19:19 + k],
            "sstar": a[:, 19 + k], "alpha": a[:, 20 + k], "sigma_serve": a[:, 21 + k]}
    t = len(a) // nch
    return [{q: v[i * nch:(i + 1) * nch] for q, v in rows.items()} for i in range(t)]


def crossings(P, side):
    v = G.judge(P, side)
    return [v["transit_step"]] if v.get("transit_step") is not None else []


def series(tag, side, nch=NCH):
    fl = load(tag, nch)
    out = []
    for i, f in enumerate(fl, 1):
        P = np.load(f"{RUN}/traj_{tag}_{i}.npy")[:, :3]
        out.append({"sstar": f["sstar"], "sigma_serve": f["sigma_serve"],
                    "top": f["pi"].max(axis=1), "arg": f["pi"].argmax(axis=1),
                    "pos": f["pos"], "cross": [c / (APC) for c in crossings(P, side)], "traj": P})
    return out


def chart(A, B, w=980, h=330, pad=54):
    """sigma* vs flight time: faint per-flight lines, bold medians, crossing ticks."""
    xs = np.arange(NCH) * APC * DT
    ymax = max(max(f["sstar"].max() for f in A), max(f["sstar"].max() for f in B)) * 1.12
    X = lambda t: pad + (w - pad - 18) * t / xs[-1]
    Y = lambda v: h - pad + 6 - (h - pad - 24) * v / ymax
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="head uncertainty over flight time">']
    for gy in np.linspace(0, ymax, 5):
        p.append(f'<line x1="{pad}" y1="{Y(gy):.1f}" x2="{w-18}" y2="{Y(gy):.1f}" class="grid"/>')
        p.append(f'<text x="{pad-9}" y="{Y(gy)+4:.1f}" class="tick ar">{gy:.0f}</text>')
    for t in xs[::2]:
        p.append(f'<text x="{X(t):.1f}" y="{h-pad+26}" class="tick ac">{t:.0f}</text>')
    p.append(f'<text x="{X(xs[-1]/2):.1f}" y="{h-10}" class="axl ac">seconds of flight</text>')
    p.append(f'<text x="14" y="{Y(ymax/2):.1f}" class="axl" transform="rotate(-90 14 {Y(ymax/2):.1f})" text-anchor="middle">head uncertainty (sigma*)</text>')
    for fl, col, cls in ((A, CS, "a"), (B, CC, "b")):
        c = f"rgb({col[0]},{col[1]},{col[2]})"
        for f in fl:
            d = " ".join(f'{"M" if i==0 else "L"}{X(xs[i]):.1f},{Y(v):.1f}' for i, v in enumerate(f["sstar"]))
            p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.4" opacity=".30"/>')
            for cr in f["cross"]:
                p.append(f'<circle cx="{X(cr*APC*DT):.1f}" cy="{Y(f["sstar"][min(int(cr), NCH-1)]):.1f}" r="3.4" fill="{c}"/>')
        med = np.median(np.stack([f["sstar"] for f in fl]), axis=0)
        d = " ".join(f'{"M" if i==0 else "L"}{X(xs[i]):.1f},{Y(v):.1f}' for i, v in enumerate(med))
        p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2.6"/>')
    p.append("</svg>")
    return "".join(p)


def main():
    A = series("unc_left", "left")
    B = series("unc_cmpl", "left")
    sec = []
    for title, fl, col, elem, scene, note in [
        ("Left gate (atomic)", A, CS, "v_l", "left",
         "Five head-authored flights of the trained left-gate task; the head predicts every command."),
        ("Left then center (compositional, no demonstrations of the pair)", B, CC, "v_c", "left_and_center",
         "Five head-authored flights of the two-gate task. No compound demonstration exists in either domain.")]:
        g = [{"label": f"flight {i+1}", "color": col, "trajs": [f["traj"]]} for i, f in enumerate(fl)]
        sec.append((title, cloudviewer.viewer_html(scene, g + marks(scene), elem_id=elem, max_pts=40000, note=note)))
    stat = lambda fl, k: np.median([np.median(f[k]) for f in fl])
    rows = [("left gate (atomic)", A, CS), ("left then center (compositional)", B, CC)]
    tab = "".join(
        f'<tr><td><span class="dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{n}</td>'
        f'<td class="n">{stat(f,"sstar"):.1f}</td><td class="n">{np.median([fl["sstar"].max() for fl in f]):.1f}</td>'
        f'<td class="n">{stat(f,"top"):.2f}</td><td class="n">{stat(f,"sigma_serve"):.2f}</td></tr>'
        for n, f, c in rows)
    page = f"""<title>Head Uncertainty Through a Flight</title>
<style>
:root{{--bg:#f7f5f1;--card:#fffefb;--line:#ddd7cb;--ink:#1e2227;--mut:#6c7079;--acc:#0b5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#13161a;--card:#1a1e24;--line:#2d333b;--ink:#e7eaef;--mut:#99a1ac;--acc:#7cc4ee}}}}
:root[data-theme="dark"]{{--bg:#13161a;--card:#1a1e24;--line:#2d333b;--ink:#e7eaef;--mut:#99a1ac;--acc:#7cc4ee}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Georgia,"Iowan Old Style",serif;padding:30px 20px 70px}}
main{{max-width:1040px;margin:0 auto}} h1{{font-size:25px;margin:0 0 6px;text-wrap:balance}} h2{{font-size:17px;margin:30px 0 8px;color:var(--acc)}}
p{{max-width:78ch}} .sub{{color:var(--mut);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px;margin:14px 0}}
.grid{{stroke:var(--line);stroke-width:1}} .tick{{fill:var(--mut);font:11px ui-monospace,Menlo,monospace}}
.ar{{text-anchor:end}} .ac{{text-anchor:middle}} .axl{{fill:var(--mut);font:11px ui-monospace,Menlo,monospace}}
table{{border-collapse:collapse;width:100%;font:13.5px/1.5 ui-monospace,Menlo,monospace;margin-top:4px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
th{{color:var(--mut);font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
.dot{{width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:8px;vertical-align:middle}}
.v3dwrap canvas{{width:100%;border-radius:7px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace;color:var(--mut)}}
.v3dui .lg{{display:flex;gap:6px;align-items:center;cursor:pointer}} .v3dui .sw{{width:20px;height:3px;border-radius:2px;display:inline-block}}
.v3dui .ct{{opacity:.6}} .v3dui .hint{{margin-left:auto;opacity:.8}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:92ch}}
</style>
<main>
<h1>What the head knows, and when it stops knowing it</h1>
<p class="sub">The mixture head authors the command at every replan. Plotted is sigma*, the norm of the
served component's own predicted standard deviation, over the five seconds each command covers. Blue is
the trained left-gate task; orange is the two-gate task, for which no compound demonstration exists in
either domain. Faint lines are single flights, bold lines the median, dots mark the gate crossing.</p>
<div class="card">{chart(A, B)}</div>
<div class="card"><table>
<tr><th>Task</th><th>median sigma*</th><th>peak sigma*</th><th>top mixture weight</th><th>served trust</th></tr>
{tab}</table></div>
{"".join(f'<h2>{t}</h2><div class="card">{v}</div>' for t, v in sec)}
</main>
"""
    out = os.path.join(SP, "uncert.html"); open(out, "w").write(page)
    print("wrote", out, len(page) // 1024, "kB")


if __name__ == "__main__":
    main()
