"""What each mixture mode was proposing (2026-09-20).

With SNMVP_CLOG_FULL=1 the command log keeps every mode's mean command, so each mode can be decoded into
the coarse 5-second path it was proposing and drawn from the pose where the head chose between them.
Row: [pos(3), c(K), pi(M), sigma*, alpha, sigma_serve, phase, mu(M*K), |sigma_j|(M)].

  python build_modes_page.py            -> modes.html
"""
import json, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, os.path.dirname(SP))
import cloudviewer
from build_traj_page import marks

RUN = os.path.expanduser("~/ctxrun")
RD = os.path.dirname(SP)
K, M, NCH, H, AD = 16, 4, 14, 50, 32
U = np.load(f"{RD}/pin_U_mh16.npy").astype(np.float32)
NS = json.load(open("/home/dfliu/code/openpi-snmvp/assets/pi0_gate3/local/gate_nav3/norm_stats.json"))["norm_stats"]["actions"]
AMEAN, ASTD = np.asarray(NS["mean"][:4], np.float32), np.asarray(NS["std"][:4], np.float32)
MODE_COL = [[122, 190, 255], [245, 160, 95], [130, 215, 160], [215, 140, 235]]


def rows(tag):
    a = np.load(f"{RUN}/clog_{tag}.npy").astype(np.float32)
    b = 3 + K + M + 4                                     # base width
    out = []
    for i in range(len(a)):
        r = a[i]
        out.append({"pos": r[:3], "c": r[3:3 + K], "pi": r[3 + K:3 + K + M],
                    "sstar": r[3 + K + M], "sigma_serve": r[3 + K + M + 2],
                    "mu": r[b:b + M * K].reshape(M, K) if len(r) >= b + M * K else None,
                    "sig": r[b + M * K:b + M * K + M] if len(r) >= b + M * K + M else None})
    return [out[i * NCH:(i + 1) * NCH] for i in range(len(out) // NCH)]


def decode(c, pos):
    """Command -> the coarse 5 s path it asks for, in world coordinates from pos."""
    ch = (U @ np.asarray(c, np.float32)).reshape(H, AD)[:, :4]
    d = ch * (ASTD + 1e-6) + AMEAN
    return np.concatenate([pos[None, :3], pos[None, :3] + np.cumsum(d[:, :3], axis=0)]).astype(np.float32)


def section(tag, k, scene, elem, note, flight=0):
    f = rows(tag)[flight][k]
    g = []
    for j in np.argsort(-f["pi"]):
        lab = f"mode {j}: weight {f['pi'][j]:.2f}, uncertainty {f['sig'][j]:.1f}"
        if abs(f["pi"][j] - f["pi"].max()) < 1e-6:
            lab += "  <- served"
        g.append({"label": lab, "color": MODE_COL[j], "trajs": [decode(f["mu"][j], f["pos"])]})
    g.append({"label": "where the drone is", "fixed": True, "color": [250, 250, 250],
              "trajs": [np.stack([f["pos"], f["pos"] + np.array([0, 0, 0.35], np.float32)])]})
    return cloudviewer.viewer_html(scene, g + marks(scene), elem_id=elem, max_pts=40000, note=note)


def main():
    secs = []
    for tag, scene, title, note in [
        ("unc_cmpl", "left_and_center", "Compound sentence (never seen in training), the replan after the left gate",
         "The head has just taken the drone through the left gate and must decide what comes next. Each line is one mixture mode's mean command decoded into the five seconds of motion it proposes, drawn from the drone's position."),
        ("unc_cfl", "center", "Center-from-left (a trained sentence), the same replan",
         "The same moment under a sentence the model was trained on."),
        ("unc_left", "left", "Left gate (a trained sentence), the same replan",
         "The atomic task, for reference.")]:
        secs.append((title, section(tag, 3, scene, "v_" + tag, note)))
    tab = []
    for tag, name in [("unc_left", "left gate (trained)"), ("unc_cfl", "center from left (trained)"),
                      ("unc_cmpl", "compound sentence (unseen)")]:
        fl = rows(tag)
        w = np.stack([[r["pi"] for r in f] for f in fl])           # (trials, replans, M)
        s = np.stack([[r["sstar"] for r in f] for f in fl])
        sp = np.sort(w, axis=2)
        tab.append(f"<tr><td>{name}</td><td class='n'>{np.median(s):.1f}</td>"
                   f"<td class='n'>{np.median(w.max(axis=2)):.2f}</td>"
                   f"<td class='n'>{np.median(sp[:, :, -2]):.2f}</td>"
                   f"<td class='n'>{np.median((w > 0.05).sum(axis=2)):.0f}</td></tr>")
    page = f"""<title>What Each Mode Proposes</title>
<style>
:root{{--bg:#f7f5f1;--card:#fffefb;--line:#ddd7cb;--ink:#1e2227;--mut:#6c7079;--acc:#0b5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#13161a;--card:#1a1e24;--line:#2d333b;--ink:#e7eaef;--mut:#99a1ac;--acc:#7cc4ee}}}}
:root[data-theme="dark"]{{--bg:#13161a;--card:#1a1e24;--line:#2d333b;--ink:#e7eaef;--mut:#99a1ac;--acc:#7cc4ee}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Georgia,"Iowan Old Style",serif;padding:30px 20px 70px}}
main{{max-width:1040px;margin:0 auto}} h1{{font-size:25px;margin:0 0 6px;text-wrap:balance}} h2{{font-size:17px;margin:30px 0 8px;color:var(--acc)}}
p{{max-width:78ch}} .sub{{color:var(--mut);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:14px 16px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font:13.5px/1.5 ui-monospace,Menlo,monospace}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
th{{color:var(--mut);font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
.v3dwrap canvas{{width:100%;border-radius:7px;display:block}}
.v3dui{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:8px;font:12px ui-monospace,Menlo,monospace;color:var(--mut)}}
.v3dui .lg{{display:flex;gap:6px;align-items:center;cursor:pointer}} .v3dui .sw{{width:20px;height:3px;border-radius:2px;display:inline-block}}
.v3dui .ct{{opacity:.6}} .v3dui .hint{{margin-left:auto;opacity:.8}} .v3dnote{{color:var(--mut);font-size:13px;margin:8px 2px 0;max-width:92ch}}
</style>
<main>
<h1>What each mode proposes after the gate</h1>
<p class="sub">The command head is a mixture: at every replan it predicts several candidate commands with
weights. Below, at the replan just after the left-gate crossing, each mode's mean command is decoded into
the five seconds of motion it asks for and drawn from the drone's position. The served command is the
highest-weighted mode.</p>
<div class="card"><table>
<tr><th>Prompt</th><th>median sigma*</th><th>top weight</th><th>second weight</th><th>modes above 5%</th></tr>
{''.join(tab)}</table></div>
{''.join(f'<h2>{t}</h2><div class="card">{v}</div>' for t, v in secs)}
</main>
"""
    out = os.path.join(SP, "modes.html"); open(out, "w").write(page)
    print("wrote", out, len(page) // 1024, "kB")


if __name__ == "__main__":
    main()
