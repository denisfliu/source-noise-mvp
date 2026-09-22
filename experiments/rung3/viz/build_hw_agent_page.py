"""What the agent saw, said and did on a real flight (2026-09-22): the server-side agent log
(~/gate_flights/agent_log/<trial>/<k>.json + <k>_front.png + <k>_down.png, one entry per replan) laid out
decision by decision, split into sessions where the workstation's k restarts, with a top-down plot per session.

  python build_hw_agent_page.py --trial agentmann_pin_01 --out hw_agent_agentmann.html
"""
import argparse, base64, glob, html, io, json, os
import numpy as np
from PIL import Image
LOG = os.path.expanduser("~/gate_flights/agent_log"); WS = os.path.expanduser("~/gate_flights/agentflights")
SP = os.path.dirname(os.path.abspath(__file__))

def b64img(path, w=256):
    im = Image.open(path).convert("RGB"); im = im.resize((w, int(im.size[1] * w / im.size[0])))
    b = io.BytesIO(); im.save(b, "JPEG", quality=80); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

def svg_track(poses, moves):
    if len(poses) < 1: return ""
    P = np.array([p[:2] for p in poses]); lo, hi = P.min(0) - 0.6, P.max(0) + 0.6; span = max(hi - lo); S = 220
    # +x up, +y left (Denis's plan-view convention)
    def X(p): return S / 2 - (p[1] - (lo[1] + hi[1]) / 2) / span * (S - 20)
    def Y(p): return S / 2 - (p[0] - (lo[0] + hi[0]) / 2) / span * (S - 20)
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(p):.1f},{Y(p):.1f}" for i, p in enumerate(P))
    dots = "".join(f'<circle cx="{X(p):.1f}" cy="{Y(p):.1f}" r="3" fill="{"#f0b040" if moves[i] else "#60d080"}"><title>k={i} {p[:3]}</title></circle>' for i, p in enumerate(P))
    arrows = "".join(f'<line x1="{X(p):.1f}" y1="{Y(p):.1f}" x2="{X(p) - 9*np.sin(p[3]):.1f}" y2="{Y(p) - 9*np.cos(p[3]):.1f}" stroke="#ccc" stroke-width="1"/>' for p in poses)
    return (f'<svg width="{S}" height="{S}" style="background:#1b1e26;border-radius:6px"><path d="{d}" fill="none" stroke="#8ab4ff" stroke-width="1.5"/>{arrows}{dots}'
            f'<text x="6" y="14" fill="#aaa" font-size="11">+x up, +y left · span {span:.1f} m · amber = agent move, green = policy chunk</text></svg>')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trial", default="agentmann_pin_01"); ap.add_argument("--out", default="hw_agent_agentmann.html"); a = ap.parse_args()
    fs = sorted(glob.glob(f"{LOG}/{a.trial}/*.json")); ents = [json.load(open(f)) for f in fs]
    ws = {}
    if os.path.exists(f"{WS}/{a.trial}.jsonl"):
        for i, l in enumerate(open(f"{WS}/{a.trial}.jsonl")): ws[i] = json.loads(l)
    # sessions: the workstation's k restarts at 0 when the node is relaunched
    sessions, cur, lastk = [], [], -1
    for i, e in enumerate(ents):
        k = int(ws[i]["k"]) if i in ws else int(e["k"])
        if k <= lastk and cur: sessions.append(cur); cur = []
        cur.append((i, k, e)); lastk = k
    if cur: sessions.append(cur)
    css = ("<style>body{font-family:system-ui;background:#111318;color:#ddd;margin:0;padding:18px}h2{margin:6px 0 2px}h3{margin:22px 0 6px;color:#9ec}"
           ".dec{display:grid;grid-template-columns:256px 256px 1fr;gap:12px;align-items:start;border-top:1px solid #2a2e38;padding:10px 0}"
           ".dec img{border-radius:4px;display:block}.say{font-size:15px;line-height:1.35}.kv{color:#9aa;font-size:12.5px;margin-top:6px;line-height:1.5}"
           ".mv{color:#f0b040}.auto{color:#60d080}.sess{display:flex;gap:18px;align-items:flex-start}.sub{color:#9aa;font-size:13px}</style>")
    out = [f"<title>Real Flight Agent Log</title>{css}<h2>What the agent saw, said and did: real flight {html.escape(a.trial)}</h2>",
           "<p class='sub'>Server-side agent log on manaan: the frames the policy received at each replan, the instruction the agent wrote, the move it asked for, "
           "the chunk that was executed (net displacement over the executed steps) and the mocap pose. Amber = agent move, green = the policy's own chunk (--auto / approve). "
           "The 2026-09-22 workstation diagnosis applies: PX4 tracked about 30% of each chunk in these sessions and the node re-anchored the hold, so 'net' is what the setpoints commanded, not what the aircraft flew.</p>"]
    for si, sess in enumerate(sessions):
        poses = [e["pose"][:4] for _, _, e in sess]; mv = [bool(e.get("move_executed") and e["move_executed"].get("move")) for _, _, e in sess]
        out.append(f"<h3>Session {si + 1}: {len(sess)} decisions, start ({poses[0][0]:.2f}, {poses[0][1]:.2f}) heading {np.degrees(poses[0][3]):.0f} deg</h3><div class='sess'>{svg_track(poses, mv)}<div>")
        for i, k, e in sess:
            front, down = f"{LOG}/{a.trial}/{i:04d}_front.png", f"{LOG}/{a.trial}/{i:04d}_down.png"
            m = (e.get("move_executed") or {}).get("move"); net = e.get("chunk_net_xyz_yaw", [])
            say = (e.get("agent") or {}).get("say") or e.get("prompt", "")
            w = ws.get(i, {})
            mvtxt = (f"<span class='mv'>agent move: forward {m['forward']:+.2f} left {m['left']:+.2f} up {m['up']:+.2f} yaw {m['yaw_deg']:+.0f} deg sigma {m['sigma']}</span>" if m
                     else "<span class='auto'>policy chunk (approved / --auto)</span>")
            out.append(f"<div class='dec'><img src='{b64img(front)}' title='forward'><img src='{b64img(down)}' title='downward'>"
                       f"<div><div class='say'><b>k={k}</b> &nbsp; {html.escape(str(say))}</div><div class='kv'>{mvtxt}<br>"
                       f"pose ({e['pose'][0]:.2f}, {e['pose'][1]:.2f}, {e['pose'][2]:.2f}) heading {np.degrees(e['pose'][3]):.0f} deg &nbsp;·&nbsp; "
                       f"executed chunk net dx {net[0]:+.2f} dy {net[1]:+.2f} dz {net[2]:+.2f} yaw {np.degrees(net[3]) if len(net) > 3 else 0:+.0f} deg &nbsp;·&nbsp; sigma {e.get('sigma_serve')}"
                       + (f" &nbsp;·&nbsp; server latency {w.get('latency_s', '')} s" if w else "") + "</div></div></div>")
        out.append("</div></div>")
    open(os.path.join(SP, a.out), "w").write("\n".join(out)); print("wrote", a.out, len(sessions), "sessions", len(ents), "decisions")
main()
