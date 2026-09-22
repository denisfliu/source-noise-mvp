"""Results dashboard: one tab per line of work, built from the ledgers so it never drifts from them
(2026-09-21; Denis: "an artifact that I can continually reference for all of our results ... click on a tab
to see results for each thing, e.g. hardware, sim").

  python build_results_dashboard.py --out results_dashboard.html

Tabs and their sources:
  Overview        inline (below)                      the headline per line, one paragraph each
  Sim             docs/status_latest.md               record board and standing flaws
  Hardware        docs/HARDWARE_RESULTS.md            every real-drone experiment
  Injection       docs/APPENDIX_INJECTION.md          pin vs SDEdit vs inject vs v-proj, bad/worse sketches, compliance
  Agent flights   docs/AGENT_FLIGHTS.md               reviewed flights: compound task, mannequin, penguin
  Pages           inline                              every published point-cloud page, newest first
Update a ledger, rerun, republish to the same URL. The markdown subset understood: #/##/### headings,
paragraphs, "- " bullets (one level), pipe tables, **bold**, `code`, [text](url), and *emphasis*.
"""
import argparse
import html
import os
import re
import datetime

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SP, "..", "..", ".."))

OVERVIEW = """
## Where the science stands (2026-09-21)

- **Sim, atomic gate cells.** The mixed-data pin (gmsig3, two seeds) is 80/80 route-clean on the four atomic cells, 77/80 clearance-clean; the scratch pi0 control is 72/80, with the gap at center-from-right. Hand-drawn sketches fly both compounds 5/5 route-clean and track drawn polylines to about 7 cm. Autonomous compounds remain 0/5 for every arm.
- **Hardware, atomic gates (n = 5 per cell).** Real-only pin: left 5/5, right 3/5, center-from-left 0/5 (no center demos). Real-only pi0: left 4/5, right 4/5. Mixed pin (gmsig3): left 0/5, all five crashed after or before the crossing. Mixed pi0: left 4/5, right 1/5, center 0/10. The paper's "ours" arm has no valid closed-loop hardware cell; two sessions were lost to an execution-side descent. Clearance is unmeasured on hardware so far.
- **Hardware, sketches.** The real-only pin tracked the figure-eight and the 1.3 m orbit at 0.06 to 0.15 m median on the replan poses; command logs only, no trajectory files, video needed.
- **Putting a sketch into a flow.** Exact velocity projection (s = 0) carries any sketch verbatim on any flow, flaws included; SDEdit contacts gates; naive injection runs away. The trained slack (pin sigma 0.5) is the only arm that clears a sketch drawn into or around the wrong side of a post while staying 12 to 14 cm from it, and it does so with a smaller command correction (0.23 to 0.27 sigma_c) than the projection leak that also clears (s = 0.3, 0.31 to 0.32 sigma_c, 45 cm off the sketch). Slack of any kind can only spend knowledge the residual has: on the real-only flow no setting clears the center post.
- **Head uncertainty.** The GMM head is unimodal (one component at weight 1.00) at every decision point; sigma does not flag task-level failure, and the compound sentence is out of distribution by design (four trained sentences). Prompt programs cannot compose legs: past the left gate the head is state-driven.
- **Agent in the loop (sim).** A Claude reviewer approving or overriding each policy command: task-aware Claude Code 2/2 on the compound task (second flight clearance-clean at 0.257 m); Sonnet with no task knowledge 3 of 15 valid attempts, Opus 1/1 and the best flight (0.369 m clean), Haiku 0/1. Search tasks from a 180-degree start: the mannequin was found on the fourth attempt and the penguin on the first, both after side marks were drawn on the decision image. Latency per decision: Haiku 14 s, Sonnet 19 to 43 s, Opus 116 s.
- **Agent in the loop (hardware).** The first flight (agentmann_pin_01) measured PX4's setpoint tracking, not the policy: the aircraft tracked about 30% of each chunk and the node forgave the shortfall. Fixed on the workstation (hold the last setpoint, settle before the snapshot); to be reflown. A uniform 65 to 72% shortfall at sigma 0 between the asked and the returned command is open and is the primitive probe's job.

## Judging rules

Strict success = `falsify.safety.posthoc` transit judge + route-clean (no wrong-direction aperture pass) + `gate_clearance.py` at 0.18 m + human video. Screens are 5 trials; claims need 10 or more and a seed replicate (protocol noise about 5 to 6 points). Hardware success is the pilot's eyewitness note plus the judge on measured poses; contact is by eye only.
"""

PAGES = """
## Published point-cloud pages, newest first

- [Search Flights](https://claude.ai/code/artifact/cb0a2180-ed4e-4a41-9603-689899333f5a): the four mannequin flights and the penguin flight over the room, both targets marked (2026-09-21).
- [Worse Sketches](https://claude.ai/code/artifact/5775d96a-33ca-40d6-baef-8fb51357fdd0): the line grazes, enters, or misses the center post; pin sigma 0.5 vs projection s (2026-09-21).
- [What Each Mode Proposes](https://claude.ai/code/artifact/48ea2f87-6e3e-4b01-b793-a1a45304a323): the GMM head's components as commands, per replan (2026-09-21).
- [Head Uncertainty Through a Flight](https://claude.ai/code/artifact/db82a37c-4986-45d4-89a4-00e80c2af934): sigma along atomic vs compound flights (2026-09-21).
- [Bad Sketches](https://claude.ai/code/artifact/2de60aef-b9d9-4270-a986-e1b8c14d068b): the 4-click compounds passing 0.07 / 0.04 m from the post, all arms (2026-09-21).
- [Sketch Injection Three Ways](https://claude.ai/code/artifact/e95012fc-d932-4633-abd0-407faccc31b6): pin vs SDEdit vs inject vs v-proj on orbit, figure-eight, compound (2026-09-20).
- [Hardware Flights](https://claude.ai/code/artifact/1237fde9-52ec-474e-9314-fdf539aff1ce): every scored real flight, chords between measured poses (2026-09-20).
- [Hardware Results Ledger](https://claude.ai/code/artifact/0af7ce7c-8eeb-4464-b1b3-56b87123a19f): the ledger as a page (2026-09-17; superseded by the Hardware tab here).
- [Orbit radii 0.9 / 1.3 / 1.8 m](https://claude.ai/code/artifact/cb775b15-5084-4b50-8ebd-a41cadbb120e): the widened orbit sketches for the collaborator (2026-09-16).
- [Denis Figure-Eight, Third Draw](https://claude.ai/code/artifact/50f1b065-1517-4009-a0f7-393f56dd60b2), [Denis Figure-Eight Sketch](https://claude.ai/code/artifact/aa965015-6903-4fda-850d-57517622c775), [Mirrored Figure-Eight Sketch](https://claude.ai/code/artifact/c6789760-01c0-4c06-9ca0-20fd253dfc4f), [Sketchpad](https://claude.ai/code/artifact/dd2624b2-9373-4490-b8b0-360279186d7e) (2026-09-15).
- [Orbit and Figure-Eight Through Two Flows](https://claude.ai/code/artifact/400697f0-3a05-4fee-b65b-b7a7eef58c81), [Real-Only Pin, Orbit and Figure-Eight](https://claude.ai/code/artifact/810d8d52-c142-421c-9e90-aff9ee88c055), [Real Demonstrations in the Scene Clouds](https://claude.ai/code/artifact/979b6186-3e0c-4382-9a88-9cb3a47fcf7e), [Real-Only Arms, Six Cells](https://claude.ai/code/artifact/7912909d-2dba-440c-b5b7-82935b3df162) (2026-09-15).
- [Coarse-Only Swap, Six Cells](https://claude.ai/code/artifact/9ed41d6b-2565-4710-a069-52979aa1495a) (2026-09-14); [Without the Swap on Center From Right](https://claude.ai/code/artifact/88b05f04-109d-4fa6-9db7-a9d71812ca44), [Baseline on Center From Right](https://claude.ai/code/artifact/e0a3ad0c-b1b1-4336-ad90-0944ce589e02) (2026-09-10).
- [Four-Click Route, Eight of Ten Clean](https://claude.ai/code/artifact/041d4513-5831-47ba-b70e-ccfe318c6af0) (2026-09-08); [Flight Realism Ledger](https://claude.ai/code/artifact/e58991b0-ffb6-4954-8e70-54f77cb7fa4e) (2026-09-05); [Command Without Denoising](https://claude.ai/code/artifact/9c3047cc-4aae-4d89-a7e6-f4179369094d), [Gate Flight Gradebook](https://claude.ai/code/artifact/1c994b45-bfc5-4c23-be63-7a6e0d5e3ca8) (2026-09-04); [Sketch Review](https://claude.ai/code/artifact/eb9e6470-0295-4cae-9fba-14aa2961eb59) (2026-09-03).
- [Sketch Prompting Results](https://claude.ai/code/artifact/0e136b10-d8ab-4ee7-96d0-e51322264e56), [New Verbs](https://claude.ai/code/artifact/dcace32b-aefb-45b6-8a05-4bb00a6f953b) (2026-09-01); [Carrot Kick Test](https://claude.ai/code/artifact/e9eb6a24-4f4c-4a58-9bd5-9e099d500e1e), [Moved Gates](https://claude.ai/code/artifact/f5d3f184-4021-4a02-94a7-e3e981f81da2), [Pin Language](https://claude.ai/code/artifact/dc5c4332-ba00-4db2-9fcb-946a57423486) (2026-08-29); [Xswap Six Cells](https://claude.ai/code/artifact/6b3bb73b-5ce1-415e-ae09-56379ccd4d2e), [Xswap Verdict](https://claude.ai/code/artifact/f35f0ebc-d9fa-4f43-91e2-3dee8b69d552) (2026-08-28).
- [Synth Pins In Real](https://claude.ai/code/artifact/4242b0ff-c79a-4aad-abb1-929161c7ded9), [Predicted From Real](https://claude.ai/code/artifact/cb568bc4-fd0c-433a-9b12-61708f902b99), [Pin Rotation Correction](https://claude.ai/code/artifact/c65bab90-e109-4d2a-be21-47302bbfe350) (2026-08-27).

## Documents behind the tabs

- `docs/status_latest.md` (sim record, 2026-08-27), `docs/HARDWARE_RESULTS.md`, `docs/APPENDIX_INJECTION.md`, `docs/AGENT_FLIGHTS.md`, `docs/HARDWARE_AGENT_BRIEFS.md`.
- `docs/RESEARCH_LOG.md` (dense trail, newest at the bottom) and `experiments/FINDINGS_INDEX.md` (one line per finding).
"""

TABS = [("overview", "Overview", None, OVERVIEW),
        ("sim", "Sim", "docs/status_latest.md", None),
        ("hardware", "Hardware", "docs/HARDWARE_RESULTS.md", None),
        ("injection", "Injection", "docs/APPENDIX_INJECTION.md", None),
        ("agent", "Agent flights", "docs/AGENT_FLIGHTS.md", None),
        ("pages", "Pages", None, PAGES)]


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)\*(?!\w)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    return s


def md_to_html(text):
    out, lines, i = [], text.split("\n"), 0
    para, bullets = [], []

    def flush():
        nonlocal para, bullets
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>"); para = []
        if bullets:
            out.append("<ul>" + "".join("<li>" + inline(b) + "</li>" for b in bullets) + "</ul>"); bullets = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush()
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            flush(); lvl = len(m.group(1))
            if lvl == 1: i += 1; continue          # the file title is the tab's name
            out.append(f"<h{lvl + 1}>{inline(m.group(2))}</h{lvl + 1}>"); i += 1; continue
        if ln.startswith("- "):
            if para: flush()
            bullets.append(ln[2:].strip()); i += 1
            while i < len(lines) and lines[i].startswith("  ") and not lines[i].startswith("- "):
                bullets[-1] += " " + lines[i].strip(); i += 1
            continue
        if not ln.strip():
            flush(); i += 1; continue
        if bullets and ln.startswith("  "):
            bullets[-1] += " " + ln.strip(); i += 1; continue
        if bullets: flush()
        para.append(ln.strip()); i += 1
    flush()
    return "\n".join(out)


CSS = """
:root{--paper:#f6f4ee;--panel:#fffdf8;--ink:#1f2430;--ink2:#4c5260;--muted:#7a8090;--rule:#d8d2c4;--accent:#176f68;--accent-ink:#0f4d48;--band:#ecf5f3;--head:#efece4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#15181e;--panel:#1c2027;--ink:#e6e3dc;--ink2:#b9bcc4;--muted:#858b98;--rule:#333944;--accent:#4fbdb3;--accent-ink:#8fdcd4;--band:#1d2b2a;--head:#242932}}
:root[data-theme="dark"]{--paper:#15181e;--panel:#1c2027;--ink:#e6e3dc;--ink2:#b9bcc4;--muted:#858b98;--rule:#333944;--accent:#4fbdb3;--accent-ink:#8fdcd4;--band:#1d2b2a;--head:#242932}
body{background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.5;margin:0}
header{padding:22px 28px 0;border-bottom:1px solid var(--rule);background:var(--panel)}
header h1{margin:0 0 4px;font-size:22px;font-weight:600;letter-spacing:-0.01em}
header .sub{color:var(--muted);font-size:13px;margin:0 0 14px}
nav{display:flex;gap:4px;flex-wrap:wrap}
nav button{background:none;border:0;border-bottom:3px solid transparent;color:var(--ink2);font:inherit;font-size:14px;padding:8px 12px;cursor:pointer;border-radius:4px 4px 0 0}
nav button:hover{color:var(--ink);background:var(--band)}
nav button[aria-selected="true"]{color:var(--accent-ink);border-bottom-color:var(--accent);font-weight:600}
nav button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
main{padding:20px 28px 60px;max-width:1180px}
section[hidden]{display:none}
.src{color:var(--muted);font-size:12.5px;margin:0 0 18px;font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
h2{font-size:18px;font-weight:600;margin:26px 0 8px;letter-spacing:-0.005em;text-wrap:balance}
h3{font-size:15.5px;font-weight:600;margin:20px 0 6px;color:var(--ink2)}
h4{font-size:14px;font-weight:600;margin:16px 0 4px;color:var(--ink2)}
p{max-width:78ch;margin:0 0 10px}
ul{max-width:82ch;padding-left:20px;margin:0 0 12px}
li{margin:0 0 6px}
code{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:var(--head);padding:1px 4px;border-radius:3px}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:2px}
.tw{overflow-x:auto;margin:8px 0 16px;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;font-size:13px;min-width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{background:var(--head);font-weight:600;white-space:nowrap;position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--band)}
strong{font-weight:600}
@media (max-width:640px){header,main{padding-left:14px;padding-right:14px}}
"""

JS = """
const tabs=[...document.querySelectorAll('nav button')],secs=[...document.querySelectorAll('main section')];
function show(id,push){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===id));secs.forEach(s=>s.hidden=s.id!==id);
  try{localStorage.setItem('snmvp-tab',id)}catch(e){} if(push){history.replaceState(null,'','#'+id)} }
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab,true)));
let start=location.hash.slice(1); if(!secs.some(s=>s.id===start)){try{start=localStorage.getItem('snmvp-tab')}catch(e){}}
if(!secs.some(s=>s.id===start))start=secs[0].id; show(start,false);
window.addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(secs.some(s=>s.id===h))show(h,false)});
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results_dashboard.html"); a = ap.parse_args()
    today = datetime.date.today().isoformat()
    nav, body = [], []
    for tid, name, src, text in TABS:
        if src:
            text = open(os.path.join(ROOT, src)).read()
            mtime = datetime.date.fromtimestamp(os.path.getmtime(os.path.join(ROOT, src))).isoformat()
            srcline = f"{src} · file last modified {mtime}"
        else:
            srcline = f"inline in build_results_dashboard.py · built {today}"
        nav.append(f'<button role="tab" data-tab="{tid}" aria-selected="false">{html.escape(name)}</button>')
        body.append(f'<section id="{tid}" role="tabpanel" hidden><p class="src">{html.escape(srcline)}</p>{md_to_html(text)}</section>')
    page = ("<title>Source-Noise Results</title>\n"
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">\n'
            f"<style>{CSS}</style>\n"
            "<header><h1>Source-noise action steering: results</h1>"
            f'<p class="sub">Every ledger in one place, one tab per line of work. Rebuilt {today} from the docs named at the top of each tab.</p>'
            f'<nav role="tablist">{"".join(nav)}</nav></header>\n<main>{"".join(body)}</main>\n<script>{JS}</script>\n')
    open(os.path.join(SP, a.out), "w").write(page); print("wrote", a.out, len(page), "bytes")


main()
