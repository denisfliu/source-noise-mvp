"""Results dashboard: one tab per line of work, built from the ledgers so it never drifts from them
(2026-09-21; Denis: "an artifact that I can continually reference for all of our results ... click on a tab
to see results for each thing, e.g. hardware, sim").

  python build_results_dashboard.py --out results_dashboard.html

Tabs and their sources (the Sim tab is inline tables followed by the status file; a 'tables only' switch, on by
default, hides paragraphs and bullets: Denis reads tables, not words):
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
## Results at a glance (2026-09-21)

| Line | Arm | Cell(s) | n | Result | Tab |
|---|---|---|---|---|---|
| Sim atomics | mixed pin, gmsig3 + seed-7 twin | left, right, CFL, CFR | 80 | 80/80 route-clean judge, 77/80 clearance-clean | Sim |
| Sim atomics | xswap (two seeds) | same four | 80 | 80/80 judge, 80/80 clearance-clean | Sim |
| Sim atomics | scratch pi0 (two seeds) | same four | 80 | 72/80 judge; CFR 7/10 on both seeds | Sim |
| Sim compounds, autonomous | every arm, both seeds | CMPL, CMPR | 5 each | 0/5 | Sim |
| Sim compounds, sketched | gmsig3 + hand-drawn sketch | CMPL, CMPR | 5 each | 5/5 route-clean each; CMPL 5/5 clearance; tracking ~7 cm | Sim |
| Sim sketches, pooled | xswap s42 + s7 | all sketch rows | 50 | 49/50 route-clean; clearance sketch-geometry-bound | Sim |
| Cross-domain | synth-authored pins on real observations | right-gate crossings | 11 | 11/11 in-aperture | Sim |
| Hardware atomics | real-only pin | left / right / CFL | 5 each | 5/5 / 3/5 / 0/5 through (eyewitness) | Hardware |
| Hardware atomics | real-only pi0 | left / right | 5 each | 4/5 / 4/5 through, one crash each | Hardware |
| Hardware atomics | mixed pin, gmsig3 | left | 5 | 3/5 through, 5/5 contact (all crashed) | Hardware |
| Hardware atomics | mixed pi0, scratch3 | left / right / CFL / CFR | 5 each | 4/5 / 1/5 / 0/5 / 0/5 through | Hardware |
| Hardware atomics | ours (xswap) | left | 5 | 0/5, execution-side failure; no valid cell | Hardware |
| Hardware sketches | real-only pin | fig8_denis3 / orbit_wide | 2 / ~5 | 0.10 m / 0.06-0.15 m median tracking on replan poses; logs only | Hardware |
| Hardware clearance | all | all | - | unmeasured (setpoint records before c933466) | Hardware |
| Injection, authored routes | v-proj s=0 vs pin vs SDEdit vs inject | orbit, fig8, compound | 5 each | s=0 tracks at 1 cm, 5/5 clean; pin 5-12 cm; SDEdit clips; inject runs away | Injection |
| Injection, worse sketches | pin sigma 0.5 vs v-proj s=0.3 | w2, w3, w4 | 5 each | pin 3-4/5 clean at 12-14 cm tracking; s=0.3 5/5 clean at 45 cm | Injection |
| Compliance | pin sigma 0.5 vs v-proj s=0.3 | 0.07 m, w2-w4 | 5 each | 0.22-0.27 sigma_c vs 0.31-0.35 sigma_c | Injection |
| Head uncertainty | gmsig3 GMM | atomic vs compound | - | unimodal (weight 1.00) at every decision; sigma does not flag task failure | Sim |
| Agent, compound task | Claude Code (task-aware) | left -> center -> hover | 2 | 2/2 success, second clearance-clean 0.257 m | Agent flights |
| Agent, compound task | Sonnet / Opus / Haiku (no task knowledge) | same | 15 / 1 / 1 | 3/15 / 1/1 (0.369 m clean) / 0/1 | Agent flights |
| Agent, search | Sonnet | mannequin / penguin | 4 / 1 | found on attempt 4 (1.0 m in front) / attempt 1 (0.11 m off, 1.0 m up) | Agent flights |
| Agent latency | Haiku / Sonnet / Opus | per decision, median | - | 14 s / 19-43 s / 116 s | Agent flights |
| Agent, hardware | Sonnet on the real drone | mannequin | 1 | invalid: aircraft tracked ~30% of each chunk; node fix applied; refly | Hardware |

## Judging rules

| Tier | Rule |
|---|---|
| Strict sim success | `falsify.safety.posthoc` transit judge + route-clean (no wrong-direction aperture pass) + `gate_clearance.py` >= 0.18 m + human video |
| Statistics | screens 5 trials; claims >= 10 trials and a seed replicate (protocol noise about 5-6 points) |
| Hardware success | pilot's eyewitness note + judge on measured poses; contact by eye only; goal box reported, not used |
"""

SIM_TABLES = """
## Atomic gate cells, per arm and seed (route-clean judge / clearance-clean, out of 40 per seed)

| Arm | Seed 42 | Seed 7 | Pooled | Note |
|---|---|---|---|---|
| mixed pin, no swap (gmsig3) | 40/40 / 40/40 | 40/40 / 37/40 | 80/80 / 77/80 | seed-7 grazes are 3 CFR |
| xswap (coarse-only swap) | 40/40 / 40/40 | 40/40 / 40/40 | 80/80 / 80/80 | dropped from the paper (swap) |
| scratch pi0 | 36/40 | 36/40 | 72/80 | CFR 7/10 on both seeds |

## Compounds and sketches (5 trials per cell)

| Cell | Arm | Route-clean | Clearance-clean | Note |
|---|---|---|---|---|
| CMPL / CMPR autonomous | every arm, both seeds | 0/5 | - | structural: the head plans goal-first past the first gate |
| CMPL, hand-drawn sketch | gmsig3 | 5/5 | 5/5 | corrective 6-waypoint polyline, sigma 0 over the switch |
| CMPR, hand-drawn sketch | gmsig3 | 5/5 | - | |
| hand-drawn sketches, pooled | gmsig3 | - | 17/20 | |
| 4-click minimal sketch | gmsig3, sigma 0 / sigma 0.5 | - | 2/10 / 9/10 | margin is the portability budget |
| all sketch rows, pooled | xswap s42 + s7 | 49/50 | sketch-geometry-bound | reproduces every gmsig3 sketch row within one trial |

## Cross-domain and command vocabulary

| Test | Result |
|---|---|
| synth-authored pins (oracle and sim-twin head) on real observations | 11/11 in-aperture right-gate crossings, full speed |
| real demo replayed as a pin sketch in sim | 5/5 strict clean |
| rotation verb as aim correction | heading error 10-20 deg -> 3-5 deg, dose gain 0.76 |
| language alone as redirect | 0.05 cstd contrast, both domains (cannot redirect) |
| pin-gap triplet (sim-to-real) | execution gap ~0; prediction gap at the head's floor; behavior gap 0.62 cstd in the endgame |
| flight realism (AUC vs real demos; 0.5 = indistinguishable) | command alone: velocity staircase; pin flight closest to real on CFR/CMPL; synth demos 0.999 (planner staircase) |

## Head uncertainty (gmsig3 GMM head)

| Observation | Value |
|---|---|
| components active at decision points | one, weight 1.00, every prompt, replans 2-4 |
| sigma vs task-level failure | uncorrelated; goal-hover is an absorbing state |
| trained sentences | four (left, right, center-from-left, center-from-right); the compound sentence is OOD by design |
| prompt programs (advance on declared intent) | cannot compose legs; state-dominated past the left gate |
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
        ("sim", "Sim", "docs/status_latest.md", SIM_TABLES),
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
nav{display:flex;gap:4px;flex-wrap:wrap;align-items:center}
.sw{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:13px;color:var(--ink2);padding:6px 4px}
.sw input{accent-color:var(--accent)}
body.tables-only main p:not(.src),body.tables-only main ul{display:none}
body.tables-only main h3,body.tables-only main h4{margin-top:14px}
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
const sw=document.getElementById('tonly');let to='1';try{to=localStorage.getItem('snmvp-tables-only')??'1'}catch(e){}
sw.checked=to==='1';document.body.classList.toggle('tables-only',sw.checked);
sw.addEventListener('change',()=>{document.body.classList.toggle('tables-only',sw.checked);try{localStorage.setItem('snmvp-tables-only',sw.checked?'1':'0')}catch(e){}});
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
            text = (text or "") + open(os.path.join(ROOT, src)).read()   # inline tables first, then the ledger
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
            f'<nav role="tablist">{"".join(nav)}<label class="sw"><input type="checkbox" id="tonly" checked> tables only</label></nav></header>\n<main>{"".join(body)}</main>\n<script>{JS}</script>\n')
    open(os.path.join(SP, a.out), "w").write(page); print("wrote", a.out, len(page), "bytes")


main()
