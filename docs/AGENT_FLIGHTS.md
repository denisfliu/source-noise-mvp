# Agent-reviewed flights, aggregated (as of 2026-09-21)

Every simulator flight in which a Claude reviewer approved or overrode the policy's command at each replan.
Harness: `scripts/run_agent_flight.sh` (gmsig3 server, per-flight mailbox `~/ctxrun/agent_sim_<tag>`),
reviewer CLI `experiments/rung3/agent_sim_cli.py`, primitives `experiments/rung3/agent_prompt.py`. Per-flight
decisions (JSON, images) are archived under `experiments/rung3/agentflight/<tag>_*_decisions/`, trajectories
as `experiments/rung3/agentflight/traj_<tag>.npy`, videos as `~/ctxrun/agent_<tag>.mp4` (every scored video was
sent to Denis). Dense narrative: `docs/RESEARCH_LOG.md` from the 2026-09-20 "agent-reviewed flight" entries
onward; one-liners in `experiments/FINDINGS_INDEX.md`.

Where the other results live:
- Real-drone flights: `docs/HARDWARE_RESULTS.md` (ledger) and `experiments/rung3/viz/hw_flights.html`.
- Injection / velocity-projection / bad-sketch / compliance series: `docs/APPENDIX_INJECTION.md` (sections A-E)
  with pages `vproj_compare.html`, `badsketch.html`, `worsesketch.html` under `experiments/rung3/viz/`.
- GMM uncertainty and modes on the compound task: `uncert.html`, `modes.html`.
- Sim record board and atomics: `docs/status_latest.md` (2026-08-27) and the RECORD BOARD in `CLAUDE.md`.

## Compound task: "go through the left gate, then the centre gate, then hover over the table"

Judge: `falsify.safety.posthoc` gates in order + goal dwell; clearance from `gate_clearance.py` (body 0.18 m,
clean means minimum over the flight >= 0.18 m). Scene `left_and_center`, start (0, 0, 1.5) heading 0.

| tag | reviewer | decisions | what changed | gates | dwell | min clearance | verdict |
|---|---|---|---|---|---|---|---|
| center1 | Claude Code (task-aware) | 10 | first reviewed flight | 2/2 | yes | 0.126 m | success, not clean |
| center2 | Claude Code (task-aware) | 10 | authored lines through aperture centres | 2/2 | yes | 0.257 m | success, clean |
| center3 | Sonnet, no task knowledge | 10 | bare interface | 2/2 | 0 | - | fail |
| center4 | - | - | two reviewers on one mailbox | - | - | - | invalid |
| center5 | Sonnet | 10 | same brief | 0/2 | - | - | fail (0.29 m from goal) |
| center6 | Sonnet | 10 | map-keeping prompt | 0/2 | - | - | fail, self-report wrong |
| center7 | Sonnet | 10 | approve by default, back up at one post | 2/2 | 51 | 0.099 m | success, not clean |
| center8 | Sonnet | 10 | hover rule (near-null command = hover) | 0/2 | - | - | fail (0.07 m from goal) |
| center9 | Sonnet | 10 | resolve the whole gate first, slack off | 0/2 | - | - | fail (0.16 m from goal) |
| center10 | Sonnet | 10 | filmstrip of the executed chunk (5 frames) | 1/2 | - | - | fail (0.07 m from goal) |
| center11 | Sonnet | 10 | adaptive filmstrip (<= 8), goal-vs-gate override rule | 2/2 | 73 | < 0.18 m | success, not clean |
| center12 | Sonnet | 10 | centring rules (aim middle, square up, no yaw crossing) | 2/2 | 0 | 0.006 m | fail |
| center13 | Opus | 10 | same brief as center12 | 2/2 | 96 | 0.369 m | success, clean (best flight) |
| center14 | Sonnet | 10 | room-frame --dx/--dy commands | 1/2 | - | 0.02 m | fail |
| center15 | Sonnet | 26 | short chunks (APC 15; exposed the scaling bug) | 1/2 | - | - | fail |
| center16 | Sonnet | 14 | adaptive execution + decision-log replay | 0/2 | - | - | fail (0.80 m from goal) |
| center17 | Sonnet | 14 | calibration frames sheet | 2/2 | 159 | 0.136 m | success, not clean (0.03 m from goal) |
| center18 | Haiku | 14 | same brief as center17 | 1/2 | - | - | fail, self-report wrong |

Reviewer latency per decision (median): Haiku 14 s, Sonnet 27-43 s (19 s on the mannequin brief), Opus 116 s.
Sonnet successes: center7, center11, center17 (3 of 15 valid attempts); Opus 1 of 1; Haiku 0 of 1.

## Search task: "find the mannequin and hover in front of it"

Start (0, 0, 1.5) heading -pi (facing away from the gates). Target: a 1.7 m figure at (7.4, -0.2) against the
back wall behind the wooden table, beyond the centre gate; not visible from the start. No judge; scored by
closest approach to the figure from the trajectory. 14 decisions each.

| tag | reviewer | brief | outcome | closest approach |
|---|---|---|---|---|
| mann1 | Sonnet | gate brief + objective | swept 360 deg in place, never moved toward the gates | 7.3 m |
| mann2 | Sonnet | + explore rule | explored the wrong half (-x bay), never turned | 7.5 m |
| mann3 | Sonnet | sweep-then-commit phases | found the gate by decision 4, sighted the figure at 11, mirrored its side and flew away | 4.0 m |
| mann4 | Sonnet | + L/R marks on the image, approach-without-turning-away, distance scale | SUCCESS: sighted at decision 8, ended 1.02 m in front, 14 deg off axis | 1.0 m |


## Search task: "find the penguin and hover over it"

Same start. Target: the stuffed penguin on the compound task's goal table at (1.525, -0.615), 1.6 m behind the
start. Scored by the judge's goal box (0.6 x 0.6 x 1.0 m about the penguin) and final xy offset.

| tag | reviewer | brief | outcome | final xy offset | in goal box |
|---|---|---|---|---|---|
| peng1 | Sonnet | mannequin brief + downward-camera hover phase | SUCCESS: found in the downward camera at decision 10, centred, descended to 1.0 m, held | 0.11 m | last 30 steps |

Briefs: `experiments/rung3/agent_brief_mannequin.md`, `experiments/rung3/agent_brief_penguin.md`. Cloud page: `experiments/rung3/viz/mannequin.html`.
