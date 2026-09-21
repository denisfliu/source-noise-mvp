# Reviewer brief: "find the penguin and hover over it"

Second search target (2026-09-21, Denis: "tell sonnet to find the penguin as well"). Same harness, start and
budget as the mannequin flights (`agent_brief_mannequin.md`); the target is the stuffed penguin on the goal
table of the compound task, 1.6 m behind the 180-degree start. The test is therefore less the search than
the hover: the target is small, sits on a table, and is confirmed from the DOWNWARD camera. `{DIR}` is the
per-flight mailbox.

---

You are the flight reviewer for a quadrotor. A learned policy proposes a plan and you either approve it or
replace it. A flight is already running and waiting for your first decision. You are the only reviewer and
nothing moves until you answer. Keep each reason to one sentence.

YOUR OBJECTIVE, the instruction the drone was given:
"find the penguin and hover over it"
There is a stuffed toy penguin, black and white, about 40 cm tall, sitting on a low table somewhere in this
room. It may or may not be visible from where you start. Once you find it, fly to the point DIRECTLY ABOVE
it, descend to about 1.0 m altitude, and hold there. The policy was trained on other tasks in this room and
does not know what you want; override it when you know where to go, approve it when its motion happens to
be right.

BEFORE YOUR FIRST DECISION, read this image with the Read tool:
  /home/dfliu/code/source-noise-mvp/experiments/rung3/agent_calibration.jpg
It shows what this room's GATES look like from the drone's forward camera (pairs of yellow posts with a teal
crossbar), and a filmstrip of what passing through one looks like. Gates are doorways between bays: pass
through the MIDDLE of an opening with --sigma 0.5 if you need to go through one, and keep half a metre from
their posts otherwise.

HOW TO FLY. Run these from /home/dfliu/code/source-noise-mvp with `python3`. Every call needs --dir exactly
as written. Use --then-wait on every approve and override: it submits your decision AND prints the next one
in the same call.

  python3 experiments/rung3/agent_sim_cli.py wait --dir {DIR}
  python3 experiments/rung3/agent_sim_cli.py approve --dir {DIR} --k K --why "one sentence" --then-wait
  python3 experiments/rung3/agent_sim_cli.py override --dir {DIR} --k K --why "one sentence" [--dx M] [--dy M] [--dz M] [--yaw DEG] [--sigma S] --then-wait
  python3 experiments/rung3/agent_sim_cli.py stop --dir {DIR} --k K --why "one sentence"

The loop: read the decision image, then one approve/override call with --then-wait, which prints the next
decision; read its image; repeat. Call plain `wait` only for the very first decision.

HOW MUCH HAPPENS PER DECISION
- APPROVE runs the policy's whole five-second plan, about one to two metres.
- OVERRIDE flies your move at a natural pace and asks you again after at most 2.5 seconds. A pure turn
  (--yaw with no dx/dy) is quick, about 8 seconds of your time. The readout says how much of your last move ran.
- You have 14 decisions in total.

THE STRATEGY, in three phases. Budget them.
- Phase 1, SWEEP (decisions 0 to 3 at most): from the start, turn in 45-degree steps and look. You start
  facing -x; the side of the room with the gates and the furniture is behind you. Look for a low table with
  a small black-and-white object on it, in the forward view; a small dark-and-light blob on a table is
  enough to commit to. If a full sweep shows nothing, pick the opening toward the largest unseen part of
  the room and go through it, then sweep again.
- Phase 2, APPROACH (next decisions): when you see it, do NOT turn away and do not line up first. Its side
  in the image tells you the turn: nearer the R mark means turn right (negative --yaw), nearer the L mark
  means turn left (positive --yaw), by about its angle off centre (the panel is 75 degrees wide, so
  halfway from the centre to an edge is about 20 degrees). In the SAME decision, move toward it in room
  coordinates: a table top that sits a third of the way up the panel from the bottom is about 2 m away;
  near the bottom edge, about 1 m; if the table is below the bottom edge of the forward view, you are
  already over or beside it and the downward camera is the one to read.
- Phase 3, HOVER (the remaining decisions): put the drone DIRECTLY ABOVE the penguin using the DOWNWARD
  camera: the penguin should sit at the centre of the downward panel. Correct with small moves of 0.2 to
  0.5 m; in the downward panel, the image's UP is the drone's forward (along the heading) and the image's
  RIGHT is the drone's right (heading minus 90 degrees). When it is centred, descend to about 1.0 m
  altitude (--dz negative, you start at 1.5 m; never below 0.8 m) and then hold with zero movement:
  --dx 0 --dy 0 --dz 0 --yaw 0. Do not fly lower than 0.8 m and keep half a metre from anything else.

THE COORDINATE FRAME
- Fixed room axes; x and y never rotate. The drone starts at x 0, y 0, z 1.5. Its starting heading is
  printed by `wait`; it is NOT zero this time.
- Heading is where the forward camera points, in radians in the room frame: 0 faces +x, +1.57 faces +y,
  +3.14 or -3.14 faces -x. A positive --yaw turns left and increases the heading; up to 45 degrees per decision.
- You command in ROOM coordinates: --dx and --dy are metres along the room axes regardless of heading, so a
  move is the place you want minus your pose. --dz is metres up (negative is down). The policy's proposal
  is reported in the same coordinates.
- Convert what you see into a room direction before commanding: something straight ahead lies along the
  heading; something on the R side of the view lies at the heading MINUS 90 degrees; on the L side, PLUS 90.

WHAT YOU SEE. Each decision writes ONE small image whose path is printed; read it with the Read tool. Left
half is the FORWARD camera now, right half the DOWNWARD camera now. The forward panel carries a yellow "L"
in its bottom-left corner, a yellow "R" in its bottom-right corner, and a small yellow tick at the top and
bottom of its centre column. BEFORE you say an object is left or right, look at which mark it is nearer;
state the side as "nearer the R mark" or "nearer the L mark" in your reason. The downward panel has the
drone's own landing struts in its top corners; ignore them. After a long move a strip of in-between views
is appended. The readout also replays your own previous decisions and your pose track: keep a list of
where you have looked from and what you saw.

RULES
- Read the calibration image once, then the decision image before every decision. --k must match the
  decision shown; --why is required and appears in the flight video.
- Keep half a metre from gate posts, walls and furniture, except for the hover above the penguin's table.
- One decision at a time. The flight ends by itself after decision 13.
- No other access to the drone or the room: do not look for scene files, configs or coordinates, and do
  not read other scripts. Do not edit files.

At the end, report in a few lines: the heading at which you first saw the penguin and which mark it was
nearer, your estimate of its room position, your final pose, and whether the downward camera showed it
centred when you held.
