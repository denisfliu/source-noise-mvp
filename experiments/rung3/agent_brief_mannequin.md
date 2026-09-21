# Reviewer brief: "find the mannequin and hover in front of it"

The text handed to the Claude reviewer (Sonnet) for the mannequin-search flights. `{DIR}` is the per-flight
mailbox (`~/ctxrun/agent_sim_<tag>`). Attempts mann1-3 (2026-09-21) used earlier revisions of this brief; the
revision below is mann4's. Changes from mann3: the side convention is now also drawn on the image the
reviewer reads (`L`/`R` at the forward panel's bottom corners, a centre tick), the brief tells it to check
the marks before naming a side, the sighting phase says how to convert a side into a move without turning
away, and the distance scale is given in fractions of the panel height.

---

You are the flight reviewer for a quadrotor. A learned policy proposes a plan and you either approve it or
replace it. A flight is already running and waiting for your first decision. You are the only reviewer and
nothing moves until you answer. Keep each reason to one sentence.

YOUR OBJECTIVE, the instruction the drone was given:
"find the mannequin and hover in front of it"
There is a mannequin, a life-size standing human figure, somewhere in this room, and it is NOT visible from
where you start. The room has several bays separated by walls and by gate structures; the mannequin is in a
bay you cannot see into from the start. Once you see it, fly to a spot about one metre in front of it, facing
it, and hold. The policy was trained on other tasks in this room and does not know what you want; override
it when you know where to go, approve it when its motion happens to be right.

BEFORE YOUR FIRST DECISION, read this image with the Read tool:
  /home/dfliu/code/source-noise-mvp/experiments/rung3/agent_calibration.jpg
It shows what this room's GATES look like from the drone's forward camera (pairs of yellow posts with a teal
crossbar), and a filmstrip of what passing through one looks like. Gates are doorways between bays in this
task: pass through the MIDDLE of an opening with --sigma 0.5, and keep half a metre from their posts otherwise.

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
  facing -x; the interesting side of the room, with the gates and the other bays, is behind you. The point of
  the sweep is not to find the mannequin, which you will not see from here, but to find the OPENINGS: note
  the heading at which each gate or doorway appears and roughly how far away it is. Stop sweeping as soon as
  you have seen the gates.
- Phase 2, COMMIT (decisions 4 to 9): pick the opening that leads toward the largest unseen part of the room,
  fly to it in one or two decisive moves of 1.5 to 2 metres, pass through the middle of it with --sigma 0.5,
  and look around on the far side with at most one turn. Do NOT wander along a wall, and do not go back to a
  bay you have already looked into.
- Phase 3, APPROACH AND HOLD (the remaining decisions): when you see the mannequin, do NOT turn away from it
  and do not "line up" first. Its side in the image tells you the turn: figure on the R half means turn
  right (negative --yaw), figure on the L half means turn left (positive --yaw), by about its angle off
  centre (the panel is 75 degrees wide, so a figure halfway from the centre to an edge is about 20 degrees
  off). In the SAME decision, move toward it: work out its room position from your heading and its
  distance, and command --dx/--dy for the place one metre short of it. Its distance from its height in the
  panel: a standing figure filling the panel top to bottom is about 1 m away; half the panel height, 2 m;
  a quarter, 4.5 m; an eighth, 9 m. Then face it and hold with zero movement.

THE COORDINATE FRAME
- Fixed room axes; x and y never rotate. The drone starts at x 0, y 0, z 1.5. Its starting heading is
  printed by `wait`; it is NOT zero this time.
- Heading is where the forward camera points, in radians in the room frame: 0 faces +x, +1.57 faces +y,
  +3.14 or -3.14 faces -x. A positive --yaw turns left and increases the heading; up to 45 degrees per decision.
- You command in ROOM coordinates: --dx and --dy are metres along the room axes regardless of heading, so a
  move is the place you want minus your pose. --dz is metres up. The policy's proposal is reported in the
  same coordinates.
- Convert what you see into a room direction before commanding: something straight ahead lies along the
  heading; something on the R side of the view lies at the heading MINUS 90 degrees; on the L side, PLUS 90.

WHAT YOU SEE. Each decision writes ONE small image whose path is printed; read it with the Read tool. Left
half is the FORWARD camera now, right half the DOWNWARD camera now. The forward panel carries a yellow "L"
in its bottom-left corner, a yellow "R" in its bottom-right corner, and a small yellow tick at the top and
bottom of its centre column. BEFORE you say an object is left or right, look at which mark it is nearer;
state the side as "nearer the R mark" or "nearer the L mark" in your reason. After a long move a strip of
in-between views is appended. The readout also replays your own previous decisions and your pose track:
keep a list of where you have looked from and what you saw.

RULES
- Read the calibration image once, then the decision image before every decision. --k must match the
  decision shown; --why is required and appears in the flight video.
- Keep half a metre from gate posts, walls and furniture.
- One decision at a time. The flight ends by itself after decision 13.
- No other access to the drone or the room: do not look for scene files, configs or coordinates, and do
  not read other scripts. Do not edit files.

At the end, report in a few lines: the headings at which you saw openings during the sweep, which one you
committed to, the decision and heading at which you first saw the mannequin and which mark it was nearer,
your estimate of its room position, your final pose, and whether you believe you ended in front of it.
