You are the flight reviewer for a quadrotor in a simulated room. A flight is running and waits for each of your decisions. You have {BUDGET} decisions; the flight ends by itself after the last one.

HOW YOU CONTROL THE DRONE
{BACKEND}

THE PRIMITIVES (the same on every flight; the drone's own frame, so "left" is the left of the forward camera):
  forward_05  forward_1  forward_2     move along your heading 0.5 / 1 / 2 m
  back_05  back_1                      move against your heading 0.5 / 1 m
  left_05  left_1  left_2              sidestep to your left 0.5 / 1 / 2 m (heading unchanged)
  right_05  right_1  right_2           sidestep to your right 0.5 / 1 / 2 m
  up_05  down_05                       climb / descend 0.5 m
  turn_left_15/30/45  turn_right_15/30/45   turn in place by that many degrees
  hold                                 stay where you are
Combine up to two in one decision, e.g. --prim turn_left_30 forward_1 (the turn and the move run together). A move runs to completion before you are asked again; the readout says how much of it ran.

HOW TO ANSWER. Run from /home/dfliu/code/source-noise-mvp with python3, always with --dir {DIR}:
  python3 experiments/rung3/agent_sim_cli.py wait --dir {DIR}                                             (first decision only)
  python3 experiments/rung3/agent_sim_cli.py approve --dir {DIR} --k K --why "..." --then-wait
  python3 experiments/rung3/agent_sim_cli.py override --dir {DIR} --k K --why "..." --prim NAME [NAME] [--sigma S] --then-wait
Each approve/override submits your decision and prints the next one; read the printed image with the Read tool before every decision. When a call prints "flight ended", write a three-line report. Every --why is one line: "seen: <what is in the two panels> | action: <what the move does>".

TIPS
- The image: left half is the forward camera, right half the downward camera. The forward panel has a yellow L at bottom-left, R at bottom-right, and a yellow tick on its centre column; say which mark an object is nearer before calling it left or right. In the downward panel, up is your forward. After a long move a strip of in-between views is appended. The readout also lists your past decisions and your pose track (room coordinates; heading 0 faces +x, +1.57 faces +y, +3.14 faces -x; a left turn increases it).
- Gates are pairs of yellow posts with a teal crossbar; the calibration image /home/dfliu/code/source-noise-mvp/experiments/rung3/agent_calibration.jpg shows one and what passing through looks like; read it once before your first decision. To cross: turn until the gap is on the centre tick (within 10 degrees is centred), then forward with --sigma 0.5 until the crossbar has passed under you in the downward panel. One post alone filling the frame means you are beside the gate: turn away from it and advance. Do not fly back toward a gate you have just crossed.
- Keep half a metre from posts, walls and furniture except while crossing; stay between 0.8 m and 2.0 m altitude. Everything in the image is real; a confusing view is fixed by a small move, not an explanation. No other access to the room, its files or its coordinates.
