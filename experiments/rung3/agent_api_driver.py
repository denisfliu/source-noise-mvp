#!/usr/bin/env python3
"""Direct-API flight reviewer (2026-09-22): the same mailbox loop as agent_sim_cli.py, but each decision is one
model call with the brief as system prompt, the calibration sheet and the decision image attached, and a JSON
answer. No tool calls, no running conversation: every decision sees the same fixed context (brief + readout +
its own previous decisions + pose track), so a 24-decision flight cannot drift.

  GEMINI_API_KEY=... python agent_api_driver.py --dir ~/ctxrun/agent_sim_<tag> --brief brief.md [--model gemini-3.6-flash]
  python agent_api_driver.py --dir DIR --brief brief.md --dry     # no API: approve/override alternately, prints the readout

Provider is behind one function (ask_gemini); add another provider by adding a function with the same signature.
Log: DIR/api_log.jsonl, one line per decision (readout, raw reply, parsed command, latency)."""
import argparse, glob, json, math, os, sys, time

# the calibration sheet at 960 px wide: fewer image tiles per call than the 1250 px original, captions still legible
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_calibration_api.jpg")
LIM = {"dx": 2.0, "dy": 2.0, "dz": 1.0, "yaw": 45.0}
SCHEMA = {"type": "OBJECT", "required": ["verdict", "why"],
          "properties": {"verdict": {"type": "STRING", "enum": ["approve", "override", "stop"]}, "why": {"type": "STRING"},
                         "dx": {"type": "NUMBER"}, "dy": {"type": "NUMBER"}, "dz": {"type": "NUMBER"},
                         "yaw": {"type": "NUMBER"}, "sigma": {"type": "NUMBER"}}}


def latest(d):
    p = os.path.join(d, "latest.json")
    for _ in range(6):
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p))
        except json.JSONDecodeError:
            time.sleep(0.05)
    return None


def readout(o, d):
    """The text the CLI prints for a decision (agent_sim_cli.show), as a string."""
    p = o["proposal"]; L = []
    L.append(f"decision k={o['k']}  pose x={o['pose'][0]:.2f} y={o['pose'][1]:.2f} z={o['pose'][2]:.2f} "
             f"heading={o['pose'][3]:.2f} rad ({o['pose'][3]*57.3:+.0f} deg; yaw is in this same sense, positive turns left)")
    L.append(f"the policy intends, over the next {o['seconds_per_decision']} s (the part of its plan that will actually execute before you are asked again):")
    L.append(f"  net move  dx {p['net_xyz'][0]:+.2f}  dy {p['net_xyz'][1]:+.2f}  dz {p['net_xyz'][2]:+.2f} m,  yaw {p['net_yaw_deg']:+.1f} deg")
    L.append(f"  ending at x {p['path_end'][0]:.2f}  y {p['path_end'][1]:.2f}  z {p['path_end'][2]:.2f}   (peak speed {p['speed_max_mps']:.2f} m/s, trust {p['sigma_serve']:.2f})")
    if o.get("last_execution"):
        L.append(f"  your last command: {o['last_execution']}")
    past = sorted(glob.glob(os.path.join(d, "cmds", "*.json")))
    if past:
        L.append("what you have decided so far (your own words):")
        for f in past:
            try:
                c = json.load(open(f))
            except json.JSONDecodeError:
                continue
            mv = c.get("move") or {}
            terse = " ".join(f"{k.replace('_deg', '')} {v:+g}" for k, v in mv.items() if v)
            L.append(f"    k={c.get('k')} {c.get('verdict', '?'):<8} {terse:<28} {(c.get('why') or '')[:160]}")
    h = o.get("history") or []
    if len(h) > 1:
        L.append("where you have been (room coordinates, one row per decision):")
        for r in h:
            L.append(f"    k={r['k']}  x {r['x']:+.2f}  y {r['y']:+.2f}  z {r['z']:.2f}  heading {r['heading_deg']:+.0f} deg")
    return "\n".join(L)


def to_cmd(o, ans, arm):
    """Model JSON -> the CLI's cmd.json (room-frame dx/dy rotated into forward/left, limits applied)."""
    v = ans.get("verdict", "override"); why = (ans.get("why") or "").strip() or "no reason given"
    if v == "stop":
        return {"k": o["k"], "stop": True, "verdict": "approve", "why": why}, None
    if v == "approve":
        if arm == "waypoints":
            return None, "waypoints arm: nothing is proposed, so approve is not available; give an override with a move"
        return {"k": o["k"], "verdict": "approve", "why": why}, None
    g = {k: float(ans.get(k) or 0.0) for k in ("dx", "dy", "dz", "yaw", "sigma")}
    clipped = [k for k in LIM if abs(g[k]) > LIM[k]]
    for k in clipped:
        g[k] = math.copysign(LIM[k], g[k])
    g["sigma"] = min(max(g["sigma"], 0.0), 1.0)
    psi = float(o["pose"][3])
    fwd = math.cos(psi) * g["dx"] + math.sin(psi) * g["dy"]
    lft = -math.sin(psi) * g["dx"] + math.cos(psi) * g["dy"]
    cmd = {"k": o["k"], "verdict": "override", "why": why,
           "move": {"forward": fwd, "left": lft, "up": g["dz"], "yaw_deg": g["yaw"], "sigma": g["sigma"]}}
    return cmd, (f"clipped {clipped} to the per-decision limits" if clipped else None)


def ask_gemini(model, system, text, images, temperature=0.2):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    parts = [types.Part.from_bytes(data=open(p, "rb").read(), mime_type="image/jpeg") for p in images] + [types.Part.from_text(text=text)]
    r = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                       config=types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                                          response_mime_type="application/json", response_schema=SCHEMA))
    return r.text


def ask_dry(model, system, text, images, temperature=0.2):
    k = int(text.split("k=")[1].split()[0])
    return json.dumps({"verdict": "approve", "why": "seen: dry run | rule: R0 | action: approve"} if k % 2 == 0 else
                      {"verdict": "override", "why": "seen: dry run | rule: R6 | action: 1 m along +x", "dx": 1.0, "sigma": 0.5})


PROVIDERS = {"gemini": ask_gemini, "dry": ask_dry}


def write(d, cmd):
    tmp = os.path.join(d, "cmd.json.tmp")
    json.dump(cmd, open(tmp, "w")); os.replace(tmp, os.path.join(d, "cmd.json"))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dir", required=True); a.add_argument("--brief", required=True)
    a.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    a.add_argument("--provider", default="gemini", choices=list(PROVIDERS)); a.add_argument("--dry", action="store_true")
    a.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for a decision before giving up")
    a.add_argument("--temperature", type=float, default=0.2)
    a.add_argument("--fallback-model", default=os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash"),
                   help="used from the fourth attempt on when the primary model keeps returning 503")
    a.add_argument("--attempts", type=int, default=8)
    a.add_argument("--min-gap", type=float, default=float(os.environ.get("GEMINI_MIN_GAP_S", "6")),
                   help="minimum seconds between calls (free-tier requests-per-minute)")
    g = a.parse_args(); d = g.dir; ask = PROVIDERS["dry" if g.dry else g.provider]
    if not g.dry and g.provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("[api-driver] GEMINI_API_KEY is not set (put it in ~/.config/gemini.env; run_agent_matrix.sh exports it)")
    system = open(g.brief).read()
    armf = os.path.join(d, "arm"); arm = open(armf).read().strip() if os.path.exists(armf) else "ours"
    logf = open(os.path.join(d, "api_log.jsonl"), "a")
    print(f"[api-driver] dir {d} model {g.model} arm {arm} provider {'dry' if g.dry else g.provider}", flush=True)
    t_wait = time.time(); n = 0; t_last_call = 0.0
    while True:
        if os.path.exists(os.path.join(d, "done")):
            print(f"[api-driver] flight ended after {n} decisions", flush=True); return
        o = latest(d)
        ready = o and o.get("status") == "waiting" and not os.path.exists(os.path.join(d, "cmd.json")) \
            and not os.path.exists(os.path.join(d, "cmds", f"{o['k']:03d}.json"))
        if not ready:
            if time.time() - t_wait > g.timeout:
                sys.exit("[api-driver] timed out waiting for the flight")
            time.sleep(0.3); continue
        text = readout(o, d); view = os.path.join(d, o["view"]) if not os.path.isabs(o["view"]) else o["view"]
        images = [CALIB, view]
        note = ""; raw = ""; cmd = None
        for attempt in range(g.attempts):
            gap = g.min_gap - (time.time() - t_last_call)
            if gap > 0 and not g.dry:
                time.sleep(gap)
            t0 = time.time(); t_last_call = t0; model = g.model if attempt < 3 or g.dry else g.fallback_model
            try:
                raw = ask(model, system, text + ("\n\nNOTE: " + note if note else ""), images, g.temperature)
                ans = json.loads(raw)
                cmd, warn = to_cmd(o, ans, arm)
                dt = time.time() - t0
                if cmd is None:
                    note = warn; print(f"[api-driver] k={o['k']} retry: {warn}", flush=True); continue
                if warn:
                    print(f"[api-driver] k={o['k']} {warn}", flush=True)
                if model != g.model:
                    print(f"[api-driver] k={o['k']} answered by fallback model {model}", flush=True)
                break
            except Exception as e:   # API or parse error: say so, back off, retry
                dt = time.time() - t0; note = f"your previous reply was not valid JSON of the required form ({type(e).__name__})"
                print(f"[api-driver] k={o['k']} attempt {attempt} ({model}): {type(e).__name__}: {str(e)[:160]}", flush=True)
                time.sleep(min(3 * 2 ** attempt, 40))
        if cmd is None:
            cmd = {"k": o["k"], "verdict": "override", "why": "seen: no valid reply from the model | rule: R8 | action: hold",
                   "move": {"forward": 0.0, "left": 0.0, "up": 0.0, "yaw_deg": 0.0, "sigma": 0.0}}
        write(d, cmd); n += 1; t_wait = time.time()
        mv = cmd.get("move") or {}
        print(f"[api-driver] k={o['k']} {cmd['verdict']} ({dt:.1f}s) {' '.join(f'{k} {v:+.2f}' for k, v in mv.items() if v)} | {cmd['why'][:150]}", flush=True)
        logf.write(json.dumps({"k": o["k"], "t": time.time(), "latency_s": round(dt, 2), "readout": text, "raw": raw, "cmd": cmd}) + "\n"); logf.flush()


if __name__ == "__main__":
    main()
