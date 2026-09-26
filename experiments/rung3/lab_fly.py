"""Fly one Sketch Lab sketch (2026-09-25): writes the sketch files, flies every requested method (ours once per sigma,
velocity projection, SDEdit t0 0.5, pi0 + injected sketch) at a 25-step replan on port 9020, scores them, and writes
~/ctxrun/lab_<slug>_results.json keyed like the Sketch Lab's results field ("ours@0.3", "vproj", ...).

  ~/code/openpi/.venv/bin/python lab_fly.py spec.json      spec = what Sketch Lab saves or copies:
      {"name", "scene", "sigmas", "arms", "trials", "points": [[x, y], ...]}
A route is completed when the flight passes each gate in order in the correct direction with no wrong-way pass and
reaches the end of the sketch without touching a gate (route_contact).
"""
import json, math, os, re, subprocess, sys, time

import numpy as np

RD = os.path.dirname(os.path.abspath(__file__))
RUN = "/home/dfliu/ctxrun"
VENVPY = "/home/dfliu/code/openpi/.venv/bin/python"; TV = "/home/dfliu/code/tv/bin/python"
HFB = "/home/dfliu/hf_bundle/gate-drone-pi0"; U = f"{RD}/pin_U_mh16.npy"; PORT = 9020
CK = "/home/dfliu/code/openpi-snmvp/checkpoints/pi0_gate3"
EV = "env -u VIRTUAL_ENV PYTHONPATH=/home/dfliu/code/openpi-snmvp/src"
GPU = "XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0"
PIN = (f"SNMVP_HEAD=1 SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U={U} SNMVP_HEAD_DETACH=0 SNMVP_HEAD_LAM=0.3 SNMVP_HEAD_GMM=1 "
       f"SNMVP_PIN_NOISE=1.5 SNMVP_PIN_NOISE_RAND=1 SNMVP_PIN_NOISE_COND=1")
# spec["pin"] picks our checkpoint: "realonly" (the paper's policy, default) or "gmsig3" (mixed real + simulated demos)
PINS = {"realonly": "gate_pin_joint_realonly", "gmsig3": "gate_pin_joint_gmsig3"}
SIDE = {"left": "left", "right": "right", "left_and_center": "left", "right_and_center": "right"}
PROMPT = {"left": "go through the gate on the left and hover over the stuffed animal",
          "right": "go through the gate on the right and hover over the stuffed animal",
          "left_and_center": "go through the gate on the left, then through the center gate and hover over the stuffed animal",
          "right_and_center": "go through the gate on the right, then through the center gate and hover over the stuffed animal"}


def sketch_file(spec, slug, sigma):
    P = spec["points"]; n = len(P); pts = []
    for i, p in enumerate(P):
        q = P[min(i + 1, n - 1)]; o = (q[0] - p[0], q[1] - p[1]) if i < n - 1 else (p[0] - P[i - 1][0], p[1] - P[i - 1][1])
        pts.append([round(p[0], 3), round(p[1], 3), 1.5 if i == 0 else 1.2 if i == n - 1 else 1.45, round(math.atan2(o[1], o[0]), 3)])
    d = {"points": pts, "prompt_after": PROMPT[spec["scene"]], "enter_radius": 0.5, "step_m": 0.025, "sigma_serve": sigma,
         "end_margin_m": 0.1, "carrot": 20, "note": f"Sketch Lab: {spec['name']}"}
    f = f"{RD}/sketch_lab_{slug}_s{sigma:g}.json"; json.dump(d, open(f, "w"), indent=1); return f


def kill_port():
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT} " in line:
            for pid in re.findall(r"pid=(\d+)", line):
                subprocess.run(["kill", "-9", pid])
    time.sleep(3)


def serve(cmd, log):
    kill_port()
    subprocess.Popen(["bash", "-c", f"setsid {cmd} >> {log} 2>&1 < /dev/null &"])
    for _ in range(200):
        if f":{PORT} " in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
            return True
        time.sleep(3)
    return False


def fly(arm, sk, tag, spec, seed=11):
    log = f"{RUN}/sv_lab.log"
    if arm == "ours":
        pin = spec.get("pin", "realonly")
        cmd = f"{EV} {PIN} SNMVP_SIGMA_MAP={RD}/sigma_map_{pin}.json SNMVP_NOISE_SEED={seed} SNMVP_PIN_PROMPT={sk} {GPU} {VENVPY} {RD}/serve_gate_pin_joint.py --ckpt {CK}/{PINS[pin]}/4999 --config pi0_gate --norm {HFB}/assets/gate_nav --pin-u {U} --port {PORT}"
    elif arm == "vproj":
        cmd = f"{EV} SNMVP_NOISE_SEED={seed} SNMVP_ZERO_PAD_ACTIONS=1 SNMVP_PIN_U={U} SNMVP_VPROJ=1 {GPU} {VENVPY} {RD}/serve_gate_plain_sketch.py --ckpt {CK}/gate_scratch_real/4999 --config pi0_gate --norm {HFB}/assets/gate_nav --pin-u {U} --sketch {sk} --port {PORT}"
    elif arm == "inject":
        cmd = f"{EV} SNMVP_NOISE_SEED={seed} SNMVP_ZERO_PAD_ACTIONS=1 {GPU} {VENVPY} {RD}/serve_gate_plain_sketch.py --ckpt {CK}/gate_scratch_real/4999 --config pi0_gate --norm {HFB}/assets/gate_nav --pin-u {U} --sketch {sk} --port {PORT}"
    else:
        cmd = f"{EV} SNMVP_NOISE_SEED={seed} SNMVP_ZERO_PAD_ACTIONS=1 {GPU} {VENVPY} {RD}/serve_gate_sdedit.py --ckpt {CK}/gate_scratch_real/4999 --config pi0_gate --norm {HFB}/assets/gate_nav --sketch {sk} --t0 0.5 --port {PORT}"
    if not serve(cmd, log):
        print("SERVER_TIMEOUT", tag); return False
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PORT=str(PORT), SIDE=SIDE[spec["scene"]], SCENE=spec["scene"],
               NCH=str(spec.get("nch", 16)), APC="25", TRIALS=str(spec["trials"]), VIDEO="0", PROMPT=PROMPT[spec["scene"]],
               TRAJ=f"{RUN}/traj_{tag}_{{t}}.npy")
    with open(f"{RUN}/roll_{tag}.log", "w") as lf:
        subprocess.run([TV, f"{RD}/gate_rollout_batch.py"], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=RD)
    kill_port(); return True


def score(spec, sk, tag):
    import glob, torch
    sys.path.insert(0, RD)
    import gate_clearance as G, gate_success as GS, route_contact as RC
    S = np.asarray(json.load(open(sk))["points"], np.float64)[:, :3]; Q, _ = RC.dense(S)
    cloud = torch.tensor(np.asarray(G.gate_cloud(spec["scene"]), np.float32))
    ok, dmins, trk = 0, [], []
    fs = sorted(glob.glob(f"{RUN}/traj_{tag}_[0-9]*.npy"))
    for f in fs:
        P = np.load(f)[:, :3].astype(np.float64)
        if spec["scene"] in ("left", "right"):
            j = GS.judge(P, spec["scene"]); route = j["transit"] and j["wrong_dir_crossings"] == 0
        else:
            j = GS.judge_compound(P, spec["scene"]); route = j["gates_latched"] == 2 and sum(j["wrong_crossings"]) == 0
        e, dmin = RC.contact(P, S, cloud)
        ok += route and e < len(P) - 1 and dmin >= RC.BODY
        dmins.append(dmin); trk.append(np.linalg.norm(P[:e + 1, None] - Q[None], axis=2).min(1).mean())
    return {"completed": int(ok), "n": len(fs), "closest_m": round(float(np.median(dmins)), 3) if fs else None,
            "tracking_cm": round(float(np.median(trk)) * 100, 1) if fs else None}


def main():
    spec = json.load(open(sys.argv[1]))
    slug = re.sub(r"[^a-z0-9]+", "-", spec["name"].lower()).strip("-")[:40] + ("" if spec.get("pin", "realonly") == "realonly" else "-" + spec["pin"])
    base = sketch_file(spec, slug, 0.0)
    cells = [(f"ours@{s:g}", "ours", sketch_file(spec, slug, float(s))) for s in spec.get("sigmas", [0])] if "ours" in spec["arms"] else []
    cells += [(a, a, base) for a in ("vproj", "sdedit", "inject") if a in spec["arms"]]
    results = {}
    for key, arm, sk in cells:
        tag = f"lab_{slug}_{key.replace('@', '_s')}"
        if fly(arm, sk, tag, spec):
            results[key] = score(spec, base, tag)
            print(key, results[key], flush=True)
            json.dump({"name": spec["name"], "results": results}, open(f"{RUN}/lab_{slug}_results.json", "w"), indent=1)
    print("LAB_DONE", slug)


if __name__ == "__main__":
    main()
