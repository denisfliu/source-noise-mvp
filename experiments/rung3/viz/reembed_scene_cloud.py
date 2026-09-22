"""Re-embed the current scene clouds into already-built viewer pages without rerunning their builders
(2026-09-22, after the goal table was given full density in extract_scene_cloud.py). Each page holds one
or more `const D = {...};` payloads whose cloud is stored relative to a centre; the scene is identified by
matching that centre to the mean of the OLD cloud (--old-dir holds the previous scene_cloud_*.npz), the
cloud is replaced by the new one re-centred on the page's existing centre, and trajectories are untouched.

  python reembed_scene_cloud.py --old-dir <dir with old npz> [--dry] page.html ...
"""
import argparse, base64, json, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
SCENES = ["left", "left_and_center", "center", "right", "right_and_center", "vocab"]

def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--old-dir", required=True); ap.add_argument("--dry", action="store_true")
    ap.add_argument("pages", nargs="+"); a = ap.parse_args()
    old = {}; new = {}
    for sc in SCENES:
        f = os.path.join(a.old_dir, f"{sc}.npz")
        if os.path.exists(f): old[sc] = np.load(f)["pts"].astype(np.float32).mean(0)
        g = os.path.join(SP, f"scene_cloud_{sc}.npz")
        if os.path.exists(g): z = np.load(g); new[sc] = (z["pts"].astype(np.float32), z["rgb"].astype(np.uint8))
    for page in a.pages:
        s = open(page).read(); out = []; pos = 0; n_done = 0; report = []
        while True:
            i = s.find("const D = {", pos)
            if i < 0: out.append(s[pos:]); break
            j = s.find("};\nconst cv", i)
            if j < 0: out.append(s[pos:]); break
            D = json.loads(s[i + len("const D = "):j + 1])
            if "centre" not in D:   # a different viewer's payload (gridviewer): leave it
                out.append(s[pos:j + 1]); pos = j + 1; n_done += 1; continue
            c = np.array(D["centre"], np.float32)
            # subsampled clouds share the full cloud's mean to within a few cm; the scenes differ by metres
            sc, dist = min(((k, float(np.linalg.norm(old[k] - c))) for k in old), key=lambda t: t[1])
            if dist > 0.25 or sc not in new:
                report.append(f"payload {n_done}: no scene match (nearest {sc} at {dist:.2f} m), left as is"); out.append(s[pos:j + 1]); pos = j + 1; n_done += 1; continue
            pts, rgb = new[sc]
            D["n"] = int(len(pts)); D["pts"] = b64((pts - c).astype(np.float32)); D["rgb"] = b64(rgb)
            out.append(s[pos:i]); out.append("const D = " + json.dumps(D)); pos = j + 1; n_done += 1
            report.append(f"payload {n_done - 1}: {sc} (centre match {dist:.2f} m) -> {len(pts)} pts")
        print(os.path.basename(page) + ": " + "; ".join(report))
        if not a.dry and n_done: open(page, "w").write("".join(out))

main()
