"""Text-embedding language prior + generalization test. Replaces the 2-dim task one-hot with a frozen
sentence-encoder (all-MiniLM-L6-v2) embedding of the instruction, so the pin can follow language MEANING
rather than a memorized label. Real data has only two instructions (left/right gate), so the test is not
in-domain R^2 (a one-hot already separates two classes) but GENERALIZATION to unseen paraphrases: train
the prior MLP([state, emb]) on the two ORIGINAL strings, then check whether held-out paraphrases steer c
the same way. Since c ~ f(phase, gate) and phase (state) dominates, we isolate the LANGUAGE effect as the
steering vector c_pred(state, right) - c_pred(state, left) and ask whether paraphrases reproduce it."""
import json
import os

import numpy as np

RD = os.path.expanduser("~/code/source-noise-mvp/experiments/rung3")
H, AD = 50, 32
import openpi.shared.normalize as NZ
ns = NZ.load(os.path.expanduser("~/code/openpi/assets/pi0_gate/local/gate_nav"))
amean, astd = np.asarray(ns["actions"].mean), np.asarray(ns["actions"].std)
smean, sstd = np.asarray(ns["state"].mean), np.asarray(ns["state"].std)
U = np.load(os.path.join(RD, "pin_U_gate_k5.npy")).astype(np.float32)

LEFT = "go through the gate on the left and hover over the stuffed animal"
RIGHT = "go through the gate on the right and hover over the stuffed animal"
# paraphrase pools; first TR_N per gate are used for training augmentation, the rest are HELD OUT
TR_N = 6
PARA = {
    LEFT: [LEFT,
           "fly through the leftmost gate then hover above the plush toy",
           "take the gate on your left and stop over the teddy bear",
           "navigate through the left-side gate to the stuffed animal",
           "pass the left gate and hold position above the toy",
           "head through the gate to the left, then hover by the stuffed toy",
           "cross the left gate and settle above the plushie",          # held out
           "enter the left-hand gate and hover over the animal"],       # held out
    RIGHT: [RIGHT,
            "fly through the rightmost gate then hover above the plush toy",
            "take the gate on your right and stop over the teddy bear",
            "navigate through the right-side gate to the stuffed animal",
            "pass the right gate and hold position above the toy",
            "head through the gate to the right, then hover by the stuffed toy",
            "cross the right gate and settle above the plushie",         # held out
            "enter the right-hand gate and hover over the animal"],      # held out
}


def seg_to_c(seg):
    m, r = seg.shape
    seg = seg[:H] if m >= H else np.concatenate([seg, np.zeros((H - m, r), np.float32)], 0)
    ch = np.zeros((H, AD), np.float32); ch[:, :r] = (seg - amean[:r]) / (astd[:r] + 1e-6)
    return ch.reshape(-1) @ U


def r2(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def main():
    import torch, torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name); enc = AutoModel.from_pretrained(name).eval()

    @torch.no_grad()
    def embed(texts):
        b = tok(texts, padding=True, truncation=True, return_tensors="pt")
        out = enc(**b).last_hidden_state
        m = b["attention_mask"][..., None].float()
        v = (out * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=1).numpy().astype(np.float32)

    PEMB = {g: embed(PARA[g]) for g in (LEFT, RIGHT)}         # [8, 384] per gate
    trE = {g: PEMB[g][:TR_N] for g in (LEFT, RIGHT)}          # train paraphrase embeddings
    teE = {g: PEMB[g][TR_N:] for g in (LEFT, RIGHT)}          # HELD-OUT paraphrase embeddings
    print(f"emb dim = {PEMB[LEFT].shape[1]}  cos(orig L, orig R) = {PEMB[LEFT][0] @ PEMB[RIGHT][0]:.3f}")
    print(f"held-out paraphrases/gate = {PEMB[LEFT].shape[0]-TR_N}, train paraphrases/gate = {TR_N}")

    meta = json.load(open(os.path.join(RD, "data_gate_real", "meta.json")))
    S, C, G, EP = [], [], [], []
    for k in sorted(meta):
        lang = meta[k]["lang"]
        if lang not in (LEFT, RIGHT):
            continue
        d = np.load(os.path.join(RD, "data_gate_real", k + ".npz"))
        acts = d["action"].astype(np.float32); states = d["state"].astype(np.float32); T = len(acts)
        for t in range(0, T, 3):
            S.append((states[t] - smean) / (sstd + 1e-6)); C.append(seg_to_c(acts[t:]))
            G.append(0 if lang == LEFT else 1); EP.append(k)
    S, C, G = np.asarray(S, np.float32), np.asarray(C, np.float32), np.asarray(G)
    eps = sorted(set(EP)); rng = np.random.default_rng(0); pe = rng.permutation(len(eps))
    trep = set(eps[i] for i in pe[:int(0.7 * len(eps))])
    tr = np.array([e in trep for e in EP]); te = ~tr
    Str, Ctr, Gtr = S[tr], C[tr], G[tr]
    Ste, Cte, Gte = S[te], C[te], G[te]
    print(f"chunks tr={tr.sum()} te={te.sum()} state_dim={S.shape[1]}")

    glang = [LEFT, RIGHT]
    # AUGMENT: each train chunk replicated with each of the TR_N train paraphrase embeddings of its gate
    Saug = np.repeat(Str, TR_N, 0)
    Eaug = np.stack([trE[glang[g]][j] for g in Gtr for j in range(TR_N)]).astype(np.float32)
    Caug = np.repeat(Ctr, TR_N, 0)
    Xm = np.concatenate([Saug, Eaug], 1).mean(0); Xs = np.concatenate([Saug, Eaug], 1).std(0) + 1e-6

    net = nn.Sequential(nn.Linear(S.shape[1] + Eaug.shape[1], 256), nn.SiLU(), nn.Dropout(0.1),
                        nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, U.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)

    def feat(s, e):
        return ((np.concatenate([s, e], 1) - Xm) / Xs).astype(np.float32)

    xt = torch.tensor(feat(Saug, Eaug)); yt = torch.tensor(Caug)
    for _ in range(5000):
        b = torch.randint(0, len(xt), (256,))
        loss = ((net(xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def pred(s, e):
        with torch.no_grad():
            return net(torch.tensor(feat(s, e))).numpy()

    # (a) held R^2 with the ORIGINAL (trained) strings
    Eorig = np.stack([trE[glang[g]][0] for g in Gte])
    print(f"\n(a) trained-string prior held R^2      = {r2(pred(Ste, Eorig), Cte):.3f}")
    # (b) GENERALIZATION: each chunk gets a HELD-OUT unseen paraphrase of its true gate
    rng2 = np.random.default_rng(1)
    Ehold = np.stack([teE[glang[g]][rng2.integers(teE[glang[0]].shape[0])] for g in Gte])
    print(f"(b) UNSEEN-paraphrase held R^2         = {r2(pred(Ste, Ehold), Cte):.3f}")
    # (c) steering fidelity on UNSEEN paraphrases: c_pred(right)-c_pred(left) at same states, trained vs unseen
    def steer(eL, eR):
        return (pred(Ste, np.tile(eR, (len(Ste), 1))) - pred(Ste, np.tile(eL, (len(Ste), 1)))).reshape(-1)
    so = steer(trE[LEFT][0], trE[RIGHT][0])
    coss, mags = [], []
    for i in range(teE[LEFT].shape[0]):
        sp = steer(teE[LEFT][i], teE[RIGHT][i])
        coss.append(float(so @ sp / (np.linalg.norm(so) * np.linalg.norm(sp) + 1e-9)))
        mags.append(float(np.linalg.norm(sp) / (np.linalg.norm(so) + 1e-9)))
    print(f"(c) UNSEEN-paraphrase steering vs trained: cos = {np.round(coss,3)}  mag ratio = {np.round(mags,3)}")
    print(f"    mean cos = {np.mean(coss):.3f}  mean mag ratio = {np.mean(mags):.3f}  (cos~1 => same steering)")
    print("TEXTEMB_DONE")


if __name__ == "__main__":
    main()
