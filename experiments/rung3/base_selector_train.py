"""Train the selector on BASE-tower features; gate-b on the untouched eval set. CPU."""
import os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
RUN = os.path.expanduser("~/ctxrun")
TASKS = [gc.PROMPT_CFL, gc.PROMPT_CFR, gc.PROMPT_L, gc.PROMPT_R]
torch.manual_seed(0); np.random.seed(0)
Xtr = np.load(f"{RUN}/Xbase_train.npy"); ytr = np.load(f"{RUN}/ybase_train.npy")
Xev = np.load(f"{RUN}/Xbase_eval.npy"); yev = np.load(f"{RUN}/ybase_eval.npy")
mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-6
net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(approximate="tanh"), nn.Linear(128, 4))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
Xn = torch.tensor((Xtr - mu) / sg, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
for ep in range(60):
    perm = np.random.permutation(len(Xn))
    for i in range(0, len(perm), 256):
        b = perm[i:i + 256]; opt.zero_grad()
        nn.functional.cross_entropy(net(Xn[b]), yt[b]).backward(); opt.step()
net.eval()
with torch.no_grad():
    pred = net(torch.tensor((Xev - mu) / sg, dtype=torch.float32)).argmax(1).numpy()
ok = True
for k, t in enumerate(TASKS):
    m = yev == k
    acc = float((pred[m] == k).mean()); ok &= acc >= 0.90
    print("BASE GATE-b %-70s acc %.3f" % (t[:66], acc), flush=True)
print("BASE VERDICT (bar >=0.90/task): %s" % ("PASS" if ok else "FAIL"), flush=True)
L = [m for m in net if isinstance(m, nn.Linear)]
np.savez(os.path.join(gc.RD, "task_selector_base.npz"), mu=mu.astype(np.float32), sg=sg.astype(np.float32),
         W1=L[0].weight.detach().numpy().T, b1=L[0].bias.detach().numpy(),
         W2=L[1].weight.detach().numpy().T, b2=L[1].bias.detach().numpy(), tasks=np.array(TASKS))
print("BASE_GATEB_DONE", flush=True)
