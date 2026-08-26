"""Progress-conditioned pin prior: c = MLP([model_state, onehot, progress]) where progress = t/T in [0,1].
Hypothesis: progress is the phase variable that both disambiguates the out/back overlap AND signals the
turn timing, so the pin commands the far-side turn instead of overshooting. PCA basis (matches current flow).
"""
import os, glob, json, numpy as np, torch, torch.nn as nn
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pin_basis
RD=os.path.expanduser("~/code/source-noise-mvp/experiments/rung3"); DD=f"{RD}/data_gate_synth"
HFB="/home/ubuntu/hf_bundle/gate-drone-pi0"
UPATH=os.environ.get("UPATH",f"{RD}/pin_U_gate_k5.npy")
U=np.load(UPATH).astype(np.float32); H,AD=50,32
K=U.shape[1]   # pin dimension follows the basis (was hardcoded 5 until 2026-08-09)
OUT=os.environ.get("OUT","/tmp/prog_prior_gate.pt"); AUG=float(os.environ.get("AUG","0.1"))
DATA=os.environ.get("DATA","synth")        # synth | real | synth+real
INIT=os.environ.get("INIT","")             # optional warm-start checkpoint (fine-tune)
NOPROG=os.environ.get("NOPROG","0")=="1"   # ablation: drop the progress input entirely
GATE=np.array([0.86,0.69,1.5]);NRM=np.array([0.7488,0.6628,0.]);NRM/=np.linalg.norm(NRM);APER=0.45
LEFT="go through the gate on the left and hover over the stuffed animal"; RIGHT=LEFT.replace("left","right")
CFL="go through the center gate from the left and hover over the stuffed animal"
CFR="go through the center gate from the right and hover over the stuffed animal"
# authoritative task by episode index (data_gate_synth task map == lerobot meta.json):
# ep0000-0049 center-from-left, ep0050-0099 center-from-right, ep0100-0149 LEFT, ep0150-0199 RIGHT
def task_of(ep_index):
    return (CFL, CFR, LEFT, RIGHT)[ep_index // 50]
def is_left(st):
    P=st[:,:3];s=(P-GATE)@NRM;cr=np.where(np.sign(s[:-1])!=np.sign(s[1:]))[0]
    for i in cr:
        t=s[i]/(s[i]-s[i+1]+1e-9);xp=P[i]+t*(P[i+1]-P[i]);d=xp-GATE
        if np.linalg.norm(d-(d@NRM)*NRM)<=APER:return True
    return False
import openpi.shared.normalize as NZ
from openpi import transforms as T
from openpi.transforms import NormStats
import openpi.training.config as C, openpi.policies.policy_config as PC
ns=NZ.load(f"{HFB}/assets/gate_nav")
def pads(nsd,dim):
    out={}
    for k,s in nsd.items():
        n=np.asarray(s.mean).shape[-1]
        if n>=dim: out[k]=s;continue
        p=dim-n; ext=lambda a,f: None if a is None else np.concatenate([np.asarray(a,np.float32),np.full(p,f,np.float32)])
        out[k]=NormStats(mean=ext(s.mean,0),std=ext(s.std,1),q01=ext(s.q01,0),q99=ext(s.q99,1))
    return out
cfg=C.get_config("pi0_gate"); nsp=pads(ns,cfg.model.action_dim)
policy=PC.create_trained_policy(cfg,f"{HFB}/checkpoints/gate_both_pin",norm_stats=nsp)
nrm=T.Normalize(nsp,use_quantiles=False)
_D=np.zeros((224,224,3),np.uint8)
ZEROPAD=os.environ.get("SNMVP_ZERO_PAD_ACTIONS")=="1"   # match the data pipeline
def c_of(chunk7):
    L=len(chunk7); ch=np.zeros((H,AD),np.float32); m=min(L,H); ch[:m,:7]=chunk7[:m]
    # repeating the last DELTA injects motion that never happens (8.5x inflation ten steps
    # from an episode end, 2026-08-11); zero = "stay put", which is what the demo does
    if m<H and not ZEROPAD: ch[m:,:7]=chunk7[m-1]
    return (nrm({"actions":ch})["actions"].reshape(-1))@U
TASKS4=[CFL,CFR,LEFT,RIGHT]
def onehot(l):
    v=np.zeros(4,np.float32); v[TASKS4.index(l)]=1.; return v
RD_REAL=f"{RD}/data_gate_real"
_real_meta=json.load(open(f"{RD_REAL}/meta.json")) if DATA!="synth" else {}
files=[]
if DATA in ("synth","synth+real"): files+= [("synth",f) for f in sorted(glob.glob(f"{DD}/ep_*.npz"))]
REAL_TASK=os.environ.get("REAL_TASK","")   # optional: restrict real files to prompts containing this word
if DATA in ("real","synth+real"):
    files+= [("real", f"{RD_REAL}/{k}.npz") for k in sorted(_real_meta)
             if not REAL_TASK or REAL_TASK in _real_meta[k]["lang"]]
rng=np.random.default_rng(0); perm=rng.permutation(len(files)); held=set(perm[int(0.8*len(files)):].tolist())  # FROZEN split (perm[:160] train) — matches every other consumer; the old perm[:20] leaked all frozen-test eps into training (2026-08-05 review)
# TAILW>1 upweights goal-phase rows (frac>=0.75) in the loss — the 2026-08-08 diagnosis
# showed prior error jumps 3-5x in the tail while the flow can execute it (oracle 3/5 full)
TAILW=float(os.environ.get("TAILW","1"))
Xtr=[];Ytr=[];Xte=[];Yte=[]; Wtr=[]; lens=[]
for ei,(src_kind,f) in enumerate(files):
    d=np.load(f,allow_pickle=True); st=d["state"].astype(np.float32); ac=d["action"].astype(np.float32)
    lang=_real_meta[os.path.basename(f)[:-4]]["lang"] if src_kind=="real" else task_of(ei); Tn=len(st); lens.append(Tn); oh=onehot(lang)
    for t in range(0,(Tn-5 if ZEROPAD else Tn-H),4):   # ZEROPAD also covers the terminal phase (rows used to stop at progress 0.783)
        ms=np.asarray(policy._input_transform({"observation/image":_D,"observation/wrist_image":_D,"observation/state":st[t],"prompt":lang})["state"]).reshape(-1)
        prog=np.float32(t/(Tn-1))
        x=(np.concatenate([ms,oh]) if NOPROG else np.concatenate([ms,oh,[prog]])).astype(np.float32); y=c_of(ac[t:t+H]).astype(np.float32)
        if ei in held: Xte.append(x); Yte.append(y)
        else: Xtr.append(x); Ytr.append(y); Wtr.append(np.float32(TAILW if prog>=0.75 else 1.0))
Xtr=np.array(Xtr,np.float32);Ytr=np.array(Ytr,np.float32);Xte=np.array(Xte,np.float32);Yte=np.array(Yte,np.float32)
Wtr=np.array(Wtr,np.float32)
EXP_LEN=int(np.mean(lens))
mu,sdv=Xtr.mean(0),Xtr.std(0)+1e-6; ntask=4
def build(indim): return nn.Sequential(nn.Linear(indim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,K))
net=build(Xtr.shape[1])
if INIT:
    _d=torch.load(INIT,map_location="cpu",weights_only=False); net.load_state_dict(_d["state_dict"])
    print("warm-started from",INIT,flush=True)
opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
Xt=torch.tensor((Xtr-mu)/sdv); Yt=torch.tensor(Ytr); Wt=torch.tensor(Wtr)
for ep in range(400):
    p=torch.randperm(len(Xt))
    for i in range(0,len(Xt),1024):
        j=p[i:i+1024]; xb=Xt[j].clone()
        if AUG>0:  # perturb state (+progress when present), never the onehot block
            nstate=xb.shape[1]-ntask-(0 if NOPROG else 1)
            xb[:, :nstate]+=AUG*torch.randn_like(xb[:, :nstate])
            if not NOPROG: xb[:, -1:]+=AUG*torch.randn_like(xb[:, -1:])
        opt.zero_grad(); loss=(Wt[j]*((net(xb)-Yt[j])**2).mean(1)).sum()/Wt[j].sum(); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    pr=net(torch.tensor((Xte-mu)/sdv)).numpy(); prtr=net(Xt).numpy()
r2=1-((Yte-pr)**2).sum()/(((Yte-Yte.mean(0))**2).sum()+1e-9)
r2tr=1-((Ytr-prtr)**2).sum()/(((Ytr-Ytr.mean(0))**2).sum()+1e-9)
print("progress-prior: train c-R2 %.4f held %.4f  in_dim=%d EXP_LEN=%d AUG=%.2f"%(r2tr,r2,Xtr.shape[1],EXP_LEN,AUG),flush=True)
torch.save({"kind":"progress_prior","in_dim":Xtr.shape[1],"hidden":[256,256],"tasks":[CFL,CFR,LEFT,RIGHT],
            "H":H,"AD":AD,"K":K,"state_dim":Xtr.shape[1]-5,"exp_len":EXP_LEN,"mu":mu.astype(np.float32),
            "sd":sdv.astype(np.float32),"state_dict":net.state_dict(),
            **pin_basis.stamp(UPATH)},OUT)
print("SAVED",OUT,"PROG_PRIOR_DONE",flush=True)
