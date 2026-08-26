"""UNION build: thin-cache rows (12k, on-route accuracy, 4 tasks) + fat-tube rows
(2.4k, basin) — the fat-only heads lost on-route precision (0/8 closed-loop wander,
2026-08-06); coverage must ADD to accuracy, not replace it. CPU."""
import glob, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ctx_common as gc
import gate_traj_algebra as ta
from fat_tube_gen import sample_rows, RETURN_STEPS
RUN=os.path.expanduser("~/ctxrun"); RD=gc.RD; STRIDE=12
ns,amean,astd=gc.load_norm(); U=np.load(os.path.join(RD,"pin_U_gate_rrr_k5.npy")); H=gc.H
policy=gc.make_policy()
_D=np.zeros((224,224,3),np.uint8)
def mstate(raw,lang):
    return np.asarray(policy._input_transform({"observation/image":_D,"observation/wrist_image":_D,
        "observation/state":raw,"prompt":lang})["state"]).reshape(-1)
# thin cache rows (identical alignment to train_vlmflow_head)
srcs=gc.load_eps(with_images=False)
rng=np.random.default_rng(0); idx=rng.permutation(len(srcs)); trep=set(idx[:160].tolist())
groups=[]
for si,e in enumerate(srcs):
    groups.append((si,e)); groups.append((si,ta.reverse(e)))
    for f in (ta.crop_to_gate,ta.crop_from_gate):
        a=f(e)
        if a is not None: groups.append((si,a))
    groups.append((si,ta.hover(e,len(e["action"])//2)))
Y1,MS1,HE1=[],[],[]
for si,e in groups:
    n=min(len(e["action"]),len(e["state"])-1)
    ac=e["action"].astype(np.float32); st=e["state"].astype(np.float32)
    for t in range(0,n,STRIDE):
        Y1.append((gc.segY(ac[t:],amean,astd)@U).astype(np.float32))
        MS1.append(mstate(st[t],e["lang"])); HE1.append(si not in trep)
X1=np.concatenate([np.load(f) for f in sorted(glob.glob(f"{RUN}/Xrendshard_*.npy"))],0)
Y1=np.stack(Y1); MS1=np.stack(MS1); HE1=np.array(HE1)
assert len(X1)==len(Y1)
# fat tube rows
srcs2,rows2=sample_rows()
phi2=np.load(f"{RUN}/fat_tube_phi.npy")
rf=np.load(f"{RUN}/fat_tube_frames.npz"); FWD,WR,ST=rf["fwd"],rf["wr"],rf["st"]
Y2,MS2=[],[]
for i,(task,ei,t,dv) in enumerate(rows2):
    e=srcs2[ei]
    chunk=e["action"][t:t+H].astype(np.float32).copy()
    if chunk.shape[0]<H: chunk=np.concatenate([chunk,np.zeros((H-len(chunk),chunk.shape[1]),np.float32)])
    chunk[:RETURN_STEPS,:3]-=dv.astype(np.float32)/RETURN_STEPS
    Y2.append((gc.segY(chunk,amean,astd)@U).astype(np.float32))
    MS2.append(np.asarray(policy._input_transform({"observation/image":FWD[i],
        "observation/wrist_image":WR[i],"observation/state":ST[i],"prompt":task})["state"]).reshape(-1))
Y2=np.stack(Y2); MS2=np.stack(MS2); HE2=np.zeros(len(Y2),bool)  # tube rows all train
X=np.concatenate([X1,phi2],0); Y=np.concatenate([Y1,Y2],0)
MS=np.concatenate([MS1,MS2],0); HE=np.concatenate([HE1,HE2],0)
print(f"union rows {len(X)} (thin {len(X1)} + tube {len(Y2)}), held {HE.sum()}",flush=True)
torch.manual_seed(0)
class VNet(nn.Module):
    def __init__(s,xdim,cdim=5,w=512):
        super().__init__(); s.net=nn.Sequential(nn.Linear(xdim+cdim+1,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,cdim))
    def forward(s,ct,t,x): return s.net(torch.cat([ct,t,x],1))
def train_head(Xa,name):
    tr=~HE
    xmu,xsd=Xa[tr].mean(0),Xa[tr].std(0)+1e-6; ymu,ysd=Y[tr].mean(0),Y[tr].std(0)+1e-6
    Xn=torch.tensor((Xa-xmu)/xsd,dtype=torch.float32); Yn=torch.tensor((Y-ymu)/ysd)
    net=VNet(Xa.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-5)
    tri=np.where(tr)[0]; rng=np.random.default_rng(0)
    for ep in range(80):
        perm=rng.permutation(tri)
        for j in range(0,len(perm),512):
            b=perm[j:j+512]; c1=Yn[b]; c0=torch.randn_like(c1); t=torch.rand(len(b),1)
            loss=((net((1-t)*c0+t*c1,t,Xn[b])-(c1-c0))**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        n=int(HE.sum()); xr=Xn[HE].repeat_interleave(8,0); c=torch.randn(n*8,5)
        for s_ in range(10):
            t=torch.full((n*8,1),s_/10); c=c+net(c,t,xr)/10
        P=(c.reshape(n,8,5).mean(1)*torch.tensor(ysd)+torch.tensor(ymu)).numpy()
    r2=1-((Y[HE]-P)**2).sum()/((Y[HE]-Y[HE].mean(0))**2).sum()
    print(f"{name}: held(thin-route) R^2 {r2:.3f}",flush=True)
    torch.save({"state_dict":net.state_dict(),"xmu":xmu.astype(np.float32),"xsd":xsd.astype(np.float32),
        "ymu":ymu.astype(np.float32),"ysd":ysd.astype(np.float32),"in_dim":Xa.shape[1],
        "H":H,"AD":gc.AD,"K":5,"arch":f"cfm-3x512-{name}"},os.path.join(RD,f"{name}.pt"))
train_head(X.astype(np.float32),"vlmflow_head_union")
train_head(np.concatenate([MS,X],1).astype(np.float32),"hybrid_head_union")
print("UNION_BUILD_DONE",flush=True)
