"""Batch rollout client — gate_video_overlay.py restructured for throughput (Denis, 2026-08-12).

Timing on this box put ~75% of a rollout in video-frame rendering (25 extra 512x384 renders per
chunk, mp4-only) and ~15% in per-trial process startup (torch+gsplat import, full splat checkpoint
load). The policy consumes only TWO 224x224 renders per chunk. So:

  - TRIALS trials run inside ONE process; the scene loads once.
  - VIDEO=0 renders nothing beyond the policy's own observations and writes no mp4 — for 5-trial
    screens and traj-only batteries. Scores are unaffected (verdicts come from the trajectory).
    CLAIM-TIER RUNS KEEP VIDEO=1: human review is part of strict success.
  - VIDEO=1 renders every VIDFRAME_STRIDE-th step (default 4, was 2) at FPS (default 9, was 18):
    same full-length playback, half the frames — per Denis, lower review fps is fine.

The OBSERVATION path is byte-identical to gate_video_overlay.py and must stay that way: forward cam
(Tbc_f, native K) + DOWNWARD wrist cam (Tbc_d, downward K) each 1024x768 -> 256 -> 224, strut overlay
composited on the wrist view after the 256 squash. Any change there is a train/serve mismatch.

Env: PORT, OUT (template with {t}), TRAJ (template with {t}), TRIALS, SIDE, SCENE, NCH, APC, PROMPT,
START, STARTYAW, VIDEO, VIDFRAME_STRIDE, FPS, COMPOSE, COMPOSE_MID, COMPOSE_MID_CHUNKS.
"""
import torch,numpy as np,os
from gsplat import rasterization
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import imageio.v2 as iio
from PIL import Image, ImageDraw
NCH=int(os.environ.get("NCH","40")); PORT=int(os.environ["PORT"]); OUT=os.environ.get("OUT","")
SIDE=os.environ.get("SIDE","left"); SCENE=os.environ.get("SCENE","left"); TRAJ=os.environ.get("TRAJ",""); DEV="cuda"
TRIALS=int(os.environ.get("TRIALS","1"))
VIDEO=os.environ.get("VIDEO","1")=="1"
VSTRIDE=int(os.environ.get("VIDFRAME_STRIDE","4")); FPS=int(os.environ.get("FPS","9"))
if VIDEO and OUT and "{t}" not in OUT and TRIALS>1: raise SystemExit("OUT needs a {t} placeholder when TRIALS>1")
if TRAJ and "{t}" not in TRAJ and TRIALS>1: raise SystemExit("TRAJ needs a {t} placeholder when TRIALS>1")
D=np.diag([1.,-1,-1,1])
# right_and_center is DEFINED ON THE RIGHT SPLAT (its YAML gsplat_path): selecting the base
# checkpoint with `SCENE in ("right",)` silently rendered compound-right in the LEFT scene —
# no right gate at all — which invalidated every compound-right eval before 2026-08-12.
if SCENE in ("right", "right_and_center"):
    CK="/home/dfliu/code/falsify/data/gate_scenes_export/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt"
    M=np.array([[0.136708,-0.001053,0.006031,-0.111938],[0.00108,0.13684,-0.000588,0.030456],[-0.006027,0.000635,0.136711,-0.201447],[0,0,0,1.]]); Tw2g=M@D
    GAABB=(np.array([-0.06,-1.55,0.05]),np.array([1.15,-0.75,2.05])); GANCH=np.array([0.544,-1.147,0.074]); GNRM=np.array([0.385,-0.923,0.])
else:
    CK="/home/dfliu/code/falsify/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt"
    Tw2g=np.array([[0.12614431661544656,2.138646801849853e-06,-0.00025306576654559085,-0.15671883492487332],[-2.138646801849853e-06,-0.1261265572041315,-0.0021319289354524646,-0.08013551648879384],[-0.00025306576654559085,0.0021319289354524646,-0.12612630156484925,-0.18772133850562778],[0,0,0,1.]])
    GAABB=(np.array([0.36,0.12,0.05]),np.array([1.36,1.27,2.05])); GANCH=np.array([0.861,0.694,0.075]); GNRM=np.array([0.749,0.663,0.])
GNRM=GNRM/np.linalg.norm(GNRM)
if SCENE=="center":
    GAABB=(np.array([2.25,-0.69,0.05]),np.array([3.25,0.19,2.05]))
    GANCH=np.array([2.756,-0.3275,0.125]); GNRM=np.array([0.,-1.,0.])
sd=torch.load(CK,map_location=DEV,weights_only=False)["pipeline"]
def gg(n):
    for p in ("_model.gauss_params.","_model."):
        if p+n in sd: return sd[p+n].to(DEV)
means=gg("means");quats=gg("quats");scales=torch.exp(gg("scales"));opac=torch.sigmoid(gg("opacities")).squeeze(-1)
colors=torch.cat([gg("features_dc")[:,None,:],gg("features_rest")],1); bg=torch.tensor([0.149,0.1647,0.2157],device=DEV)
if SCENE=="center":
    import sys as _sys, os as _os
    _sys.path.insert(0,_os.path.expanduser("~/code/source-noise-mvp/experiments/rung3"))
    from gsplat_scene_edit import apply_move_gate as _amg
    means,quats,_nmoved=_amg(means,quats,Tw2g)
    print(f"[scene] center edit applied: {_nmoved} gaussians moved",flush=True)
elif SCENE in ("left_and_center","right_and_center"):
    import sys as _sys, os as _os
    _sys.path.insert(0,_os.path.expanduser("~/code/source-noise-mvp/experiments/rung3"))
    from gsplat_scene_edit import apply_duplicate_gate as _adg
    means,quats,scales,opac,colors,_ncopy=_adg(means,quats,scales,opac,colors,Tw2g,SCENE)
    print(f"[scene] {SCENE}: {_ncopy} gaussians duplicated to center",flush=True)
COMPOSE=os.environ.get("COMPOSE","")
COMPOSE_MID=os.environ.get("COMPOSE_MID","")
COMPOSE_MID_CHUNKS=int(os.environ.get("COMPOSE_MID_CHUNKS","2"))
Wv,Hv=512,384; sx,sy=Wv/1024.,Hv/768.
Kp=np.array([[502.2632*sx,0,506.3971*sx],[0,500.6736*sy,385.41*sy],[0,0,1.]])
Kv=torch.tensor(Kp,device=DEV,dtype=torch.float32)[None]
Kf=torch.tensor([[502.2632,0.,506.3971],[0.,500.6736,385.41],[0.,0.,1.]],device=DEV)[None].float()
Kd=torch.tensor([[478.2450,0.,511.9041],[0.,476.7944,383.5003],[0.,0.,1.]],device=DEV)[None].float()
Tbc_f=np.array([[0,0,-1,0.10],[1,0,0,-0.03],[0,-1,0,-0.01],[0,0,0,1.]])
Tbc_d=np.array([[0,1,0,0.0],[1,0,0,0.0],[0,0,-1,0.05],[0,0,0,1.]])
_ov=np.asarray(Image.open("/home/dfliu/code/falsify/configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png").convert("RGBA").resize((256,256),Image.BILINEAR),np.uint8)  # hard-fail if missing: a silent None here would strip the strut from the wrist view
def Rz(p):c,s=np.cos(p),np.sin(p);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def vm(pos,yaw,Tbc):
    pn=np.array([pos[0],-pos[1],-pos[2]]);T=np.eye(4);T[:3,:3]=Rz(yaw);T[:3,3]=pn
    c2w=Tw2g@(T@Tbc);R=c2w[:3,:3]*np.array([1,-1,-1]);Ri=R.T;ti=-Ri@c2w[:3,3]
    V=np.eye(4);V[:3,:3]=Ri;V[:3,3]=ti;return V
@torch.no_grad()
def rend(pos,yaw,Tbc,Kk,Ww,Hh):
    V=torch.tensor(vm(pos,yaw,Tbc),device=DEV,dtype=torch.float32)[None]
    r,a,_=rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=colors,viewmats=V,Ks=Kk,width=Ww,height=Hh,packed=False,near_plane=0.001,far_plane=1e10,render_mode="RGB",sh_degree=3,rasterize_mode="classic")
    return ((r[...,:3]+(1-a)*bg).clamp(0,1).squeeze(0)*255).byte().cpu().numpy()
def to256(a): return np.asarray(Image.fromarray(a).resize((256,256),Image.BILINEAR),np.uint8)
def ov(x):
    rgb=_ov[...,:3].astype(np.float32);al=_ov[...,3:4].astype(np.float32)/255.;return (al*rgb+(1-al)*x.astype(np.float32)).clip(0,255).astype(np.uint8)
def obs_fwd(pos,yaw): return np.asarray(Image.fromarray(to256(rend(pos,yaw,Tbc_f,Kf,1024,768))).resize((224,224),Image.BICUBIC),np.uint8)
def obs_wrist(pos,yaw): return np.asarray(Image.fromarray(ov(to256(rend(pos,yaw,Tbc_d,Kd,1024,768)))).resize((224,224),Image.BICUBIC),np.uint8)
D3=np.diag([1.,-1,-1])
def to_ns(p): q=Tw2g@np.array([*(D3@p),1.0]); return q[:3]
def proj(p_ns,V):
    pc=V[:3,:3]@p_ns+V[:3,3]; z=float(pc[2])
    if z<=0.05: return None
    return (Kp[0,0]*pc[0]/z+Kp[0,2], Kp[1,1]*pc[1]/z+Kp[1,2], z)
def draw_overlay(frame,fut_world,V):
    im=Image.fromarray(frame).convert("RGB"); dr=ImageDraw.Draw(im,"RGBA")
    pts=[proj(to_ns(p),V) for p in fut_world]
    vis=[pt for pt in pts if pt]
    for i in range(len(vis)-1):
        (u0,v0,_),(u1,v1,_)=vis[i],vis[i+1]; f=i/max(1,len(vis)-1)
        col=(int(60+40*f),255,int(120+120*f),235)
        dr.line([(u0,v0),(u1,v1)],fill=col,width=5)
    for i,(u,v,_) in enumerate(vis[::3]):
        dr.ellipse([u-3,v-3,u+3,v+3],fill=(80,255,160,220))
    if len(vis)>=2:
        (u1,v1,_),(u0,v0,_)=vis[-1],vis[-2]; import math
        ang=math.atan2(v1-v0,u1-u0); L=16
        for da in (2.6,-2.6):
            dr.line([(u1,v1),(u1+L*math.cos(ang+da),v1+L*math.sin(ang+da))],fill=(255,240,80,255),width=5)
        dr.ellipse([u1-6,v1-6,u1+6,v1+6],fill=(255,240,80,255))
    dr.text((10,10),"pin intends -->",fill=(255,255,255,235))
    return np.asarray(im,np.uint8)
BASE_PROMPT=os.environ.get("PROMPT") or "go through the gate on the %s and hover over the stuffed animal"%SIDE
_st=[float(v) for v in os.environ.get("START","0,0,1.5").split(",")]
pol=WebsocketClientPolicy(host="127.0.0.1",port=PORT)
apc=int(os.environ.get("APC","8"))

def run_trial(t):
    PROMPT=BASE_PROMPT
    pos=np.array(_st[:3]); yaw=float(os.environ.get("STARTYAW","0")); fr=[]
    executed=0; traj=[pos.copy()]
    _switched=False; _mid_left=0
    for ci in range(NCH):
        if COMPOSE and not _switched:
            sd_=(np.array(traj)-GANCH)@GNRM
            if len(sd_)>2 and (np.sign(sd_[:-1])!=np.sign(sd_[1:])).any():
                if COMPOSE_MID:
                    PROMPT=COMPOSE_MID; _mid_left=COMPOSE_MID_CHUNKS
                    print(f"[compose] milestone at step {len(traj)}; splicing {COMPOSE_MID_CHUNKS} chunks of intermediate task",flush=True)
                else:
                    PROMPT=COMPOSE
                _switched=True
        elif _mid_left>0:
            _mid_left-=1
            if _mid_left==0:
                PROMPT=COMPOSE
                print(f"[compose] splice done at step {len(traj)}; final task",flush=True)
        imf=obs_fwd(pos,yaw); imw=obs_wrist(pos,yaw)
        o={"observation/image":imf,"observation/wrist_image":imw,"observation/state":np.array([pos[0],pos[1],pos[2],-yaw,0,0,0],np.float32),"prompt":PROMPT,"progress":min(1.0,executed/271.0)}
        o["snmvp_trial"]=f"{SCENE}_{SIDE}_{t}"  # per-trial key: the MDN server's pi-hysteresis latch must not leak across interleaved clients/trials
        if ci==0: o["reset"]=True
        act=np.asarray(pol.infer(o)["actions"])[:,:7]; n=min(len(act),apc)
        cs_all=np.cumsum(act[:,:3],0); cs=cs_all[:n]
        if VIDEO:
            for i in range(0,n,VSTRIDE):
                wp=pos+cs[i]; wy=yaw-float(act[:i+1,3].sum())
                frame=rend(wp,wy,Tbc_f,Kv,Wv,Hv)
                fr.append(draw_overlay(frame, pos+cs_all[i:], vm(wp,wy,Tbc_f)))
        for i in range(n): traj.append(pos+cs[i])
        pos=pos+cs[-1]; yaw=yaw-float(act[:n,3].sum()); executed+=n
        if abs(pos[0])>60 or abs(pos[1])>60: break
    out=OUT.replace("{t}",str(t)) if OUT else ""
    if VIDEO and fr and out: iio.mimsave(out,fr,fps=FPS,codec="libx264",quality=6)
    P=np.array(traj); sd_=(P-GANCH)@GNRM; inb=int(np.all((P>=GAABB[0])&(P<=GAABB[1]),axis=1).sum())
    cr=np.where(np.sign(sd_[:-1])!=np.sign(sd_[1:]))[0]; thru=False; cf=None
    for i in cr:
        tt=sd_[i]/(sd_[i]-sd_[i+1]+1e-9); X=P[i]+tt*(P[i+1]-P[i])
        if GAABB[0][0]<=X[0]<=GAABB[1][0] and GAABB[0][2]<=X[2]<=GAABB[1][2] and not thru: thru=True; cf=(round(float(X[0]),2),round(float(X[1]),2),round(float(X[2]),2))
    if TRAJ: np.save(TRAJ.replace("{t}",str(t)),P)
    print("trial %d: %d frames end x=%.2f y=%.2f z=%.2f | in_aabb=%d THROUGH=%s at %s"%(t,len(fr),pos[0],pos[1],pos[2],inb,thru,cf),flush=True)

for t in range(1,TRIALS+1):
    run_trial(t)
print("BATCH_DONE",flush=True)
