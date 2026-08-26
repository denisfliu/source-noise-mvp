import torch,numpy as np,os
from gsplat import rasterization
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import imageio.v2 as iio
from PIL import Image
NCH=int(os.environ.get("NCH","40")); PORT=int(os.environ["PORT"]); OUT=os.environ["OUT"]
SIDE=os.environ.get("SIDE","left"); SCENE=os.environ.get("SCENE","left")
TRAJ=os.environ.get("TRAJ","")   # optional path to save state-space trajectory .npy
DEV="cuda"
# Tw2g = M_mocap_to_ns @ diag(1,-1,-1); M from joint_mocap_to_nerf.json (ICP-composed, per scene).
# state is in MOCAP frame (z-up); pn=[x,-y,-z] reproduces M_mocap_to_ns@pos.
D=np.diag([1.,-1,-1,1])
if SCENE=="right":
    CK="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/right_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_144353/nerfstudio_models/step-000029999.ckpt"
    M=np.array([[0.136708,-0.001053,0.006031,-0.111938],[0.00108,0.13684,-0.000588,0.030456],[-0.006027,0.000635,0.136711,-0.201447],[0,0,0,1.]])
    Tw2g=M@D
    # right_gate.yaml gate_region (mocap)
    GAABB=(np.array([-0.06,-1.55,0.05]),np.array([1.15,-0.75,2.05]))
    GANCH=np.array([0.544,-1.147,0.074]); GNRM=np.array([0.385,-0.923,0.0])
else:
    CK="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt"
    Tw2g=np.array([[0.12614431661544656,2.138646801849853e-06,-0.00025306576654559085,-0.15671883492487332],[-2.138646801849853e-06,-0.1261265572041315,-0.0021319289354524646,-0.08013551648879384],[-0.00025306576654559085,0.0021319289354524646,-0.12612630156484925,-0.18772133850562778],[0,0,0,1.]])
    # left_gate.yaml gate_region (mocap)
    GAABB=(np.array([0.36,0.12,0.05]),np.array([1.36,1.27,2.05]))
    GANCH=np.array([0.861,0.694,0.075]); GNRM=np.array([0.749,0.663,0.0])
GNRM=GNRM/np.linalg.norm(GNRM); APER=0.45
sd=torch.load(CK,map_location=DEV,weights_only=False)["pipeline"]
def gg(n):
    for p in ("_model.gauss_params.","_model."):
        if p+n in sd: return sd[p+n].to(DEV)
means=gg("means");quats=gg("quats");scales=torch.exp(gg("scales"));opac=torch.sigmoid(gg("opacities")).squeeze(-1)
colors=torch.cat([gg("features_dc")[:,None,:],gg("features_rest")],1); bg=torch.tensor([0.149,0.1647,0.2157],device=DEV)
Wv,Hv=512,384; sx,sy=Wv/1024.,Hv/768.
Kv=torch.tensor([[502.2632*sx,0,506.3971*sx],[0,500.6736*sy,385.41*sy],[0,0,1.]],device=DEV,dtype=torch.float32)[None]
Kf=torch.tensor([[502.2632,0.,506.3971],[0.,500.6736,385.41],[0.,0.,1.]],device=DEV,dtype=torch.float32)[None]   # forward 1024x768
Kd=torch.tensor([[478.2450,0.,511.9041],[0.,476.7944,383.5003],[0.,0.,1.]],device=DEV,dtype=torch.float32)[None]  # downward 1024x768
Tbc_f=np.array([[0,0,-1,0.10],[1,0,0,-0.03],[0,-1,0,-0.01],[0,0,0,1.]])           # cam_body<-forward (inv of scene edge)
Tbc_d=np.array([[0,1,0,0.0],[1,0,0,0.0],[0,0,-1,0.05],[0,0,0,1.]])                 # cam_body<-downward (inv of scene edge)
# downward wrist strut overlay (RGBA 256^2), composited after the 256 resize (CameraPostprocess.apply)
OVP="/home/ubuntu/code/falsify-pi/configs/embodiments/assets/carl_wrist_overlay_pinhole_rgb.png"
try:
    _ov=np.asarray(Image.open(OVP).convert("RGBA").resize((256,256),Image.BILINEAR),np.uint8)
except Exception:
    _ov=None
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
def _to256(a): return np.asarray(Image.fromarray(a).resize((256,256),Image.BILINEAR),np.uint8)  # native->256 squash (PIL bilinear, matches CameraPostprocess)
def _overlay(img256):
    if _ov is None: return img256
    rgb=_ov[...,:3].astype(np.float32); al=_ov[...,3:4].astype(np.float32)/255.
    return (al*rgb+(1-al)*img256.astype(np.float32)).clip(0,255).astype(np.uint8)
def obs_fwd(pos,yaw):   # observation/image : forward 1024x768 -> 256 -> 224 bicubic (matches make_u_rrr_gate)
    a=_to256(rend(pos,yaw,Tbc_f,Kf,1024,768)); return np.asarray(Image.fromarray(a).resize((224,224),Image.BICUBIC),np.uint8)
def obs_wrist(pos,yaw): # observation/wrist_image : downward 1024x768 -> 256 -> strut overlay -> 224 bicubic
    a=_overlay(_to256(rend(pos,yaw,Tbc_d,Kd,1024,768))); return np.asarray(Image.fromarray(a).resize((224,224),Image.BICUBIC),np.uint8)
PROMPT="go through the gate on the %s and hover over the stuffed animal"%SIDE
pol=WebsocketClientPolicy(host="127.0.0.1",port=PORT); pos=np.array([0.,0.,1.5]); yaw=0.0; fr=[]; apc=8; executed=0
traj=[pos.copy()]
for ci in range(NCH):
    imf=obs_fwd(pos,yaw); imw=obs_wrist(pos,yaw)
    o={"observation/image":imf,"observation/wrist_image":imw,"observation/state":np.array([pos[0],pos[1],pos[2],-yaw,0,0,0],np.float32),"prompt":PROMPT,"progress":min(1.0,executed/271.0)}
    if ci==0: o["reset"]=True
    act=np.asarray(pol.infer(o)["actions"])[:,:7]; n=min(len(act),apc); cs=np.cumsum(act[:n,:3],0)
    for i in range(0,n,4):
        wp=pos+cs[i]; wy=yaw-float(act[:i+1,3].sum()); fr.append(rend(wp,wy,Tbc_f,Kv,Wv,Hv))
    for i in range(n): traj.append(pos+cs[i])
    pos=pos+cs[-1]; yaw=yaw-float(act[:n,3].sum()); executed+=n
    if abs(pos[0])>60 or abs(pos[1])>60: break
iio.mimsave(OUT,fr,fps=24,codec="libx264",quality=6)
# ---- state-space THROUGH-APERTURE scoring (render-independent) ----
# at each gate-plane crossing, is the crossing point within APER of the anchor, in-plane? (matches make_u_rrr_gate.is_left)
P=np.array(traj); sd=(P-GANCH)@GNRM
inb=int(np.all((P>=GAABB[0])&(P<=GAABB[1]),axis=1).sum())
cr=np.where(np.sign(sd[:-1])!=np.sign(sd[1:]))[0]
best=9.9; thru=False; cf=-1
for i in cr:
    t=sd[i]/(sd[i]-sd[i+1]+1e-9); xp=P[i]+t*(P[i+1]-P[i]); d=xp-GANCH
    rad=float(np.linalg.norm(d-(d@GNRM)*GNRM))
    if rad<best: best=rad
    if rad<=APER and not thru: thru=True; cf=int(i)
if TRAJ: np.save(TRAJ,P)
print("%s: %d frames, end x=%.2f y=%.2f z=%.2f | GATE in_aabb=%d aperture_closest=%.3f (APER=%.2f) THROUGH=%s @f%d"%(OUT,len(fr),pos[0],pos[1],pos[2],inb,best,APER,thru,cf),flush=True)
