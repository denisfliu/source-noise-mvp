import torch, numpy as np, time
from gsplat import rasterization
from PIL import Image
from openpi_client.websocket_client_policy import WebsocketClientPolicy
DEV="cuda"
CK="/home/ubuntu/code/falsify-pi/data/gate_scenes_export/left_scene/mocap_outputs/sagesplat_mocap/sagesplat/2026-05-11_153901/nerfstudio_models/step-000029999.ckpt"
sd=torch.load(CK,map_location=DEV,weights_only=False)["pipeline"]
def g(n):
    for p in ("_model.gauss_params.","_model."):
        if p+n in sd: return sd[p+n].to(DEV)
means=g("means");quats=g("quats");scales=torch.exp(g("scales"));opac=torch.sigmoid(g("opacities")).squeeze(-1)
colors=torch.cat([g("features_dc")[:,None,:],g("features_rest")],1)
K=torch.tensor([[502.2632,0.,506.3971],[0.,500.6736,385.41],[0.,0.,1.]],device=DEV)[None]
bg=torch.tensor([0.149,0.1647,0.2157],device=DEV); W,H=1024,768
Tw2g=np.array([[0.12614431661544656,2.138646801849853e-06,-0.00025306576654559085,-0.15671883492487332],[-2.138646801849853e-06,-0.1261265572041315,-0.0021319289354524646,-0.08013551648879384],[-0.00025306576654559085,0.0021319289354524646,-0.12612630156484925,-0.18772133850562778],[0,0,0,1.]])
Tbc=np.array([[0,0,-1,0.10],[1,0,0,-0.03],[0,-1,0,-0.01],[0,0,0,1.]])
def Rz(p): c,s=np.cos(p),np.sin(p); return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def viewmat(pos,yaw):
    pn=np.array([pos[0],-pos[1],-pos[2]]); T=np.eye(4); T[:3,:3]=Rz(yaw); T[:3,3]=pn
    c2w=Tw2g@(T@Tbc); R=c2w[:3,:3]*np.array([1,-1,-1]); Ri=R.T; ti=-Ri@c2w[:3,3]
    V=np.eye(4); V[:3,:3]=Ri; V[:3,3]=ti; return V
@torch.no_grad()
def render(pos,yaw):
    V=torch.tensor(viewmat(pos,yaw),device=DEV,dtype=torch.float32)[None]
    r,a,_=rasterization(means=means,quats=quats,scales=scales,opacities=opac,colors=colors,viewmats=V,Ks=K,width=W,height=H,packed=False,near_plane=0.001,far_plane=1e10,render_mode="RGB",sh_degree=3,rasterize_mode="classic")
    rgb=(r[...,:3]+(1-a)*bg).clamp(0,1).squeeze(0)
    im=Image.fromarray((rgb*255).byte().cpu().numpy()).resize((224,224),Image.BILINEAR)
    return np.asarray(im,np.uint8)

LEFT="go through the gate on the left and hover over the stuffed animal"
RIGHT="go through the gate on the right and hover over the stuffed animal"
def rollout(port,prompt,apc=8,nchunks=30):
    pol=WebsocketClientPolicy(host="127.0.0.1",port=port)
    pos=np.array([0.,0.,1.5]); yaw=0.0; tr=[(0.,0.)]
    for _ in range(nchunks):
        img=render(pos,yaw)
        obs={"observation/image":img,"observation/wrist_image":img,"observation/state":np.array([pos[0],pos[1],pos[2],-yaw,0,0,0],np.float32),"prompt":prompt}
        act=np.asarray(pol.infer(obs)["actions"])[:,:7]; n=min(len(act),apc)
        pos=pos+act[:n,:3].sum(0); yaw=yaw-float(act[:n,3].sum()); tr.append((float(pos[0]),float(pos[1])))
    return tr
for name,port,prompt in [("scratch_LEFT",8777,LEFT),("scratch_RIGHT",8777,RIGHT),("pin_LEFT",8778,LEFT),("pin_RIGHT",8778,RIGHT)]:
    tr=rollout(port,prompt); xs=[p[0] for p in tr]; ys=[p[1] for p in tr]
    print(f"{name:14s} | end (x={xs[-1]:+.2f}, y={ys[-1]:+.2f}) | y[::6]={[round(v,2) for v in ys[::6]]}",flush=True)
print("STEER_DONE")
