"""Does the toy have enough MULTIMODALITY to require flow matching? And does
predicting Fourier modes bypass the modality gap?

Centered-obstacle toy: obstacle on the start->target line (lateral=0), so BOTH
detour sides are equally valid for every scene => irreducible bimodality GIVEN
obs (the side is not determined by the observation). The straight (mode-averaged)
path goes into the obstacle.

Arms (all trained on the same bimodal data):
  regress    MLP obs->action, MSE (deterministic; MUST mode-average -> ~straight)
  flow       plain flow-matching, source noise, no command (samples a side)
  reg_assign regress, then OVERWRITE lateral bin-1 with a commanded side (+A)
  flow_pin   flow with lateral bin-1 pinned to the commanded side (+A)

Predictions: regress FAILS (averages sides, |bin1|~0, hits obstacle); flow
SUCCEEDS (bimodal, picks a side); reg_assign & flow_pin SUCCEED (command picks
the mode). If regress succeeds anyway, the toy can't tell flow from regression.
"""
import json, os, sys, numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.misc.optimizers import adam
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import dataset as ds
from dataset import H, ACT_SCALE, to_canonical, success, scene_obs, make_demo, make_scene
from pin import pin_noise
np.seterr(all="ignore")

OBS=5; HID=128; ITERS=10000; BATCH=256; EULER=20
LATPIN=[{"axis":(0.0,1.0),"omega":1,"mode":"mod2pi","mag":True}]

def centered_dataset(n_scenes,n_demos,rng):
    scenes,obs,chunks,angles=[],[],[],[]
    for _ in range(n_scenes):
        sc=make_scene(rng); sc["lateral"]=0.0          # obstacle ON the line -> both sides viable
        scenes.append(sc); obs.append(scene_obs(sc)); angles.append(sc["angle"])
        chunks.append(np.stack([make_demo(sc,rng) for _ in range(n_demos)]))
    return scenes,np.array(obs),np.stack(chunks),np.array(angles)

def mlp_init(dims,rng):
    return [(rng.normal(size=(a,b))/np.sqrt(a),np.zeros(b)) for a,b in zip(dims[:-1],dims[1:])]
def mlp(p,x):
    h=x
    for w,b in p[:-1]: h=anp.maximum(0.0,h@w+b)
    w,b=p[-1]; return h@w+b

def train_regress(obs,chunks,seed=0):
    y=chunks.reshape(chunks.shape[0],-1); n=obs.shape[0]
    p=mlp_init([OBS,HID,HID,HID,H*2],np.random.default_rng(seed))
    def loss(pp,it):
        r=np.random.default_rng(it); idx=r.integers(0,n,size=BATCH)
        return anp.mean((mlp(pp,obs[idx])-y[idx])**2)
    return adam(grad(loss),p,num_iters=ITERS,step_size=1e-3)

def vfield(vp,xt,t,obs):
    h=anp.concatenate([xt.reshape(xt.shape[0],-1),t.reshape(-1,1),obs],axis=1)
    return mlp(vp,h).reshape(xt.shape[0],H,2)

def train_flow(obs,chunks,angles,pin,seed=0):
    n=obs.shape[0]; canon=to_canonical(chunks,angles)
    p=mlp_init([H*2+1+OBS,HID,HID,HID,H*2],np.random.default_rng(seed))
    def loss(pp,it):
        r=np.random.default_rng(it); idx=r.integers(0,n,size=BATCH)
        a0=chunks[idx]; ob=obs[idx]; ang=angles[idx]; eps=r.normal(size=(BATCH,H,2))
        if pin:
            ec=to_canonical(eps,ang); c=canon[idx][...,1]
            spec=np.fft.rfft(c,axis=-1)[:,1]
            ph=np.angle(spec)[:,None]; mg=np.abs(spec)[:,None]
            ec=pin_noise(ec,LATPIN,ph,mag_targets=mg); eps=to_canonical(ec,-ang)
        t=r.uniform(0,1,size=BATCH); xt=t[:,None,None]*eps+(1-t[:,None,None])*a0
        return anp.mean((vfield(pp,xt,t,ob)-(eps-a0))**2)
    return adam(grad(loss),p,num_iters=ITERS,step_size=1e-3)

def flow_rollout(vp,obs,angles,rng,pin_cmd=None):
    n=obs.shape[0]; eps=rng.normal(size=(n,H,2))
    if pin_cmd is not None:
        ec=to_canonical(eps,angles); ph=np.full((n,1),pin_cmd[0]); mg=np.full((n,1),pin_cmd[1])
        ec=pin_noise(ec,LATPIN,ph,mag_targets=mg,orient_from_noise=False); eps=to_canonical(ec,-angles)
    x=eps
    for k in range(EULER):
        t=np.full(n,1.0-k/EULER); x=x-(1.0/EULER)*np.asarray(vfield(vp,x,t,obs))
    return x

def bin1_mag(chunk,angles):
    c=to_canonical(chunk,angles)[...,1]; s=np.fft.rfft(c,axis=-1)[...,1]; return np.abs(s), np.angle(s)

def main():
    rng=np.random.default_rng(7)
    sc,obs,ch,ang=centered_dataset(200,8,rng)
    fo,fc,fa=np.repeat(obs,8,0),ch.reshape(-1,H,2),np.repeat(ang,8)
    he_sc,he_o,he_ch,he_a=centered_dataset(80,8,np.random.default_rng(7777))
    # data bimodality check: sides in demos
    dmag,dph=bin1_mag(fc,fa)
    frac_pos=float(np.mean(np.sin(dph)>0))
    print(f"data: mean|bin1|={dmag.mean():.2f}, frac side+={frac_pos:.2f} (0.5=balanced bimodal)",flush=True)
    res={"data_mean_bin1":round(float(dmag.mean()),3),"data_frac_sidepos":round(frac_pos,3),"arms":{}}
    typ=float(dmag.mean())

    # regress
    reg=train_regress(fo,fc)
    pred=np.asarray(mlp(reg,he_o)).reshape(-1,H,2)
    m,_=bin1_mag(pred,he_a)
    s=float(np.mean([success(he_sc[i],pred[i]) for i in range(len(he_sc))]))
    res["arms"]["regress"]={"success":round(s,3),"produced_mean_bin1":round(float(m.mean()),3)}
    print(f"regress: success={s:.3f} produced|bin1|={m.mean():.3f} (data {typ:.2f}) -> if ~0, averaged the sides",flush=True)

    # reg_assign: overwrite lateral bin-1 with +A
    Aph=0.0 if np.sin(dph[dmag>0]).mean()>=0 else np.pi   # a definite side
    def assign(pred,phase,mag):
        c=to_canonical(pred,he_a); z=c[...,1]; spec=np.fft.rfft(z,axis=-1)
        spec[...,1]=mag*np.exp(1j*phase); c2=c.copy(); c2[...,1]=np.fft.irfft(spec,n=H,axis=-1)
        return to_canonical(c2,-he_a)
    ga=assign(pred,np.pi/2,typ)   # phase pi/2 = pure +Im (one side), magnitude=typical
    s=float(np.mean([success(he_sc[i],ga[i]) for i in range(len(he_sc))]))
    res["arms"]["reg_assign"]={"success":round(s,3)}
    print(f"reg_assign(+side): success={s:.3f}",flush=True)

    # flow plain (no command): sample a side
    fp=train_flow(fo,fc,fa,pin=False)
    # eval with 8 rollouts to see multimodality
    accs=[]; sidewise=[]
    for rr in range(8):
        ro=flow_rollout(fp,he_o,he_a,np.random.default_rng(rr))
        accs.append(np.mean([success(he_sc[i],ro[i]) for i in range(len(he_sc))]))
        m2,p2=bin1_mag(ro,he_a); sidewise.append(float(np.mean(np.sin(p2)>0)))
    res["arms"]["flow"]={"success":round(float(np.mean(accs)),3),
                         "mean_frac_side_pos_per_rollout":round(float(np.mean(sidewise)),3)}
    print(f"flow: success={np.mean(accs):.3f} frac side+ per rollout={np.mean(sidewise):.2f} (mix=samples both)",flush=True)

    # flow + pin (+side)
    fpin=train_flow(fo,fc,fa,pin=True)
    ro=flow_rollout(fpin,he_o,he_a,np.random.default_rng(0),pin_cmd=(np.pi/2,typ))
    s=float(np.mean([success(he_sc[i],ro[i]) for i in range(len(he_sc))]))
    res["arms"]["flow_pin"]={"success":round(s,3)}
    print(f"flow_pin(+side): success={s:.3f}",flush=True)

    json.dump(res,open(os.path.join(HERE,"results","multimodal.json"),"w"),indent=2)
    print("VERDICT: regress vs flow success gap =",
          round(res["arms"]["flow"]["success"]-res["arms"]["regress"]["success"],3),
          "(large + => toy IS multimodal & flow matters; ~0 => toy inadequate)")
    print("DONE")

if __name__=="__main__": main()
