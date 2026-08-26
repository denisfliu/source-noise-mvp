"""Is the pin just a pass-through? Instrument the ODE: track the pinned lateral
bin-1 coefficient of x_t across all Euler steps (pin vs condition), and the
velocity component along the pinned direction. If the pin clamps the coordinate
(v~0 there), bin-1(x_t) is CONSTANT = command for all t (not denoised); the
complement still flows. Conditioning must GENERATE bin-1, so it evolves from the
random source value toward the command.
"""
import json, os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import steer_probe as sp
from dataset import H, make_dataset, to_canonical

def bin1(chunk_canon):
    z=chunk_canon[...,1]; c=np.fft.rfft(z,axis=-1)[...,1]
    return np.stack([c.real,c.imag],-1)

def run(arm, params, obs, ang, C):
    n=obs.shape[0]; rng=np.random.default_rng(0); eps=rng.normal(size=(n,H,2)); cmd=None
    if arm=="pin":
        ec=to_canonical(eps,ang); ph,mg=sp.cmd_to_pin(C)
        ec=sp.pin_noise(ec,sp.PINSPEC,ph,mag_targets=mg,orient_from_noise=False); S=to_canonical(ec,-ang)
    else:
        S=eps;
        if arm=="condition": cmd=C
    x=S; traj=[]; vproj=[]
    # pinned direction (lateral bin-1 real/imag) unit vectors in chunk space
    dRe=sp.fourier_dir("lat",1,"re"); dIm=sp.fourier_dir("lat",1,"im")
    for k in range(sp.EULER):
        t=np.full(n,1.0-k/sp.EULER)
        v=np.asarray(sp.vfield(params["v"],x,t,obs,cmd)).reshape(n,-1)
        # velocity magnitude along pinned dirs vs total
        vp=np.sqrt((v@dRe)**2+(v@dIm)**2).mean(); vt=np.sqrt((v**2).sum(1)).mean()
        vproj.append(float(vp/(vt+1e-9)))
        xc=to_canonical(x,ang); b=bin1(xc); traj.append([float(b[:,0].mean()),float(b[:,1].mean())])
        x=x-(1.0/sp.EULER)*v.reshape(n,H,2)
    xc=to_canonical(x,ang); b=bin1(xc); traj.append([float(b[:,0].mean()),float(b[:,1].mean())])
    return traj, vproj

def main():
    sc,obs,ch,ang=make_dataset(200,8,np.random.default_rng(7))
    fo,fc,fa=np.repeat(obs,8,0),ch.reshape(-1,H,2),np.repeat(ang,8)
    he_sc,he_o,he_ch,he_a=make_dataset(60,8,np.random.default_rng(7777))
    C_nat=sp.coeffs(to_canonical(he_ch,he_a[:,None])).mean(1)
    C=C_nat.copy(); C[:,0]=0.0; C[:,1]=2.0   # command a big pure-Im1 bend (=2.0)
    out={"command_Im1":2.0,"arms":{}}
    for arm in ["pin","condition"]:
        p=sp.train(arm,fo,fc,fa)
        traj,vproj=run(arm,p,he_o,he_a,C)
        out["arms"][arm]={"bin1_Im_over_t":[round(b[1],3) for b in traj],
                          "bin1_Re_over_t":[round(b[0],3) for b in traj],
                          "vel_frac_on_pinned_dir":[round(v,3) for v in vproj]}
        print(arm,"Im1(x_t) t=1..0:",[round(b[1],2) for b in traj],flush=True)
        print("   vel-frac on pinned dir:",[round(v,2) for v in vproj[:6]],"...",flush=True)
    json.dump(out,open(os.path.join(HERE,"results","passthrough.json"),"w"),indent=2)
    print("DONE")

if __name__=="__main__": main()
