"""Ablation: is the source-noise pin equivalent to just PREDICTING FOURIER MODES
(no flow matching)?

no_flow: a plain MLP regresses obs -> canonical action chunk (MSE, deterministic,
NO source noise, NO flow). To steer, overwrite the chosen structure modes
(lateral bins 1,2) of the predicted chunk with the command and inverse-FFT.
Compare to the pin (flow) on: steer exactness (trivially perfect for both) and
TASK SUCCESS. If no_flow matches the pin, flow matching adds nothing for control
or success here -- we are just predicting Fourier modes.
"""
import json, os, sys, numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.misc.optimizers import adam
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import steer_probe as sp
from dataset import H, ACT_SCALE, make_dataset, success, to_canonical

OBS=5; HID=128; ITERS=12000; BATCH=256
BINS=[1,2]  # lateral structure modes to clamp/assign

def mlp_init(dims,rng):
    return [(rng.normal(size=(a,b))/np.sqrt(a),np.zeros(b)) for a,b in zip(dims[:-1],dims[1:])]
def mlp(p,x):
    h=x
    for w,b in p[:-1]: h=anp.maximum(0.0,h@w+b)
    w,b=p[-1]; return h@w+b

def train_regress(obs,canon,seed=0):
    y=canon.reshape(canon.shape[0],-1); n=obs.shape[0]
    p=mlp_init([OBS,HID,HID,HID,H*2],np.random.default_rng(seed))
    def loss(pp,it):
        r=np.random.default_rng(it); idx=r.integers(0,n,size=BATCH)
        return anp.mean((mlp(pp,obs[idx])-y[idx])**2)
    return adam(grad(loss),p,num_iters=ITERS,step_size=1e-3)

def steer_modes(chunk_canon, C):
    """Overwrite lateral bins 1,2 (from command C=[Re1,Im1,Re2,Im2]) and iFFT."""
    z=chunk_canon[...,1].copy()
    spec=np.fft.rfft(z,axis=-1)
    spec[...,1]=C[:,0]+1j*C[:,1]; spec[...,2]=C[:,2]+1j*C[:,3]
    lat=np.fft.irfft(spec,n=H,axis=-1)
    out=chunk_canon.copy(); out[...,1]=lat
    return out

def main():
    sc,obs,ch,ang=make_dataset(200,8,np.random.default_rng(7))
    fo,fc,fa=np.repeat(obs,8,0),ch.reshape(-1,H,2),np.repeat(ang,8)
    canon=to_canonical(fc,fa)
    he_sc,he_o,he_ch,he_a=make_dataset(60,8,np.random.default_rng(7777))
    C_nat=sp.coeffs(to_canonical(he_ch,he_a[:,None])).mean(1)  # 6D; use first 4 (lat bins)
    scale=np.abs(C_nat).mean(0)+1e-6

    reg=train_regress(fo,canon)
    def predict(obs): return np.asarray(mlp(reg,obs)).reshape(-1,H,2)

    def evalC(C4):
        pred_c=predict(he_o)                              # canonical chunk
        steered_c=steer_modes(pred_c, C4)                 # overwrite structure modes
        glob=to_canonical(steered_c,-he_a)                # back to global
        succ=np.mean([success(he_sc[i],glob[i]) for i in range(len(he_sc))])
        prod=sp.coeffs(to_canonical(glob,he_a))[:, :4]    # produced structure modes
        follow=float(np.sqrt(((prod-C4)**2).sum(1)).mean())
        return round(float(succ),3), round(follow,4)

    # natural command (scene structure) and a pure-Im1 steer sweep
    nat4=C_nat[:, :4].copy()
    s_nat,f_nat=evalC(nat4)
    slope=[]
    for a in np.linspace(-2.5,2.5,9):
        C=nat4.copy(); C[:,0]=0.0; C[:,1]=a
        pred_c=predict(he_o); steered=steer_modes(pred_c,C); glob=to_canonical(steered,-he_a)
        prod=sp.coeffs(to_canonical(glob,he_a))[:,1]
        slope.append(float(prod.mean()))
    sl=float(np.polyfit(np.linspace(-2.5,2.5,9),slope,1)[0])
    # contradictory (flip natural bend) success = obedience tax check
    s_con,_=evalC(-nat4)
    out={"no_flow":{"success_natural":s_nat,"follow_err_natural":f_nat,
                    "steer_slope":round(sl,3),"success_contradictory":s_con},
         "pin_reference":{"success_natural":0.8,"steer_slope":1.0,"phase_err_deg":1.1},
         "note":"if no_flow success_natural ~ pin 0.8 and slope ~1, flow matching "
                "adds nothing for control/success -> we are just predicting Fourier modes."}
    json.dump(out,open(os.path.join(HERE,"results","ablation.json"),"w"),indent=2)
    print(json.dumps(out,indent=2)); print("DONE")

if __name__=="__main__": main()
