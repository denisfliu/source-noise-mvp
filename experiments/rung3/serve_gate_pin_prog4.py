"""Serve the gate pi0 pin with a PROGRESS-conditioned prior (4-task exact-match variant): c = MLP([model_state, onehot, progress]).
progress in [0,1] is supplied by the client as obs['progress'] (its step count / expected length)."""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import pin_basis
import steer_c
import openpi.training.config as _cfg
import openpi.policies.policy_config as _pc
import openpi.shared.normalize as _nz
from openpi.transforms import NormStats
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
def _pad(ns,dim):
    out={}
    for k,s in ns.items():
        n=np.asarray(s.mean).shape[-1]
        if n>=dim: out[k]=s;continue
        p=dim-n; ext=lambda a,f: None if a is None else np.concatenate([np.asarray(a,np.float32),np.full(p,f,np.float32)])
        out[k]=NormStats(mean=ext(s.mean,0.),std=ext(s.std,1.),q01=ext(s.q01,0.),q99=ext(s.q99,1.))
    return out
class ProgPinPolicy:
    def __init__(self, policy, pin_u_path, prior_path, astd=None):
        self.policy=policy; self.U=np.load(pin_u_path).astype(np.float32)
        # NUDGE='z:+0.30,y:-0.10' steers the command in METRES of net chunk displacement
        # (steer_c: closed-form, ~99% expressible for x/y/z in this basis)
        self._astd = astd
        self.last_c = None
        # CLOG=<path>: (model_state, c) per inference, so the two command sources can be compared
        # offline at the states the flights actually visited — not just at demo states.
        self.CLOG = os.environ.get("CLOG", ""); self._log = []
        self.dc = steer_c.parse_env(self.U, astd) if astd is not None else None
        if self.dc is not None:
            print(f"[steer] NUDGE={os.environ.get('NUDGE')} -> dc={np.round(self.dc,2)}", flush=True)
        d=torch.load(prior_path,map_location="cpu",weights_only=False)
        pin_basis.verify(d, pin_u_path)
        self.tasks=d["tasks"]; self.H=d["H"]; self.AD=d["AD"]; self.K=d["K"]
        self.mu=d["mu"].astype(np.float32); self.sd=d["sd"].astype(np.float32); self.exp_len=d.get("exp_len",271)
        layers,din=[],d["in_dim"]
        for h in d["hidden"]: layers+=[nn.Linear(din,h),nn.SiLU()]; din=h
        layers+=[nn.Linear(din,self.K)]
        self.prior=nn.Sequential(*layers); self.prior.load_state_dict(d["state_dict"]); self.prior.eval()
        self._rng=np.random.default_rng()
        print(f"[prog] in_dim={d['in_dim']} exp_len={self.exp_len} K={self.K}",flush=True)
    def _oh(self,prompt):
        # exact task-string match (the 4-task list contains "left" in two prompts,
        # so keyword matching is ambiguous — fail loudly rather than guess)
        v=np.zeros(len(self.tasks),np.float32)
        p=str(prompt).strip()
        if p in self.tasks:
            v[self.tasks.index(p)]=1.0; return v
        raise ValueError(f"prompt not in prior's task list: {p!r}")
    def infer(self, obs):
        ms=np.asarray(self.policy._input_transform(dict(obs))["state"]).reshape(-1)
        oh=self._oh(obs.get("prompt","")); prog=np.float32(obs.get("progress",0.0))
        # no-progress ablation priors omit the clock input; detect by width
        parts=[ms,oh] if len(self.mu)==len(ms)+len(oh) else [ms,oh,[prog]]
        x=np.concatenate(parts).astype(np.float32); xn=(x-self.mu)/self.sd
        with torch.no_grad(): c=self.prior(torch.tensor(xn[None]))[0].numpy()
        if self.dc is not None: c = c + self.dc
        # live steering: the client may send metres per axis, e.g. {"snmvp_nudge":[0,0,0.3]}
        nud = obs.get("snmvp_nudge")
        if nud is not None and self._astd is not None:
            nud = np.asarray(nud, np.float32).reshape(-1)
            for j, metres in enumerate(nud[:3]):
                if abs(float(metres)) > 1e-6:
                    c = c + steer_c.nudge_vector(self.U, self._astd, j, float(metres))
        self.last_c = c
        g=self._rng.standard_normal((self.H,self.AD)).astype(np.float32).reshape(-1)
        if self.CLOG:
            self._log.append(np.concatenate([np.asarray(x).reshape(-1), np.asarray(c).reshape(-1)]).astype(np.float32))
            np.save(self.CLOG, np.stack(self._log))

        noise=(g-(g@self.U)@self.U.T+(c@self.U.T)).reshape(self.H,self.AD).astype(np.float32)
        return self.policy.infer(obs, noise=noise)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True); ap.add_argument("--norm",required=True)
    ap.add_argument("--pin-u",required=True); ap.add_argument("--prior",required=True)
    ap.add_argument("--config",default="pi0_gate"); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=8784)
    a=ap.parse_args()
    cfg=_cfg.get_config(a.config); ns=_pad(_nz.load(a.norm),cfg.model.action_dim)
    policy=_pc.create_trained_policy(cfg,a.ckpt,norm_stats=ns)
    pin=ProgPinPolicy(policy,a.pin_u,a.prior,astd=np.asarray(ns["actions"].std))
    print(f"[serve_gate_pin_prog] ready on ws://{a.host}:{a.port}",flush=True)
    WebsocketPolicyServer(pin,host=a.host,port=a.port).serve_forever()
if __name__=="__main__": main()
