import os,re,sys,json,numpy as np
os.environ.update(SCENE="left_and_center",SIDE="left",PORT="1",TRIALS="1",OUT="/dev/null/{t}",TRAJ="",AGENT_LR_MARKS="1")
sys.path.insert(0,"experiments/rung3")
src=open("experiments/rung3/gate_rollout_batch.py").read(); src=src[:src.index("apc=int(")]
src=re.sub(r"pol=WebsocketClientPolicy\([^\n]*\)","pol=None",src)
g={"__name__":"diag"}; exec(compile(src,"grb","exec"),g)
from PIL import Image, ImageDraw
obs_fwd,obs_wrist=g["obs_fwd"],g["obs_wrist"]
M=np.array([7.4,-0.2]); out=sys.argv[1]; V=224; truth=[]
rng=np.random.default_rng(0); i=0
for dist in (3.0,4.5):
    for rel in (-30,-20,-12,-6,6,12,20,30):   # relative bearing of the mannequin, deg; negative = right of heading
        # pick a viewpoint at `dist` from the figure, heading such that the figure sits at `rel`
        az=np.radians(180+rng.uniform(-25,25))   # approach roughly from -x side
        pos=np.array([M[0]+dist*np.cos(az), M[1]+dist*np.sin(az), 1.4])
        b=np.arctan2(M[1]-pos[1],M[0]-pos[0]); head=b-np.radians(rel); yaw=-head
        imf=obs_fwd(pos,yaw); imw=obs_wrist(pos,yaw)
        big=Image.new("RGB",(6+V+8+V+6,6+V+6),(18,20,26)); big.paste(Image.fromarray(imf),(6,6)); big.paste(Image.fromarray(imw),(6+V+8,6))
        dr=ImageDraw.Draw(big); yb=6+V-13
        for t,x in (("L",10),("R",6+V-14)): dr.text((x,yb),t,fill=(0,0,0)); dr.text((x-1,yb-1),t,fill=(255,230,0))
        dr.line([(6+V//2,6),(6+V//2,14)],fill=(255,230,0)); dr.line([(6+V//2,6+V-8),(6+V//2,6+V)],fill=(255,230,0))
        fn=f"{out}/probe_{i:02d}.jpg"; big.save(fn,quality=85)
        truth.append({"file":fn,"rel_deg":rel,"dist":dist,"side":"right" if rel<0 else "left"}); i+=1
json.dump(truth,open(f"{out}/truth.json","w"),indent=1); print("rendered",len(truth))
