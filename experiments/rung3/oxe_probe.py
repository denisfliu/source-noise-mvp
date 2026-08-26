import tensorflow_datasets as tfds
import numpy as np
b = tfds.builder_from_directory("gs://gresearch/robotics/berkeley_autolab_ur5/0.1.0")
ds = b.as_dataset(split="train[:2]")
n=0
for ep in ds:
    acts=[]
    for st in ep["steps"]:
        a=st["action"]
        wv=a["world_vector"].numpy(); rd=a["rotation_delta"].numpy()
        acts.append(np.concatenate([wv, rd]))
    acts=np.array(acts)
    print("episode len", acts.shape, "sample", np.round(acts[0],3))
    n+=1
print("EPISODES_READ", n, "OXE_PROBE_OK")
