"""Idempotent patch: env-gated task holdout in create_torch_dataset via a torch
Subset over the source episodes' frame ranges. Avoids LeRobot's `episodes=` filter,
which re-indexes episode_data_index to the subset size yet still looks it up with
original global episode indices (IndexError). When SNMVP_EPISODES points to a JSON list
of episode indices, only frames from those episodes are used. No-op otherwise."""
import re, sys
PATH = sys.argv[1]
src = open(PATH).read()
if "SNMVP_EPISODES" in src:
    print("already patched"); sys.exit(0)

# insert the Subset holdout just before create_torch_dataset returns
anchor = "    if data_config.prompt_from_task:\n        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n\n    return dataset\n"
inject = (
    "    if data_config.prompt_from_task:\n"
    "        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n"
    "\n"
    "    import os as _os, json as _json\n"
    "    _ep_path = _os.environ.get(\"SNMVP_EPISODES\")\n"
    "    if _ep_path:\n"
    "        import torch as _torch\n"
    "        _eps = sorted(set(_json.load(open(_ep_path))))\n"
    "        _base = dataset\n"
    "        while hasattr(_base, \"_dataset\"):\n"
    "            _base = _base._dataset\n"
    "        _edi = _base.episode_data_index\n"
    "        _idx = []\n"
    "        for _e in _eps:\n"
    "            _idx.extend(range(int(_edi[\"from\"][_e]), int(_edi[\"to\"][_e])))\n"
    "        import logging as _lg; _lg.getLogger(\"openpi\").info(\n"
    "            f\"SNMVP holdout: {len(_eps)} episodes -> {len(_idx)} frames (Subset)\")\n"
    "        dataset = _torch.utils.data.Subset(dataset, _idx)\n"
    "\n"
    "    return dataset\n"
)
src, n = re.subn(re.escape(anchor), inject, src, count=1)
assert n == 1, "return-dataset anchor not found"
open(PATH, "w").write(src)
print("patched data_loader.py: SNMVP_EPISODES Subset holdout")
