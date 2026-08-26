import sys
PATH = sys.argv[1]
src = open(PATH).read()
if "SNMVP_TASK_ID" in src:
    print("already patched"); sys.exit(0)
old = "    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):\n"
new = ("    import os as _os\n"
       "    _only = _os.environ.get(\"SNMVP_TASK_ID\")\n"
       "    _ids = [int(_only)] if _only is not None else list(range(task_suite.n_tasks))\n"
       "    for task_id in tqdm.tqdm(_ids):\n")
assert src.count(old) == 1, "task loop anchor not found once"
open(PATH, "w").write(src.replace(old, new))
print("patched client: SNMVP_TASK_ID single-task filter")
