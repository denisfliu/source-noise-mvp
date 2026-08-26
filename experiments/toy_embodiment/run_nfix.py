import oat_complexity as oc
oc.N = 100
oc.NOBS = [1, 2, 3]
oc.BODIES_B = ["point", "arm4"]
oc.SEEDS = [0, 1, 2]
oc.OUT = oc.OUT.replace("oat_complexity", "oat_complexity_n100")
import os; os.makedirs(oc.OUT, exist_ok=True)
oc.main()
