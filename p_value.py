import numpy as np
import numpy as np
from scipy.stats import ttest_rel, wilcoxon,ttest_1samp

ASAP_ari=[0.8789759093850085,0.8886731111270717,0.921614366479547,0.8877366919093754,0.8973971856944121]

t_stat1, p_value_t1 = ttest_1samp(ASAP_ari, 0.8789759093850085)/2

print("Paired t-test p-value:", p_value_t1)
