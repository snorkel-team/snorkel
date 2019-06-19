#%% [markdown]
# ## Simple Synthetic Data Example
# In this example, we generate a simple synthetic dataset, and then:
#
#   1. Apply labeling functions of various accuracies to it
#   2. Train a `LabelModel` to reweight and combine their outputs
#   3. And finally train an `EndModel` using the generated training labels 

#%%
import numpy as np
import matplotlib.pyplot as plt

# TODO: Turn this into a utility function
# Generate d-dim data from two Gaussians
n = 1000
d = 10
cov = np.diag(np.random.random(d))
X_pos = np.random.multivariate_normal(np.ones(d), cov, int(np.floor(n/2)))
X_neg = np.random.multivariate_normal(-1 * np.ones(d), cov, int(np.ceil(n/2)))

# Combine and shuffle
X = np.vstack([X_pos, X_neg])
Y = np.concatenate([
    np.ones(int(np.floor(n/2))),
    -1 * np.ones(int(np.ceil(n/2)))
])
order = list(range(n))
np.random.shuffle(order)
X = X[order]
Y = Y[order]

#%%
# Plotting the first two dimensions of the data
plt.scatter(X_pos[:,0], X_pos[:,1], color='red')
plt.scatter(X_neg[:,0], X_neg[:,1], color='blue')
plt.show()

#%%
from snorkel.labeling.lf import LabelingFunction
from functools import partial

def lf_template(x, index=0):
    if x[index] > 0:
        return 1
    else:
        return -1

# Generate a set of m LFs that each label based on one feature of the data
m = int(d/2)
LFs = []
for i in range(m):
    LFs.append(LabelingFunction(f"LF_feat_{i}", partial(lf_template, index=i)))

#%%
from snorkel.labeling.apply import LFApplier

# Apply the labeling functions to the data
lf_applier = LFApplier(LFs)
L = lf_applier.apply(X)

#%%
# Compute the true (empirical) LF accuracies
Ld = np.array(L.todense())
accs = Ld.T @ Y / n

#%%
# Run the LabelModel to estimate the LF accuracies
# TODO
