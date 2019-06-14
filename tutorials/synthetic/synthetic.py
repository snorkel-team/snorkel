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
X_neg = np.random.multivariate_normal(-1*np.ones(d), cov, int(np.ceil(n/2)))

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
from snorkel.labeling.lf.core import labeling_function
from snorkel.types.data import Example

# TODO: Use an LF generator here
@labeling_function()
def lf_0(x: Example) -> int:
    if x[0] > 0:
        return 1
    else:
        return -1

@labeling_function()
def lf_1(x: Example) -> int:
    if x[1] > 0:
        return 1
    else:
        return -1

LFS = [lf_0, lf_1]

#%%
from scipy.sparse import lil_matrix

# TODO: Replace this with an LF applier
m = len(LFS)
L = lil_matrix((n, m))
for i in range(n):
    for j, lf in enumerate(LFS):
        L[i,j] = lf(X[i,:])
L = L.tocsr()

#%%
# Compute the true (empirical) LF accuracies
Ld = np.array(L.todense())
accs = Ld.T @ Y / n

#%%
# Run the LabelModel to estimate the LF accuracies
# TODO