#%%
import matplotlib.pyplot as plt
import numpy as np

from snorkel.labeling.apply import LFApplier
from snorkel.synthetic.synthetic_data import (
    generate_mog_dataset,
    generate_single_feature_LFs,
)

#%% [markdown]
# ## Simple Synthetic Data Example
# In this example, we generate a simple synthetic dataset, and then:
#
#   1. Apply labeling functions of various accuracies to it
#   2. Train a `LabelModel` to reweight and combine their outputs
#   3. And finally train an `EndModel` using the generated training labels

#%%
# Generating a simple mixture of gaussians dataset of d-dim one-hot vectors
# as a pandas DataFrame
n = 1000
d = 10
cov = np.diag(np.random.random(d))
cov[0, 0] = 0.0
data = generate_mog_dataset(n, d, cov=cov)

#%%
X_pos = np.vstack([d.x for d in data if d.y == 1])
X_neg = np.vstack([d.x for d in data if d.y == -1])

# Plotting the first two dimensions of the data
plt.scatter(X_pos[:, 0], X_pos[:, 1], color="red")
plt.scatter(X_neg[:, 0], X_neg[:, 1], color="blue")
plt.show()

#%%
# Generate a set of m LFs that each fire based on a single feature
m = 5
abstain_rate = 0.0
LFs = generate_single_feature_LFs(m, abstain_rate=abstain_rate)

#%%
# Apply the labeling functions to the data
lf_applier = LFApplier(LFs)
L = lf_applier.apply(data)

#%%
# Compute the true (empirical) LF accuracies
Ld = np.array(L.todense())
Y = np.array([d.y for d in data])
accs = Ld.T @ Y / n
accs

#%%
# Run the LabelModel to estimate the LF accuracies
# TODO

#%%
# Run the EndModel to make final predictions
# TODO
