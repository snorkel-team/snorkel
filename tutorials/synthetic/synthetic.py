#%%
import matplotlib.pyplot as plt
import numpy as np

from snorkel.augmentation.apply import PandasTFApplier
from snorkel.augmentation.policy import RandomAugmentationPolicy
from snorkel.labeling.analysis import lf_empirical_accuracies
from snorkel.labeling.apply import PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel
from snorkel.synthetic.synthetic_data import (
    generate_mog_dataset,
    generate_resampling_tfs,
    generate_single_feature_lfs,
)

#%% [markdown]
# ## Simple Synthetic Data Example
# In this example, we generate a simple synthetic dataset, and then:
#
#   1. Apply labeling functions of various accuracies to it
#   2. Train a `LabelModel` to reweight and combine their outputs
#   3. And finally train an `EndModel` using the generated training labels

#%%
# Generating a simple mixture of gaussians dataset of d-dim vectors
# as a list of SimpleNamespaces
n = 10000
d = 10
n_noise_dim = 2
cov = np.diag(np.random.random(d))
data = generate_mog_dataset(n, d, cov=cov, n_noise_dim=n_noise_dim)

#%%
X_pos = np.array(data[data.y == 1].x.tolist())
X_neg = np.array(data[data.y == 2].x.tolist())

# Plotting the first two dimensions of the data
plt.scatter(X_pos[:, 0], X_pos[:, 1], color="red")
plt.scatter(X_neg[:, 0], X_neg[:, 1], color="blue")
plt.show()


#%%
m = 4
tfs = generate_resampling_tfs(list(range(d, d + n_noise_dim)))
policy = RandomAugmentationPolicy(len(tfs), sequence_length=1)
tf_applier = PandasTFApplier(tfs, policy, k=1, keep_original=True)
data_augmented = tf_applier.apply(data)


#%%
# Generate a set of m LFs that each fire based on a single feature
m = 10
abstain_rate = 0.0
lfs = generate_single_feature_lfs(m, abstain_rate=abstain_rate)

#%%
# Apply the labeling functions to the data
lf_applier = PandasLFApplier(lfs)
L = lf_applier.apply(data_augmented)

#%%
# Run the LabelModel to estimate the LF accuracies
label_model = LabelModel()
label_model.train_model(L)

#%%
# Compare to the true (empirical) accuracies
Y = data_augmented.y.values.astype(int)
accs_emp = lf_empirical_accuracies(L, Y)
accs_est = label_model.get_accuracies()
print(f"Avg. LF accuracy estimation error: {np.mean(np.abs(accs_emp - accs_est))}")

#%%
# Get probabilistic training labels
Y_prob = label_model.predict_proba(L)

# Get the predictive performance of the LabelModel, i.e. the accuracy of the
# hard-thresholded probabilistic labels
label_model.score((L, Y))

#%%
# Run the EndModel to make final predictions
# TODO
