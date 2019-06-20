import torch
import torch.nn as nn
import torch.nn.functional as F

from snorkel.model.classifier import Classifier
from snorkel.model.utils import MetalDataset, pred_to_prob, recursive_merge_dicts

from .em_defaults import em_default_config
from .identity_module import IdentityModule
from .loss import SoftCrossEntropyLoss


class EndModel(Classifier):
    """A dynamically constructed discriminative classifier

        layer_out_dims: a list of integers corresponding to the output sizes
            of the layers of your network. The first element is the
            dimensionality of the input layer, the last element is the
            dimensionality of the head layer (equal to the cardinality of the
            task), and all other elements dictate the sizes of middle layers.
            The number of middle layers will be inferred from this list.
        input_module: (nn.Module) a module that converts the user-provided
            model inputs to torch.Tensors. Defaults to IdentityModule.
        middle_modules: (nn.Module) a list of modules to execute between the
            input_module and task head. Defaults to nn.Linear.
        head_module: (nn.Module) a module to execute right before the final
            softmax that outputs a prediction for the task.
    """

    def __init__(
        self,
        layer_out_dims,
        input_module=None,
        middle_modules=None,
        head_module=None,
        **kwargs,
    ):

        if len(layer_out_dims) < 2 and not kwargs["skip_head"]:
            raise ValueError(
                "Arg layer_out_dims must have at least two "
                "elements corresponding to the output dim of the input module "
                "and the cardinality of the task. If the input module is the "
                "IdentityModule, then the output dim of the input module will "
                "be equal to the dimensionality of your input data points"
            )

        # Add layer_out_dims to kwargs so it will be merged into the config dict
        kwargs["layer_out_dims"] = layer_out_dims
        config = recursive_merge_dicts(em_default_config, kwargs, misses="insert")
        super().__init__(k=layer_out_dims[-1], config=config)

        self._build(input_module, middle_modules, head_module)

        # Show network
        if self.config["verbose"]:
            print("\nNetwork architecture:")
            self._print()
            print()

    def _build(self, input_module, middle_modules, head_module):
        """
        TBD
        """
        input_layer = self._build_input_layer(input_module)
        middle_layers = self._build_middle_layers(middle_modules)

        # Construct list of layers
        layers = [input_layer]
        if middle_layers is not None:
            layers += middle_layers
        if not self.config["skip_head"]:
            head = self._build_task_head(head_module)
            layers.append(head)

        # Construct network
        if len(layers) > 1:
            self.network = nn.Sequential(*layers)
        else:
            self.network = layers[0]

        # Construct loss module
        loss_weights = self.config["train_config"]["loss_weights"]
        if loss_weights is not None and self.config["verbose"]:
            print(f"Using class weight vector {loss_weights}...")
        reduction = self.config["train_config"]["loss_fn_reduction"]
        self.criteria = SoftCrossEntropyLoss(
            weight=self._to_torch(loss_weights, dtype=torch.FloatTensor),
            reduction=reduction,
        )

    def _build_input_layer(self, input_module):
        if input_module is None:
            input_module = IdentityModule()
        output_dim = self.config["layer_out_dims"][0]
        input_layer = self._make_layer(
            input_module,
            "input",
            self.config["input_layer_config"],
            output_dim=output_dim,
        )
        return input_layer

    def _build_middle_layers(self, middle_modules):
        layer_out_dims = self.config["layer_out_dims"]
        num_mid_layers = len(layer_out_dims) - 2
        if num_mid_layers == 0:
            return None

        middle_layers = nn.ModuleList()
        for i in range(num_mid_layers):
            if middle_modules is None:
                module = nn.Linear(*layer_out_dims[i : i + 2])
                output_dim = layer_out_dims[i + 1]
            else:
                module = middle_modules[i]
                output_dim = None
            layer = self._make_layer(
                module,
                "middle",
                self.config["middle_layer_config"],
                output_dim=output_dim,
            )
            middle_layers.add_module(f"layer{i+1}", layer)
        return middle_layers

    def _build_task_head(self, head_module):
        if head_module is None:
            head = nn.Linear(self.config["layer_out_dims"][-2], self.k)
        else:
            # Note that if head module is provided, it must have input dim of
            # the last middle module and output dim of self.k, the cardinality
            head = head_module
        return head

    def _make_layer(self, module, prefix, layer_config, output_dim=None):
        if isinstance(module, IdentityModule):
            return module
        layer = [module]
        if layer_config[f"{prefix}_relu"]:
            layer.append(nn.ReLU())
        if layer_config[f"{prefix}_batchnorm"] and output_dim:
            layer.append(nn.BatchNorm1d(output_dim))
        if layer_config[f"{prefix}_dropout"]:
            layer.append(nn.Dropout(layer_config[f"{prefix}_dropout"]))
        if len(layer) > 1:
            return nn.Sequential(*layer)
        else:
            return layer[0]

    def _print(self):
        print(self.network)

    def forward(self, x):
        """Returns a list of outputs for tasks 0,...t-1

        Args:
            x: a [batch_size, ...] batch from X
        """
        return self.network(x)

    @staticmethod
    def _reset_module(m):
        """A method for resetting the parameters of any module in the network

        First, handle special cases (unique initialization or none required)
        Next, use built in method if available
        Last, report that no initialization occured to avoid silent failure.

        This will be called on all children of m as well, so do not recurse
        manually.
        """
        if callable(getattr(m, "reset_parameters", None)):
            m.reset_parameters()

    def update_config(self, update_dict):
        """Updates self.config with the values in a given update dictionary"""
        self.config = recursive_merge_dicts(self.config, update_dict)

    def _preprocess_Y(self, Y, k):
        """Convert Y to prob labels if necessary"""
        Y = Y.clone()

        # If preds, convert to probs
        if Y.dim() == 1 or Y.shape[1] == 1:
            Y = pred_to_prob(Y.long(), k=k)
        return Y

    def _create_dataset(self, *data):
        return MetalDataset(*data)

    def _get_loss_fn(self):
        criteria = self.criteria.to(self.config["device"])
        # This self.preprocess_Y allows us to not handle preprocessing
        # in a custom dataloader, but decreases speed a bit
        loss_fn = lambda X, Y: criteria(self.forward(X), self._preprocess_Y(Y, self.k))
        return loss_fn

    def train_model(self, train_data, valid_data=None, log_writer=None, **kwargs):
        self.config = recursive_merge_dicts(self.config, kwargs)

        # If train_data is provided as a tuple (X, Y), we can make sure Y is in
        # the correct format
        # NOTE: Better handling for if train_data is Dataset or DataLoader...?
        if isinstance(train_data, (tuple, list)):
            X, Y = train_data
            Y = self._preprocess_Y(self._to_torch(Y, dtype=torch.FloatTensor), self.k)
            train_data = (X, Y)

        # Convert input data to data loaders
        train_loader = self._create_data_loader(train_data, shuffle=True)

        # Create loss function
        loss_fn = self._get_loss_fn()

        # Execute training procedure
        self._train_model(
            train_loader, loss_fn, valid_data=valid_data, log_writer=log_writer
        )

    def predict_proba(self, X):
        """Returns a [n, k] tensor of probs (probabilistic labels)."""
        return F.softmax(self.forward(X), dim=1).data.cpu().numpy()
