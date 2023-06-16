from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from collections import defaultdict
from typing import Dict, Tuple, Union
import warnings

import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict, \
    ParamData


class MyPluginNAI(SupervisedPlugin):
    """
    Implemented your plugin (if any) here.
    """

    def __init__(self, 
                 alpha,
                 eps = 0.0001,
        ):
        """
        :param
        """
        super().__init__()

        self.alpha = alpha
        self.eps = eps

        self.saved_params = defaultdict(dict)
        self.importances = defaultdict(dict)


    def before_backward(self, strategy, **kwargs):
        """
        Compute NAI penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        for experience in range(exp_counter):
            for k, cur_param in strategy.model.named_parameters():
                # new parameters do not count
                if k not in self.saved_params[experience]:
                    continue
                saved_param = self.saved_params[experience][k]
                imp = self.importances[experience][k]
                new_shape = cur_param.shape
                penalty += (imp.expand(new_shape) *
                            (cur_param -
                                saved_param.expand(new_shape))
                            .pow(2)).sum()
        
        strategy.loss += self.alpha * torch.abs(penalty)

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(
        self, model, criterion, optimizer, dataset, device, batch_size
    ):
        """
        Compute NAI importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        collate_fn = (
            dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                model.named_parameters(), importances.items()
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp.data += p.grad.data.clone()

        # average over mini batch length
        for _, imp in importances.items():
            nominator = 0
            for imp_part in imp.data:
                nominator += (imp_part - (imp.data/float(len(dataloader)))).pow(2)
            sigma = torch.sqrt(nominator/float(len(dataloader)))
            imp.data = (imp.data / float(len(dataloader))) / (sigma + self.eps)

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        self.importances[t] = importances