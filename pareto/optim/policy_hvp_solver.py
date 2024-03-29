from functools import partial
from typing import Tuple, List, Iterable, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import parameters_to_vector


__all__ = ['HVPSolver', 'AutogradHVPSolver', 'VisionHVPSolver']


class HVPSolver(object):
    """
    Hessian-Vector product calculation
    network:     PyTorch network to compute hessian for
    parameters:  parameters which are computed hessian w.r.t.
    dataloader:  PyTorch dataloader that we get examples from to compute grads
    device:      gpu/cpu device
    """

    def __init__(
            self,
            network: nn.Module,
            parameters: Iterable[Tensor],
            device: torch.device,
            dataloader: torch.utils.data.DataLoader,
        ) -> None:

        self.parameters = list(parameters)
        self.size = int(sum(p.numel() for p in self.parameters))
        self.network = network
        self.device = device
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataiter = iter(dataloader) if dataloader else None
        self.apply = self.apply_batch
        self.grad = self.grad_batch


    def close(self) -> None:

        try:
            while True:
                _ = next(self.dataiter)
        except StopIteration:
            pass

        self.dataiter = None
        self.dataloader = None


    def set_hess(
            self,
            *,
            batch: bool = True,
            num_batches: int = None,
        ) -> None:

        self.apply = self.apply_batch if batch else partial(self.apply_full, num_batches=num_batches)


    def set_grad(
            self,
            *,
            batch: bool = True,
            num_batches: int = None,
        ) -> None:

        self.grad = self.grad_batch if batch else partial(self.grad_full, num_batches=num_batches)


    @torch.enable_grad()
    def apply_batch(
            self,
            vec: Tensor,
            weights: Tensor = None,
            *,
            grads: Tensor = None,
            retain_graph: bool = True,
        ) -> Tuple[Tensor, Tensor]:

        """
        Returns H * vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """

        raise NotImplementedError


    @torch.enable_grad()
    def apply_full(
            self,
            vec: Tensor,
            weights: Tensor = None,
            *,
            grads: Tensor = None,
            num_batches: int = None,
            retain_graph: bool = False,
        ) -> Tensor:

        apply_batch = self.apply_batch

        num_batches = len(self.dataloader) if num_batches is None else num_batches
        weighted_hvp = None
        for _ in range(num_batches):
            weighted_hvp_batch, _ = apply_batch(
                vec, weights, grads=grads, retain_graph=retain_graph)
            if weighted_hvp is None:
                weighted_hvp = weighted_hvp_batch
            else:
                weighted_hvp.add_(weighted_hvp_batch)
        weighted_hvp.div_(num_batches)
        return weighted_hvp


    def zero_grad(self) -> None:

        """
        Zeros out the gradient info for each parameter in the model
        """

        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()


    def set_data(
            self,
            dataloader: torch.utils.data.DataLoader,
        ) -> None:

        self.dataloader = dataloader
        self.dataiter = iter(dataloader)


    @torch.enable_grad()
    def get_losses(self) -> List[Tensor]:
        raise NotImplementedError


    @torch.enable_grad()
    def grad_batch(
            self,
            *,
            create_graph: bool = True,
        ) -> Tuple[Tensor, List[Tensor]]:

        parameters = self.parameters

        losses = self.get_losses()
        param_grads = [list(torch.autograd.grad(
            loss, parameters,
            allow_unused=True, retain_graph=True, create_graph=create_graph)) for loss in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, parameters)):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
        grads = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)

        return grads, losses


    @torch.enable_grad()
    def grad_full(
            self,
            *,
            create_graph: bool = False,
            num_batches: int = None,
        ) -> Tensor:

        grad_batch = self.grad_batch

        num_batches = len(self.dataloader) if num_batches is None else num_batches
        grads = None
        for _ in range(num_batches):
            grads_batch, _ = grad_batch(create_graph=create_graph)
            if grads is None:
                grads = grads_batch
            else:
                grads.add_(grads_batch)
        grads.div_(num_batches)
        grads = grads.clone().detach()

        return grads


class AutogradHVPSolver(HVPSolver):

    """
    Use PyTorch autograd for Hessian-Vector product calculation
    """

    def get_losses(self) -> List[Tensor]:
        raise NotImplementedError


    @torch.enable_grad()
    def apply_batch(
            self,
            vec: Tensor,
            weights: Tensor = None,
            *,
            grads: Tensor = None,
            retain_graph: bool = True,
        ) -> Tuple[Tensor, Tensor]:

        """
        Returns H * vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """

        if grads is None:
            # compute original gradient, tracking computation graph
            self.zero_grad()
            grads, _ = self.grad_batch(create_graph=True)
            self.zero_grad()

        if weights is None:
            weighted_grad = grads.sum(dim=0)
        else:
            weighted_grad = torch.matmul(weights, grads)

        dot = vec.dot(weighted_grad)
        param_weighted_hvp = torch.autograd.grad(dot, self.parameters, retain_graph=retain_graph) # allow_unused=True

        # concatenate the results over the different components of the network
        weighted_hvp = parameters_to_vector(p.contiguous() for p in param_weighted_hvp)

        # param_weighted_hvp = [p.contiguous() if p is not None else None for p in param_weighted_hvp]
        # filtered_params = [p for p in param_weighted_hvp if p is not None]
        # weighted_hvp = parameters_to_vector(param_weighted_hvp)
        return weighted_hvp, grads


class VisionHVPSolver(AutogradHVPSolver):

    def __init__(
            self,
            s_min,
            s_max,
            y_min,
            y_max,
            network: nn.Module,
            device: torch.device,
            dataloader: torch.utils.data.DataLoader,
            estimator,
            # closures: List[Callable],
            *,
            shared: bool = False,
        ) -> None:

        parameters = network.shared_parameters() if shared else network.parameters()
        super(VisionHVPSolver, self).__init__(network, parameters, device, dataloader)
        # self.closures = closures
        self.s_min = s_min
        self.s_max = s_max
        self.y_min = y_min
        self.y_max = y_max
        self.estimator = estimator
        self.criterion = nn.L1Loss()

    @staticmethod
    def expand(bs, val, device):
        gt = torch.tensor(val)
        gt = gt.unsqueeze(0)
        gt = gt.expand(bs, -1)
        return gt.to(device)

    @torch.enable_grad()
    def get_losses(self) -> List[Tensor]:

        try:
            X, _, _, _ = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            X, _, _, _Y = next(self.dataiter)

        X = X.to(self.device)

        if isinstance(X, list):
            features = [feature.to(self.device) for feature in X]
        else:
            features = X.to(self.device)

        T = self.network(X)
        # inputs = torch.cat([features, T], dim=1)
        # _, hat_s, hat_y = self.estimator(inputs)
        _, hat_s, hat_y = self.estimator(features, T)

        batch_size = len(X)
        gt_s = self.expand(batch_size, self.s_max + 1, self.device)
        gt_y = self.expand(batch_size, self.y_max + 1, self.device)

        loss_s = self.criterion(gt_s, hat_s)
        loss_y = self.criterion(gt_y, hat_y)

        return [loss_s, loss_y]
        # return [loss_s, loss_rep]
