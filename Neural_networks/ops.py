from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, List, Optional, Literal
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
import math
from matrix import subroutine1, finite_consensus


class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.2, name=None, device=None, eps=1e-5, weight_decay=0, stratified=True):

        defaults = dict(idx=idx, lr=lr, w=w, agents=agents, name=name, device=device,
                        eps=eps, weight_decay=weight_decay, stratified=stratified)

        super().__init__(params, defaults)

    @classmethod
    def cls_collect_params_grads(cls, optimizer: Optimizer, independent: bool = False):
        var_s = []
        grads = []
        for group in optimizer.param_groups:
            if independent:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    var_s.append(p.data.clone().detach())
                    grads.append(p.grad.data.clone().detach())
                return var_s, grads
            for p in group['params']:
                if p.grad is None:
                    continue
                var_s.append(p.data)
                grads.append(p.grad.data)
        return var_s, grads
    def collect_params_grads(self, independent: bool = False):
        return self.cls_collect_params_grads(self, independent)

    def collect_lr(self):
        for group in self.param_groups:
            return group["lr"]

    def collect_prev_lr(self):
        for group in self.param_groups:
            return group["prev_lr"]

    @property
    def _device(self) -> torch.device:
        return self.param_groups[0]["device"]

    @property
    def _w(self) -> ndarray:
        return self.param_groups[0]["w"]

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """step method in Optimizer class"""


class Algorithm3(Base):
    gamma: Optional[ndarray] = None
    def __init__(self, *args, **kwargs):
        self.y = kwargs.pop('var_y')
        self.const_q = kwargs.pop('const_q')
        super().__init__(*args, **kwargs)

        Algorithm3.set_gamma(self._w) if Algorithm3.gamma is None else None

    @classmethod
    def set_gamma(cls, w) -> None:
        cls.gamma = subroutine1(w)
        return None

    @classmethod
    def collect_grad(cls, params: Iterable) -> List[Tensor]:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            grads.append(p.grad.data)
        return grads

    def set_y(self, var_y: List[Tensor]) -> None:
        if self.y is not None:
            for p, g in zip(self.y, var_y):
                p.data = g
            return None
        self.y = var_y.copy()
        return None

    def collect_y(self) -> list[Tensor]:
        var_y: list[Tensor] = []
        if self.y is not None:
            for p in self.y:
                var_y.append(p.data.clone().detach())
            return var_y
        raise ValueError("The y is not initialized")

    def collect_x(self) -> list[Tensor]:
        var_x: list[Tensor] = []
        for group in self.param_groups:
            for p in group['params']:
                var_x.append(p.data.clone().detach())
        return var_x

    def collect_prev_track_grad(self) -> list[Tensor]:
        return self.param_groups[0]["prev_track_grad"]

    def step(self, _type: Literal['x', 'y'], *args, **kwargs) -> None:
        var = {}
        k: int = kwargs.pop('k')
        if _type == 'x':
            curr_lr_dict: dict[int, float] = kwargs.pop('lr')
            vars_x: dict[int, list[Tensor]] = kwargs.pop('vars_x')
            vars_y: dict[int, list[Tensor]] = kwargs.pop('vars_y')

            for group in self.param_groups:
                idx = group['idx']
                agents = group["agents"]
                device = group["device"]

                for i in range(agents):
                    var[i] = []

                sub = 0
                for i, p in enumerate(group['params']):
                    if 0 == (k + 1) % self.const_q:
                        for j in range(agents):
                            var[j].append(
                                vars_x[j][i + sub].to(device) - curr_lr_dict[j] * vars_y[j][i + sub].to(device))

                        p.data = finite_consensus(
                            r=var,
                            layer_idx=i + sub,
                            matrix=self._w,
                            gamma=self.gamma[idx],
                            device=device,
                            agent_idx=idx,
                        )
                    else:
                        p.data.add_(vars_y[idx][i + sub].to(device), alpha=-curr_lr_dict[idx])
        elif _type == 'y':
            grads: list[Tensor] = kwargs.pop('grads')
            track_grads: dict[int, list[Tensor]] = kwargs.pop('track_grads')
            prev_track_grads: dict[int, list[Tensor]] = kwargs.pop('prev_track_grads')
            vars_y: dict[int, list[Tensor]] = kwargs.pop('vars_y')

            lr: float = kwargs.pop('lr')
            prev_lr: float = kwargs.pop('prev_lr')
            var_x: list[Tensor] = kwargs.pop('var_x')

            for group in self.param_groups:
                idx = group['idx']
                agents = group["agents"]
                device = group["device"]

                for i in range(agents):
                    var[i] = []

                sub = 0
                b1 = .0
                b2 = .0
                for i, p in enumerate(group['params']):
                    if 0 == (k + 1) % self.const_q:
                        for j in range(agents):
                            var[j].append(vars_y[j][i + sub].to(device)
                                          + track_grads[j][i + sub].to(device)
                                          - prev_track_grads[j][i + sub].to(device)
                                          )
                        self.y[i + sub] = finite_consensus(
                            r=var,
                            layer_idx=i + sub,
                            matrix=self._w,
                            gamma=self.gamma[idx],
                            device=device,
                            agent_idx=idx,
                        )
                    else:
                        self.y[i + sub] = (vars_y[idx][i + sub].to(device) + track_grads[idx][i + sub].to(device)
                                           - prev_track_grads[idx][i + sub].to(device))

                    b1 += (p.data - var_x[i + sub]).norm().item() ** 2
                    b2 += (self.y[i + sub] - grads[i + sub]).norm().item() ** 2
                item_1 = np.sqrt(1 + (lr / prev_lr)) * lr
                item_2 = np.sqrt(b1) / (lr * np.sqrt(b2))
                lr_new = min(item_1, item_2) if k not in (0, 1) else 0.2

                group["lr"] = lr_new
                group["prev_lr"] = lr
                group["prev_track_grad"] = track_grads[idx]
        else:
            raise ValueError(f"Unknown type: {_type}")
        return None

class DSGDN(Base):
    def __init__(self, *args, **kwargs):
        super(DSGDN, self).__init__(*args, **kwargs)

    def collect_u(self):
        for group in self.param_groups:
            return group["u"]

    def step(self, k=None, vars=None, u=None, grads=None, closure=None):
        loss = None
        b = 0.11
        alpha = 0.5

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            u_tilde = {}
            v_tilde = {}
            x_tilde = {}
            u_new = []

            for j in range(agents):
                u_tilde[j], v_tilde[j], x_tilde[j] = [], [], []

            if k == 0:
                for j in range(agents):
                    sub = 0
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            sub -= 1
                            continue
                        u[j].append(torch.zeros(p.data.size()).to(device))

            for j in range(agents):
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    u_tilde[j].append(b * u[j][i + sub].to(device) + grads[j][i + sub].to(device))
                    v_tilde[j].append(b * u_tilde[j][i + sub].to(device) + grads[j][i + sub].to(device))
                    x_tilde[j].append(vars[j][i + sub].to(device) - alpha * v_tilde[j][i + sub].to(device))

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat_x = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat_x += w[idx, j] * (x_tilde[j][i + sub].to(device))
                    summat_u += w[idx, j] * (u_tilde[j][i + sub].to(device))
                p.data = summat_x
                u_new.append(summat_u)

            lr_new = alpha

            group["u"] = u_new
            group["lr"] = lr_new
        return loss

class DADAM(Base):
    def __init__(self, *args, **kwargs):
        super(DADAM, self).__init__(*args, **kwargs)

    def collect_m(self):
        for group in self.param_groups:
            return group["m"]

    def collect_v(self):
        for group in self.param_groups:
            return group["v"]

    def collect_v_hat(self):
        for group in self.param_groups:
            return group["v_hat"]

    def step(self, k=None, vars=None, m=None, v=None, v_hat=None):
        b1 = 0.9
        b2 = 0.99
        b3 = 0.9
        epsilon = 1e-4
        alpha = 0.005

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            m_list = []
            v_list = []
            v_hat_list = []

            if k == 0:
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    m.append(torch.zeros(p.data.size()).to(device))
                    v.append(torch.zeros_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    v_hat.append((epsilon ** 2)* torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))

            lr = 0
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                m_new = b1 * m[i + sub].to(device) + (1 - b1) * p.grad.data
                v_new = b2 * v[i + sub].to(device) + (1 - b2) * torch.mul(p.grad.data, p.grad.data)
                v_hat_new = b3 * v_hat[i + sub].to(device) + (1 - b3) * torch.max(v_hat[i + sub].to(device), v_new)
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i + sub].to(device))

                p.data = summat - alpha * (m_new / torch.sqrt(v_hat_new))
                m_list.append(m_new.clone().detach())
                v_list.append(v_new)
                v_hat_list.append(v_hat_new)
                lr += torch.sqrt(v_hat_new).norm().item() ** 2
            lr_new = alpha / np.sqrt(lr)

            group["m"] = m_list
            group["v"] = v_list
            group["v_hat"] = v_hat_list
            group["lr"] = lr_new
        return None

class DAMSGrad(Base):
    def __init__(self, *args, **kwargs):
        super(DAMSGrad, self).__init__(*args, **kwargs)

    def collect_m(self):
        for group in self.param_groups:
            return group["m"]

    def collect_v(self):
        for group in self.param_groups:
            return group["v"]

    def collect_v_hat(self):
        for group in self.param_groups:
            return group["v_hat"]

    def collect_u_tilde(self):
        for group in self.param_groups:
            return group["u_tilde"]

    def step(self, k=None, vars=None, m=None, v=None, v_hat=None, u_tilde=None):
        b1 = 0.9
        b2 = 0.99
        epsilon = 1e-4
        alpha = 0.1

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            m_list = []
            v_list = []
            v_hat_list = []
            u_tilde_list = []

            if k == 0:
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    m.append(torch.zeros(p.data.size()).to(device))
                    v.append(torch.zeros_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    v_hat.append((epsilon ** 2) * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    for j in range(agents):
                        u_tilde[j].append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))

            lr = 0
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                m_new = b1 * m[i + sub].to(device) + (1 - b1) * p.grad.data
                v_new = b2 * v[i + sub].to(device) + (1 - b2) * torch.mul(p.grad.data, p.grad.data)
                v_hat_new = torch.max(v_hat[i + sub].to(device), v_new)
                summat_x = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros_like(torch.mul(p.grad.data, p.grad.data))
                for j in range(agents):
                    summat_x += w[idx, j] * vars[j][i + sub].to(device)
                    summat_u += w[idx, j] * u_tilde[j][i + sub].to(device)
                x_temp = summat_x
                u_temp = summat_u
                u_new = torch.max(u_temp,
                                  epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)))

                p.data = x_temp - alpha * (m_new / torch.sqrt(u_new))
                u_tilde_list.append(u_temp - v_hat[i + sub].to(device) + v_hat_new)
                m_list.append(m_new.clone().detach())
                v_list.append(v_new.clone().detach())
                v_hat_list.append(v_hat_new.clone().detach())
                lr += torch.sqrt(u_new).norm().item() ** 2
            lr_new = alpha / np.sqrt(lr)

            group["u_tilde"] = u_tilde_list
            group["m"] = m_list
            group["v"] = v_list
            group["v_hat"] = v_hat_list
            group["lr"] = lr_new
        return None

class ATCDIGing(Base):
    def __init__(self, *args, **kwargs):
        self.y = kwargs.pop('var_y')
        super().__init__(*args, **kwargs)

    @classmethod
    def collect_grad(cls, params: Iterable) -> List[Tensor]:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            grads.append(p.grad.data)
        return grads

    def set_y(self, var_y: List[Tensor]) -> None:
        if self.y is not None:
            for p, g in zip(self.y, var_y):
                p.data = g
            return None
        self.y = var_y.copy()
        return None

    def collect_y(self) -> list[Tensor]:
        var_y: list[Tensor] = []
        if self.y is not None:
            for p in self.y:
                var_y.append(p.data.clone().detach())
            return var_y
        raise ValueError("The y is not initialized")

    def collect_x(self) -> list[Tensor]:
        var_x: list[Tensor] = []
        for group in self.param_groups:
            for p in group['params']:
                var_x.append(p.data.clone().detach())
        return var_x

    def collect_prev_track_grad(self) -> list[Tensor]:
        return self.param_groups[0]["prev_track_grad"]

    def step(self, _type: Literal['x', 'y'], *args, **kwargs) -> None:
        var = {}
        if _type == 'x':
            lr = 0.02
            vars_x: dict[int, list[Tensor]] = kwargs.pop('vars_x')
            vars_y: dict[int, list[Tensor]] = kwargs.pop('vars_y')

            for group in self.param_groups:
                idx = group['idx']
                agents = group["agents"]
                device = group["device"]
                w = group["w"]

                for i in range(agents):
                    var[i] = []

                sub = 0
                for i, p in enumerate(group['params']):
                    for j in range(agents):
                        var[j].append(vars_x[j][i + sub].to(device) - lr * vars_y[j][i + sub].to(device))

                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * var[j][i + sub].to(device)
                    p.data = summat_x

        elif _type == 'y':
            track_grads: dict[int, list[Tensor]] = kwargs.pop('track_grads')
            prev_track_grads: dict[int, list[Tensor]] = kwargs.pop('prev_track_grads')
            vars_y: dict[int, list[Tensor]] = kwargs.pop('vars_y')

            for group in self.param_groups:
                idx = group['idx']
                agents = group["agents"]
                device = group["device"]
                w= group["w"]

                for i in range(agents):
                    var[i] = []

                sub = 0
                for i, p in enumerate(group['params']):
                    for j in range(agents):
                        var[j].append(vars_y[j][i + sub].to(device)
                                      + track_grads[j][i + sub].to(device)
                                      - prev_track_grads[j][i + sub].to(device)
                                      )
                    summat_y = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_y += w[idx, j] * var[j][i + sub].to(device)
                    self.y[i + sub] = summat_y

                group["prev_track_grad"] = track_grads[idx]
        else:
            raise ValueError(f"Unknown type: {_type}")
        return None

class DSGD(Base):
    def __init__(self, *args, **kwargs):
        super(DSGD, self).__init__(*args, **kwargs)

    def step(self, k=None, vars=None, grads=None):
        lr_new = 0.02 / math.sqrt(k + 1)
        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat_x = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat_x += w[idx, j] * vars[j][i + sub].to(device)
                p.data = summat_x - lr_new * grads[idx][i+sub].to(device)

            group["lr"] = lr_new

        return None