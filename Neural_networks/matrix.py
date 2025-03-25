from functools import cache
from typing import Optional
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

def cycle_graph(m):
    W = np.zeros((m, m))
    np.fill_diagonal(W, 0.4)

    for i in range(m):
        W[i, (i + 1) % m] = 0.3 
        W[i, (i - 1 + m) % m] = 0.3 

    return W

def fully_connected_graph(num_agents, weight):
    W = np.ones((num_agents, num_agents))
    W *= weight

    return W

def subroutine1(matrix: ndarray):
    agents = len(matrix)
    Psi = {}
    Gamma = {}
    idx_set, d = hash_generate_Ei(hashable_ndarray(matrix))
    d_min= min(d.values())
    for idx in range(agents):
        Psi[idx] = np.zeros((d[idx] * (agents - d[idx] + 1), agents))
    for l in range(agents):
        z = np.zeros((1, agents))
        z[0, l] = 1
        if agents == d_min:
            for i in range(agents):
                Psi[i][0:d_min,l] = np.array([z[0, s] for s in idx_set[i]])
        else:
            for k in range(agents - d_min + 1):
                for i in range(agents):
                    Psi[i][k * d_min:(k + 1) * d_min, l] = np.array([z[0, s] for s in idx_set[i]])
                z = np.dot(z, matrix.T)
    for i in range(agents):
        b = 1 / agents * np.ones((1, agents))
        Gamma[i] = np.linalg.lstsq(Psi[i].T, b.T, rcond=None)[0].T
    return Gamma

@cache
def hash_generate_Ei(hash_matrix: tuple):
    matrix = np.array(hash_matrix)
    idx_set: dict[int, list] = {}
    d: dict[int, int] = {}
    for i in range(len(matrix)):
        idx_set[i] = list(np.argwhere(matrix[i] > 0).flatten())
        d[i] = len(idx_set[i])
    return idx_set, d

def generate_Ei(matrix: ndarray):
    return hash_generate_Ei(hashable_ndarray(matrix))

def hashable_ndarray(arr: ndarray) -> tuple:
    return tuple(map(tuple, arr))

def get_neighbor_value_sum(
        neighbors: list, group_idx: int, var_dict: dict, shape: torch.Size,
        gamma: list[ndarray],
        device: torch.device,
        var_idx: Optional[int] = None,
) -> Tensor:
    """
    calculate $sum_{p=0}^{d_i}Gamma_{i, z}^{p}[x_{j, t}]_{Idx(p)}$
    :param neighbors: store Idx(p)
    :param group_idx: `z` in Gamma_{i, z}^{p}
    :param var_dict: `x_{j, t}`
    :param shape: shape of $x_{j, t}(k)$, to initialize the var for sum
    :param gamma: $Gamma_{i, z}^{p}$
    :param device: device to store the tensor
    :param var_idx: None for all-layer values, int for one specific layer value
    :return:
    """
    _sum = torch.zeros(shape).to(device)
    if var_idx is not None:
        for z, neighbor_index in enumerate(neighbors):
            _sum += gamma[group_idx][z] * var_dict[neighbor_index][var_idx]
        return _sum
    for z, neighbor_index in enumerate(neighbors):
        # * 如无必要，gamma不需要转换为tensor并放入device
        _sum += gamma[group_idx][z] * var_dict[neighbor_index]
    return _sum

def finite_consensus(
        r: dict,
        layer_idx: int,
        matrix: ndarray,
        gamma,
        device: torch.device='cuda' if torch.cuda.is_available() else 'cpu',
        agent_idx: int=0,
) -> Tensor:
    agents = len(matrix)
    res = torch.zeros_like(r[0][layer_idx])
    shape: torch.Size = res.shape
    neighbor_idx, d = generate_Ei(matrix)
    true_d = d[agent_idx] - 1
    gamma_i = [arr.flatten() for arr in np.split(gamma, agents - true_d, axis=1)]

    res = get_neighbor_value_sum(
            neighbor_idx[agent_idx],
            group_idx=0,
            var_dict=r,
            shape=shape,
            gamma=gamma_i,
            device=device,
            var_idx=layer_idx,
        )
    if true_d == agents - 1:
        return res
    temp_len: int = agents - true_d - 1
    temp_curr_res_dict: dict[int, Tensor] = {}
    for n in range(agents):
        temp_curr_res_dict[n] = r[n][layer_idx]
    for q in range(1, temp_len + 1):
        temp_prev_res_dict = temp_curr_res_dict.copy()
        for j in range(agents):
            temp_w: Tensor = torch.Tensor(matrix[j, :]).to(device).view(
                (agents,) + (1,) * (len(shape))
            )
            temp_curr_res_dict[j] = torch.sum(
                torch.stack(list(temp_prev_res_dict.values())) * temp_w,
                dim=0,
            )
        res += get_neighbor_value_sum(
            neighbor_idx[agent_idx],
            group_idx=q,
            var_dict=temp_curr_res_dict,
            shape=shape,
            gamma=gamma_i,
            device=device,
        )
    return res

if __name__ == '__main__':
    test_agents = 5
    test_W: ndarray = cycle_graph(test_agents)
    test_gamma = subroutine1(test_W)
    for _ in range(5):
        _ = subroutine1(test_W)
    pass
