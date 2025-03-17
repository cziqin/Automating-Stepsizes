import numpy as np


def cycle_graph(m):
    W = np.zeros((m, m))
    np.fill_diagonal(W, 0.4)
    for i in range(m):
        W[i, (i + 1) % m] = 0.3
        W[i, (i - 1 + m) % m] = 0.3
    return W



def fully_connected_graph(num_agents):
    W = np.ones((num_agents, num_agents))
    W *= 1/ num_agents

    return W

def Subrounte1(matrix,agents):
    Psi = {}
    Gamma = {}
    idxset, d = generate_Ei(matrix,agents)
    d_min= min(d.values())
    for idx in range(agents):
        Psi[idx] = np.zeros((d[idx] * (agents - d[idx] + 1), agents))
    for l in range(agents):
        z = np.zeros((1, agents))
        z[0, l] = 1
        if agents == d_min:
            for i in range(agents):
                Psi[i][0:d_min,l] = np.array([z[0, s] for s in idxset[i]])
        else:
            for k in range(agents - d_min + 1):
                for i in range(agents):
                    Psi[i][k * d_min:(k + 1) * d_min, l] = np.array([z[0, s] for s in idxset[i]])
                z = np.dot(z, matrix.T)
    for i in range(agents):
        b = 1 / agents * np.ones((1, agents))
        Gamma[i] = np.linalg.lstsq(Psi[i].T, b.T, rcond=None)[0].T
    return Gamma

def generate_Ei(matrix,agents):
    idxset = {}
    d = {}
    for i in range(agents):
        idxset[i] = []
        for j in range(agents):
            if matrix[i,j] != 0:
                idxset[i].append(j)
        d[i]= len(idxset[i])
    return idxset, d

def finite_consensus(r, matrix, agents,Gamma):
    phi = {i: [] for i in range(agents)}
    Phi = {}
    degree_set, d = generate_Ei(matrix, agents)
    d_min = min(d.values())
    for i in range(agents):
        phi[i].append(np.zeros((1, r[i].shape[0])))
        for p, p1 in enumerate(degree_set[i]):
            phi[i][-1] += Gamma[i][0][p] * r[p1]
    if agents == d_min:
        pass
    else:
        for k in range(agents - d_min):
            r = np.dot(matrix, r)
            for i in range(agents):
                phi[i].append(phi[i][-1])
                for p, p1 in enumerate(degree_set[i]):
                    phi[i][-1] += Gamma[i][0][(k+1) * d_min + p] * r[p1]
    for i in range(agents):
        Phi[i] = phi[i][agents - d_min].reshape(-1)
    return Phi

def imperfect_consensus(x, K, matrix, agents):
    x_k = {}
    for i in range(agents):
        x_k[i] = []
        x_k[i].append(x[i])
    for k in range(K):
        for i in range(agents):
            sum_x = np.zeros_like(x_k[i][k])
            for j in range(agents):
                sum_x += matrix[i,j] * x_k[j][k]
            x_k[i].append(sum_x)
    for i in range(agents):
        x[i] = x_k[i][K]
    return x
