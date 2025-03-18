import copy
import csv
import time
import numpy as np
import scipy.linalg as LA
import numpy.linalg as la
import math
from matrix import CONST_R, CONST_L, SEED, Subrounte1, finite_consensus, imperfect_consensus
np.random.seed(SEED)


def func(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    return 0.5 * LA.norm(U @ V.T - A) ** 2


def cgrad(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    res = U @ V.T - A
    grad_U = res @ V
    grad_V = res.T @ U
    return np.vstack([grad_U, grad_V])


def safe_division(x, y):
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e4

def record_results(filename, optimizers, skip):
    for i, opt in enumerate(optimizers):
        with open(filename, mode="a") as csv_file:
            file = csv.writer(csv_file, lineterminator="\n")
            file.writerow([f"{opt[0]}"])
            file.writerow(opt[1][::skip])
            file.writerow(["loss"])
            file.writerow([round(math.log10(i), 4) for i in opt[2]][::skip])
            file.writerow(["stepsizes"])
            file.writerow(opt[3][::skip])
            file.writerow(["wallclock time"])
            file.writerow(opt[4][::skip])
            file.writerow([])
            file.writerow([])


def Algorithm1(w=None, data=None, agents=None, iterations=None, skip=None, const_q=None):
    name = "Algorithm1"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    old_eta_dict = {}
    row, col = data[0].shape

    np.random.seed(SEED)
    for i in range(agents):
        param_dict[i] = np.random.randn(row + col, CONST_R)
        eta_dict[i] = CONST_L
        old_eta_dict[i] = CONST_L
        grad_dict[i] = cgrad(param_dict[i], data[i])
        loss_dict[i] = func(param_dict[i], data[i])
        y_dict[i] = copy.deepcopy(grad_dict[i])

    iters = []
    loss_value = []
    average_eta = []
    wallclock = []

    gamma = Subrounte1(w, agents)
    start = time.time()

    for t in range(iterations):
        x = {}
        y = {}
        for i in range(agents):
            x[i] = param_dict[i] - eta_dict[i] * y_dict[i]

        if t % const_q == 0:
            phi = finite_consensus(x, w, agents, gamma)
        else:
            phi = copy.deepcopy(x)

        temp_model = np.zeros((row + col, CONST_R))
        for i in range(agents):
            temp_model += phi[i]

        for i in range(agents):
            new_grad = cgrad(phi[i], data[i])
            y[i] = y_dict[i] + new_grad - grad_dict[i]
            grad_dict[i] = new_grad
            loss_dict[i] = func(temp_model/agents, data[i])

        if t % const_q == 0:
            theta = finite_consensus(y, w, agents, gamma)
        else:
            theta = copy.deepcopy(y)

        new_eta = {}
        for i in range(agents):
            new_eta[i] = min(np.sqrt(1 + eta_dict[i] / old_eta_dict[i]) * eta_dict[i],
                             0.5 * la.norm(phi[i] - param_dict[i]) / la.norm(theta[i] - y_dict[i]))
        old_eta_dict = copy.deepcopy(eta_dict)
        eta_dict = copy.deepcopy(new_eta)

        for i in range(agents):
            param_dict[i] = copy.deepcopy(phi[i])
            y_dict[i] = copy.deepcopy(theta[i])

        iters.append(t)
        loss_value.append(sum([loss for loss in loss_dict.values()]) / agents)
        average_eta.append(sum([eta for eta in eta_dict.values()]) / agents)
        c_t = time.time()
        ttt = c_t - start
        wallclock.append(ttt)
        if t % skip == 0:
            print(
                f"Optimizer: {name}, Iteration: {t}, Loss: {math.log10(loss_value[t]):.4f}, Stepsize: {average_eta[t]:.4f}, Wallclock time: {ttt:.4f}")

    return name, iters, loss_value, average_eta, wallclock

def Algorithm2(w=None, data=None, agents=None, iterations=None, skip=None, k_loop=None):
    name = "Algorithm2"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    old_eta_dict = {}
    row, col = data[0].shape

    np.random.seed(SEED)
    for j in range(agents):
        param_dict[j] = np.random.randn(row + col, CONST_R)
        eta_dict[j] = CONST_L
        old_eta_dict[j] = CONST_L
        grad_dict[j] = cgrad(param_dict[j], data[j])
        loss_dict[j] = func(param_dict[j], data[j])
        y_dict[j] = copy.deepcopy(grad_dict[j])

    iters = []
    loss_value = []
    average_eta = []
    wallclock = []

    start = time.time()

    for t in range(iterations):
        x = {}
        y = {}

        for i in range(agents):
            x[i] = param_dict[i] - eta_dict[i] * y_dict[i]

        phi = imperfect_consensus(x, k_loop, w, agents)

        temp_model = np.zeros((row + col, CONST_R))
        for i in range(agents):
            temp_model += phi[i]

        for i in range(agents):
            new_grad = cgrad(phi[i], data[i])
            y[i] = y_dict[i] + new_grad - grad_dict[i]
            grad_dict[i] = new_grad
            loss_dict[i] = func(temp_model/agents, data[i])

        theta = imperfect_consensus(y, k_loop, w, agents)

        new_eta = {}
        for i in range(agents):
            new_eta[i] = min(np.sqrt(1 + eta_dict[i] / old_eta_dict[i]) * eta_dict[i],
                             0.5 * la.norm(phi[i] - param_dict[i]) / la.norm(theta[i] - y_dict[i]))
        old_eta_dict = copy.deepcopy(eta_dict)
        eta_dict = copy.deepcopy(new_eta)

        for i in range(agents):
            param_dict[i] = copy.deepcopy(phi[i])
            y_dict[i] = copy.deepcopy(theta[i])

        iters.append(t)
        loss_value.append(sum([loss for loss in loss_dict.values()]) / agents)
        average_eta.append(sum([eta for eta in eta_dict.values()]) / agents)
        c_t = time.time()
        ttt = c_t - start
        wallclock.append(ttt)
        if t % skip == 0:
            print(
                f"Optimizer: {name}, Iteration: {t}, Loss: {math.log10(loss_value[t]):.4f},  Stepsize: {average_eta[t]:.4f}, Wallclock time: {ttt:.4f}")

    return name, iters, loss_value, average_eta, wallclock


def DBBC(w=None, data=None, agents=None, iterations=None, skip=None, k_loop=None):
    name = "DGM-BB-C"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    row, col = data[0].shape

    np.random.seed(SEED)
    for j in range(agents):
        param_dict[j] = np.random.randn(row + col, CONST_R)
        eta_dict[j] = CONST_L
        grad_dict[j] = cgrad(param_dict[j], data[j])
        loss_dict[j] = func(param_dict[j], data[j])
        y_dict[j] = copy.deepcopy(grad_dict[j])

    iters = []
    loss_value = []
    average_eta = []
    wallclock = []

    start = time.time()

    for t in range(iterations):
        x = {}
        y = {}
        old_param_dict = copy.deepcopy(param_dict)
        old_grad_dict = copy.deepcopy(grad_dict)

        for i in range(agents):
            x[i] = param_dict[i] - eta_dict[i] * y_dict[i]

        param_dict = imperfect_consensus(x, k_loop, w, agents)

        temp_model = np.zeros((row + col, CONST_R))
        for i in range(agents):
            temp_model += param_dict[i]

        for i in range(agents):
            new_grad = cgrad(param_dict[i], data[i])
            y[i] = y_dict[i] + new_grad - grad_dict[i]
            grad_dict[i] = new_grad
            loss_dict[i] = func(temp_model/agents, data[i])

        y_dict = imperfect_consensus(y, k_loop, w, agents)

        new_eta = {}
        for i in range(agents):
            diff_param = param_dict[i] - old_param_dict[i]
            diff_grad = grad_dict[i] - old_grad_dict[i]
            a = la.norm(np.dot(diff_param.T, diff_param)) / la.norm(np.dot(diff_param.T, diff_grad))
            b = la.norm(np.dot(diff_param.T, diff_grad)) / la.norm(np.dot(diff_grad.T, diff_grad))
            new_eta[i] = min(a, b, 10)
        eta_dict = copy.deepcopy(new_eta)

        iters.append(t)
        loss_value.append(sum([loss for loss in loss_dict.values()]) / agents)
        average_eta.append(sum([eta for eta in eta_dict.values()]) / agents)
        c_t = time.time()
        ttt = c_t - start
        wallclock.append(ttt)
        if t % skip == 0:
            print(
                f"Optimizer: {name}, Iteration: {t}, Loss: {math.log10(loss_value[t]):.4f},  Stepsize: {average_eta[t]:.4f}, Wallclock time: {ttt:.4f}")

    return name, iters, loss_value, average_eta, wallclock


def DGD(w=None, data=None, agents=None, iterations=None, skip=None):
    name = "DGD"
    param_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    row, col = data[0].shape

    np.random.seed(SEED)
    for j in range(agents):
        param_dict[j] = np.random.randn(row + col, CONST_R)
        eta_dict[j] = CONST_L
        grad_dict[j] = cgrad(param_dict[j], data[j])
        loss_dict[j] = func(param_dict[j], data[j])

    iters = []
    loss_value = []
    grad_norms = []
    average_eta = []
    wallclock = []

    start = time.time()

    for t in range(iterations):
        param_dict = imperfect_consensus(copy.deepcopy(param_dict), 1, w, agents)
        for i in range(agents):
            param_dict[i] = param_dict[i] - eta_dict[i] * grad_dict[i]

        temp_model = np.zeros((row + col, CONST_R))
        for i in range(agents):
            temp_model += param_dict[i]

        for i in range(agents):
            new_grad = cgrad(param_dict[i], data[i])
            grad_dict[i] = new_grad
            loss_dict[i] = func(temp_model/agents, data[i])

        iters.append(t)
        grad_norms.append(np.linalg.norm(sum([grad for grad in grad_dict.values()])) / agents)
        loss_value.append(sum([loss for loss in loss_dict.values()]) / agents)
        average_eta.append(sum([eta for eta in eta_dict.values()]) / agents)
        c_t = time.time()
        ttt = c_t - start
        wallclock.append(ttt)
        if t % skip == 0:
            print(
                f"Optimizer: {name}, Iteration: {t}, Loss: {math.log10(loss_value[t]):.4f},  Stepsize: {average_eta[t]:.4f}, Wallclock time: {ttt:.4f}")

    return name, iters, loss_value, average_eta, wallclock



