import time
import numpy as np
import numpy.linalg as la
import swanlab
import csv
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from loss_function import logistic_smoothness, logistic_loss, logistic_gradient


def load_data(data_path, agents, stratified):
    """Loads data from the specified path and evenly splits it among multiple agents."""
    agent_samples = {}
    agent_targets = {}
    agent_L = {}
    agent_L2 = {}

    data = load_svmlight_file(data_path)
    total_samples, total_targets = data[0].toarray(), data[1]
    if (np.unique(total_targets) == [1, 2]).all():
        total_targets -= 1

    total_samples, total_targets = shuffle(total_samples, total_targets)
    length = int(len(total_samples))
    agent_sample_len, remainder = int(
        length // agents), length % agents

    if stratified:
        print("homogeneous functions")
        for i in range(agents):
            agent_samples[i] = total_samples[i * agent_sample_len:i * agent_sample_len + agent_sample_len]
            agent_targets[i] = total_targets[i * agent_sample_len:i * agent_sample_len + agent_sample_len]
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (10 * agent_samples[i].shape[0])

    else:
        print("heterogeneous functions")
        targets_0 = np.array(list(map(lambda x: x, np.where(total_targets == 0))))[0]
        targets_1 = np.array(list(map(lambda x: x, np.where(total_targets == 1))))[0]

        agent_samples[0] = total_samples[targets_0[:agent_sample_len]]
        agent_targets[0] = total_targets[targets_0[:agent_sample_len]]
        agent_samples[1] = total_samples[targets_0[agent_sample_len + 1:agent_sample_len * 2 + 1]]
        agent_targets[1] = total_targets[targets_0[agent_sample_len + 1:agent_sample_len * 2 + 1]]
        targets_0 = np.delete(targets_0, [range(1, agent_sample_len * 2 + 1)])

        remaining_samples = total_samples[np.append(targets_0, targets_1)]
        remaining_targets = total_targets[np.append(targets_0, targets_1)]

        agent_samples[2] = remaining_samples[:agent_sample_len]
        agent_targets[2] = remaining_targets[:agent_sample_len]
        agent_samples[3] = remaining_samples[agent_sample_len + 1:agent_sample_len * 2 + 1]
        agent_targets[3] = remaining_targets[agent_sample_len + 1:agent_sample_len * 2 + 1]
        agent_samples[4] = remaining_samples[agent_sample_len * 2 + 1:agent_sample_len * 3 + 1]
        agent_targets[4] = remaining_targets[agent_sample_len * 2 + 1:agent_sample_len * 3 + 1]
        for i in range(agents):
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (10 * agent_samples[i].shape[0])

    return agent_samples, agent_targets, agent_L, agent_L2



class Trainer:
    def __init__(self, data, agent_matrix, iterations, interval, min_allow=0,
                 const_q=None, k_loop=None,
                 ):
        self.agent_samples = data[0]
        self.agent_targets = data[1]
        self.agent_L = data[2]
        self.agent_L2 = data[3]
        self.agent_matrix = agent_matrix
        self.min_allow = min_allow
        self.interval = interval
        self.const_q = const_q
        self.k_loop = k_loop

        self.iterations_tot = iterations

        self.agents = agent_matrix.shape[0]
        self.grad_norm = []
        self.eta = {}
        self.eta_record = {}
        self.final_losses_plotted = []
        self.final_grad_plotted = []
        self.losses = []
        self.loss_time=[]
        self.wallclock_time = []
        self.interval_time = []
        self.iterations = []
        self.grads = {}
        self.et = []

        """ initialization """
        self.init_run()
        self.agent_parameters = {}
        for i in range(self.agents):
            self.agent_parameters[i] = [np.zeros(self.agent_samples[i].shape[1])]

    def compute_loss(self,iteration):
        temp_loss = 0
        temp_model = np.zeros(self.agent_samples[0].shape[1])
        for i in range(self.agents):
            temp_model += self.agent_parameters[i][iteration] / self.agents
        for i in range(self.agents):
            temp_loss += logistic_loss(temp_model, self.agent_samples[i],
                                       self.agent_targets[i], self.agent_L2[i])
        if iteration == 0:
            self.losses = [temp_loss / self.agents]
        else:
            self.losses.append(temp_loss / self.agents)

    def compute_grad(self, agent_num, iterations):
        params = self.agent_parameters[agent_num][iterations]

        samples = self.agent_samples[agent_num]
        targets = self.agent_targets[agent_num]
        l2 = self.agent_L2[agent_num]
        if iterations == 0:
            self.grads[agent_num] = [logistic_gradient(params, samples, targets, l2)]
            return self.grads[agent_num]
        else:
            self.grads[agent_num].append(logistic_gradient(params, samples, targets, l2))
        return la.norm(self.grads[agent_num][-1]) > self.min_allow

    def save_data(self, filename, skip):
        with open(filename, mode="a") as csv_file:
            file = csv.writer(csv_file, lineterminator="\n")
            file.writerow([self.name])
            file.writerow(self.iterations[::skip])
            file.writerow(["Loss"])
            file.writerow(self.losses[::skip])
            file.writerow(["Stepsizes"])
            file.writerow(self.et[::skip])
            file.writerow(["Wallclock time"])
            file.writerow(self.wallclock_time[::skip])
            file.writerow([])
            file.writerow([])

    def train(self):
        start = time.time()
        for t in range(self.iterations_tot):
            self.compute_loss(iteration=t)
            current_time = time.time()
            elapsed_time = current_time - start
            self.wallclock_time.append(elapsed_time)
            for j in range(self.agents):
                self.compute_grad(agent_num=j, iterations=t)
            self.grad_norm.append(np.linalg.norm(sum([grad[-1] for grad in self.grads.values()]) / self.agents))
            self.iterations.append(t)
            self.step(t)

            self.et.append(sum([et[-1] for et in self.eta.values()]) / self.agents)

            if t % self.interval == 0:
                print(
                    f"Optimizer: {self.name}, Iteration: {self.iterations[t]}, Loss: {self.losses[-1]},"
                    f"Stepsize: {self.et[t]}, Wallclock time: {self.wallclock_time[-1]}")

