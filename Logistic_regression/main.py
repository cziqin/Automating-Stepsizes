import argparse
import os
from optimizers import *
from matrix import *
from train import load_data
import numpy as np

np.random.seed(1010)

agents = 5
dataset = "mushrooms"

agent_matrix = cycle_graph(agents)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", type=int, default=0)
    parser.add_argument("-s", "--stratified", default=0, type=int)
    parser.add_argument("-i", "--iterations", type=int, default=150)
    parser.add_argument("-l", "--interval", type=int, default=1)
    parser.add_argument("-q", "--const_q", default=1, type=int)
    parser.add_argument("-k", "--k_loop", default=1, type=int)

    return parser.parse_args()

args = parse_args()
iterations = args.iterations
stratified = args.stratified
interval = args.interval
const_q = args.const_q
k_loop = args.k_loop

cwd = os.getcwd()
data_path = os.path.join(cwd, "mushrooms")
data = load_data(data_path=data_path, agents=agents, stratified=stratified)

Algorithm_name = None
if args.test_num == 0:
    Algorithm_name = "Algorithm1"
elif args.test_num == 1:
    Algorithm_name = "Algorithm2"
elif args.test_num == 2:
    Algorithm_name = "DBBG"
elif args.test_num == 3:
    Algorithm_name = "DGD"


optimizers = None
if args.test_num == 0:
    algorithm1 = Algorithm1(agent_matrix=agent_matrix, iterations=iterations, data=data, interval=interval,
                            const_q=const_q, k_loop=k_loop,
                            )
    algorithm1.train()
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithm2 = Algorithm2(agent_matrix=agent_matrix, iterations=iterations, data=data, interval=interval,
                            const_q=const_q, k_loop=k_loop,
                            )
    algorithm2.train()
    optimizers = [algorithm2]
elif args.test_num == 2:
    dbbg = DBBG(agent_matrix=agent_matrix, iterations=iterations, data=data, interval=interval,
                const_q=const_q, k_loop=k_loop,
                )
    dbbg.train()
    optimizers = [dbbg]
elif args.test_num == 3:
    dgd = DGD(agent_matrix=agent_matrix, iterations=iterations, data=data, interval=interval,
              const_q=const_q, k_loop=k_loop,
              )
    dgd.train()
    optimizers = [dgd]


results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{Algorithm_name}.csv")
for opt in optimizers:
    opt.save_data(filename, skip=interval)
