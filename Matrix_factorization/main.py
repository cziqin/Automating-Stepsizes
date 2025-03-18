import argparse
import os
import numpy as np
import pandas as pd
from matrix import cycle_graph
from optimizer import Algorithm1, Algorithm2, DBBC, DGD, record_results

agents = 5
agent_matrix = cycle_graph(agents)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=3, type=int)
    parser.add_argument("-s", "--stratified", default=0, type=int)
    parser.add_argument("-i", "--iterations", type=int, default=150)
    parser.add_argument("-q", "--const_q", default=1, type=int)
    parser.add_argument("-k", "--k_loop", default=1, type=int)
    parser.add_argument("-l", "--skip", default=1, type=int)
    return parser.parse_args()


def load_data(stratified, agent_num=5):
    agent_A = {}
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for n in range(agent_num):
        agent_A[n] = ratings.copy()
    if stratified:
        print("homogeneous_functions")
        for row in df.itertuples():
            ratings[row[1] - 1, row[2] - 1] = row[3]
        for n in range(agent_num):
            agent_A[n] = ratings
    else:
        print("hetergeneous_functions")
        for row in df.itertuples():
            agent_A[int(row[3] - 1)][row[1] - 1, row[2] - 1] = row[3]
    return agent_A


args = parse_args()
iterations = args.iterations
stratified = args.stratified
const_q = args.const_q
k_loop = args.k_loop
skip = args.skip


cwd = os.getcwd()
type_t = "homogeneous" if stratified else "heterogeneous"
A = load_data(stratified, agents)

Algorithm_name = None
if args.test_num == 0:
    Algorithm_name = "Algorithm1"
elif args.test_num == 1:
    Algorithm_name = "Algorithm2"
elif args.test_num == 2:
    Algorithm_name = "DGM-BB_C"
elif args.test_num == 3:
    Algorithm_name = "DGD"

optimizers = None
if args.test_num == 0:
    algorithm1 = Algorithm1(w=agent_matrix, data=A, agents=agents, iterations=iterations, skip=skip,
                            const_q=const_q,
                            )
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithm2 = Algorithm2(w=agent_matrix, data=A, agents=agents, iterations=iterations, skip=skip,
                            k_loop=k_loop,
                            )
    optimizers = [algorithm2]
elif args.test_num == 2:
    dbbc = DBBC(w=agent_matrix, data=A, agents=agents, iterations=iterations, skip=skip,
                k_loop=k_loop,
                )
    optimizers = [dbbc]
elif args.test_num == 3:
    dgd = DGD(w=agent_matrix, data=A, agents=agents, iterations=iterations, skip=skip)
    optimizers = [dgd]

results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{Algorithm_name}_{type_t}_it{iterations}.csv")
record_results(filename, optimizers, skip=skip)
