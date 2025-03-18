import argparse
from matrix import cycle_graph
from train import  Algorithm3Trainer, DSGDNTrainer, DADAMTrainer, DAMSGradTrainer, ATCDIGingTrainer, DSGDTrainer, np, torch, os, random
from numpy import ndarray


def main():

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--test_num", default=0, type=int)
        parser.add_argument("-e", "--epochs", default=70, type=int)
        parser.add_argument("-b", "--batch_size", default=128, type=int)
        parser.add_argument("-a", "--agents", default=5, type=int)
        parser.add_argument("-d", "--dataset", default="imagenet")
        parser.add_argument("-s", "--stratified", default=False)
        parser.add_argument("-q", "--const_q", default=1, type=int)
        parser.add_argument("-i", "--interval", default=95, type=int)
        parser.add_argument("-o", "--seed", default=42, type=int)
        return parser.parse_args()

    args = parse_args()
    cwd = os.getcwd()
    results_path = os.path.join(cwd, "results")
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    agents = args.agents
    w: ndarray = cycle_graph(agents)

    dataset = args.dataset
    epochs = args.epochs
    bs = args.batch_size
    stratified = args.stratified
    const_q = args.const_q
    log_interval = args.interval
    seed = args.seed

    fname = os.path.join(results_path, f"{args.test_num}_{dataset}_seed{seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.test_num == 0:
        Algorithm3Trainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                          agents=agents,
                          const_q=const_q,
                          log_interval=log_interval,
                          seed=seed,
                          )
    elif args.test_num == 1:
        DADAMTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                     agents=agents,
                     log_interval=log_interval,
                     seed=seed,
                     )
    elif args.test_num == 2:
        DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                        agents=agents,
                        log_interval=log_interval,
                        seed=seed,
                        )
    elif args.test_num == 3:
        DSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                     agents=agents,
                     log_interval=log_interval,
                     seed=seed,
                     )
    elif args.test_num == 4:
        ATCDIGingTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                         agents=agents,
                         log_interval=log_interval,
                         seed=seed,
                         )
    elif args.test_num == 5:
        DSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified,
                    agents=agents,
                    log_interval=log_interval,
                    seed=seed,
                    )

if __name__ == "__main__":
    main()
