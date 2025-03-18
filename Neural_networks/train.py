import os
import csv
import random
from abc import abstractmethod
import warnings
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch import Tensor
from torch.nn import Module
import resnet
import threading
from models import CIFAR10CNN
from ops import Algorithm3, DADAM, DAMSGrad, DSGDN, ATCDIGing, DSGD, Optimizer
from matrix import generate_Ei
from collections import defaultdict
from collections.abc import Generator

warnings.filterwarnings("ignore")

def model_copy(model):
    copy_model = type(model)()
    copy_model.load_state_dict(model.state_dict())
    return copy_model

def average_model(agent_models: dict[int, Module]) -> Module:
    if agent_models is None:
        raise ValueError("Agent models dict are None")
    averaged_model = type(agent_models[0])()
    temp_state_dict = averaged_model.state_dict()

    for key in temp_state_dict.keys():
        if any(k in key for k in ['weight', 'bias', 'running_mean', 'running_var']):
            stacked = torch.stack([am.state_dict()[key] for am in agent_models.values()], dim=0)
            temp_state_dict[key] = torch.mean(stacked, dim=0)
    averaged_model.load_state_dict(temp_state_dict)

    return averaged_model

class DTrainer:
    def __init__(self,
                 dataset="cifar10",
                 epochs=20,
                 batch_size=128,
                 lr=0.02,
                 agents=5,
                 w=None,
                 fname=None,
                 stratified=True,
                 log_interval=14,
                 seed=None,
                 ):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.agents = agents
        self.f_name = fname
        self.stratified = stratified
        self.log_interval = log_interval
        self.seed = seed
        self.w = w
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_iterations = []
        self.train_accuracy = []
        self.test_accuracy_top1 = []
        self.test_accuracy_top5 = []
        self.train_loss = []
        self.test_loss = []
        self.lr_logs = []
        self.running_iteration = 0
        self.tol_train_loss: float = .0

        self.track_grads = {}

        self.load_data()
        self.agent_setup()

    def _log(self, accuracy):
        """ Helper function to log accuracy values"""
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        self.f_name = f"{self.f_name}.csv"

        with open(self.f_name, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator='\n')
            file.writerow([f"{self.opt_name}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(["train_accuracy"])
            file.writerow(self.train_accuracy)
            file.writerow(["test_accuracy_top1"])
            file.writerow(self.test_accuracy_top1)
            file.writerow(["test_accuracy_top5"])
            file.writerow(self.test_accuracy_top5)
            file.writerow(["train_loss"])
            file.writerow(self.train_loss)
            file.writerow(["test_loss"])
            file.writerow(self.test_loss)
            file.writerow(["Stepsize"])
            file.writerow(self.lr_logs)
            file.writerow([])

        torch.save(self.agent_models[0].state_dict(), f"{self.f_name[:-4]}.pt")

    def homogeneous_distribution(self, trainset, testset):
        print("homogeneous data distribution")
        dataset_size = len(trainset)

        split_sizes = [dataset_size // self.agents] * self.agents
        remainder = dataset_size % self.agents
        for i in range(remainder):
            split_sizes[i] += 1

        indices = np.random.permutation(dataset_size)

        start_idx = 0
        for i in range(self.agents):
            subset_indices = indices[start_idx:start_idx + split_sizes[i]]
            subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
            self.train_loader[i] = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=subset_sampler,
                                                   pin_memory=True, num_workers=2)
            start_idx += split_sizes[i]

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                       pin_memory=True, num_workers=3)

    def heterogeneous_distribution(self, train_set, test_set):
        def block_idx_gen(start: int = 0, step: int = 1, bound: int = 1000):
            index = 1
            bound_index = bound // step
            while start < bound:
                start = index * step
                yield start - step, (start if index < bound_index else bound)
                index += 1

        print("Heterogeneous data distribution")
        indices_per_class: defaultdict = defaultdict(list)
        for idx, label in enumerate(train_set.targets):
            indices_per_class[label].append(idx)

        agent_indices: defaultdict = defaultdict(list)

        cls_num = len(train_set.classes)
        full_cls_set: set = set(range(cls_num))
        block_size = cls_num // self.agents
        gen: Generator[tuple[int, ...], None, None] = block_idx_gen(step=block_size, bound=cls_num)
        primary_percentage: float = 0.4

        for agent in range(self.agents):
            primary_classes = next(gen)
            for cls in full_cls_set:
                np.random.shuffle(indices_per_class[cls])
                cls_len = len(indices_per_class[cls])
                rate = primary_percentage if cls in primary_classes else 1 - primary_percentage
                agent_indices[agent].extend(indices_per_class[cls][:int(rate * cls_len)])

        for i in range(self.agents):
            temp_train = torch.utils.data.Subset(train_set, agent_indices[i])
            self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True,
                                                               pin_memory=True, num_workers=3, drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                                       pin_memory=True, num_workers=3, drop_last=True)


    def load_data(self):
        print("==> Loading Data")
        self.train_loader = {}
        self.test_loader = {}

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)), ])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)), ])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        elif self.dataset == "imagenet":
            transform_train = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.class_num = 1000
            trainset = datasets.ImageFolder('./data/imagenet/train', transform_train)
            testset = datasets.ImageFolder('./data/imagenet/val', transform_test)

        if self.stratified:
            self.homogeneous_distribution(trainset, testset)
        else:
            self.heterogeneous_distribution(trainset, testset)

    def agent_setup(self):
        self.agent_models = {}
        self.test_agent_models = {}
        self.agent_optimizers = {}

        self.prev_agent_models = {}
        self.prev_agent_optimizers: dict[int, Optimizer] = {}
        model = None
        model1 = None

        if self.dataset == 'cifar10':
            model = CIFAR10CNN()
            model1= CIFAR10CNN()
        elif self.dataset == "imagenet":
            model = resnet.resnet1()
            model1=resnet.resnet1()

        for i in range(self.agents):
            if i == 0:
                self.agent_models[0] = model
                self.test_agent_models[0] = model1
            else:
                self.agent_models[i] = model_copy(self.agent_models[0])
                self.test_agent_models[i] = model_copy(self.test_agent_models[0])

            self.agent_models[i].to(self.device)
            self.test_agent_models[i].to(self.device)
            self.agent_models[i].train()

            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                name=self.opt_name,
                device=self.device,
                stratified=self.stratified
            )

    def eval(self, dataloader, agent_models):
        total_top1_acc, total_top5_acc, total_count = 0, 0, 0
        tot_t_loss = 0
        device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

        averaged_model = average_model(agent_models)
        averaged_model.eval()
        averaged_model.to(self.device)

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                predicted_label = averaged_model(inputs)
                
                tot_t_loss += self.criterion(predicted_label, labels).item() / len(dataloader)
                total_top1_acc += (predicted_label.argmax(1) == labels).sum().item()

                _, top5_pred = predicted_label.topk(5, dim=1)
                correct_top5 = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))
                total_top5_acc += correct_top5.sum().item()

                total_count += labels.size(0)

        avg_top1_acc = total_top1_acc / total_count
        avg_top5_acc = total_top5_acc / total_count
        self.test_accuracy_top1.append(avg_top1_acc)
        self.test_accuracy_top5.append(avg_top5_acc)

        return avg_top1_acc, avg_top5_acc, tot_t_loss * self.agents

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss, agent_optimizers,
                  running_iteration, agent_models,
                  ):
        self._log(total_acc / total_count)
        t1_acc, t5_acc, tot_t_loss = self.eval(self.test_loader, agent_models)

        self.train_loss.append(tot_loss / (self.agents * log_interval))
        self.test_loss.append(tot_t_loss / self.agents)

        total_sum_step = 0
        for i in range(self.agents):
            total_sum_step += agent_optimizers[i].collect_lr()
        average_step = total_sum_step / self.agents
        self.lr_logs.append(average_step)

        print(
            f"Epoch: {epoch + 1}, Iteration: {running_iteration}, " +
            f"Training loss: {tot_loss / (self.agents * log_interval):.4f}, " +
            f"Training accuracy: {total_acc / total_count:.4f}, " +
            f"Test loss: {tot_t_loss / self.agents :.4f}, " +
            f"Test accuracy top1: {t1_acc:.4f}, " +
            f"Test accuracy top5: {t5_acc:.4f}, " +
            f"Stepsize: {average_step:.4f}, "
        )


    def trainer(self):
        print(
            f"==> Starting Training for {self.opt_name}, {self.epochs} epochs, and {self.agents} agents on the {self.dataset} dataset via {self.device}")

        for i in range(self.epochs):
            self.epoch_iterations(i)

    @abstractmethod
    def epoch_iterations(self, epoch):
        pass

class Algorithm3Trainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = Algorithm3
        self.opt_name = "Algorithm3"
        _, self.d_dict = generate_Ei(kwargs['w'])
        self.const_q = kwargs.pop('const_q')
        super().__init__(*args, **kwargs)

        self.trainer()
        self._save()

    def agent_setup(self):
        self.agent_models: dict = {}
        self.test_agent_models: dict = {}
        self.agent_optimizers: dict = {}
        self.temp_models: dict = {}

        self.y_dict: dict = {}
        model = None
        model1 = None

        if self.dataset == 'cifar10':
            model = CIFAR10CNN()
            model1= CIFAR10CNN()
        elif self.dataset == "imagenet":
            model = resnet.resnet1()
            model1=resnet.resnet1()

        for i in range(self.agents):
            if i == 0:
                self.agent_models[0] = model
                self.test_agent_models[0] = model1
                self.temp_models[0] = type(model)()
            else:
                self.agent_models[i] = model_copy(self.agent_models[0])
                self.test_agent_models[i] = model_copy(self.test_agent_models[0])
                self.temp_models[i] = model_copy(self.temp_models[0])

            self.agent_models[i].to(self.device)
            self.test_agent_models[i].to(self.device)
            self.temp_models[i].to(self.device)
            self.agent_models[i].train()

            self.y_dict[i] = None
            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                name=self.opt_name,
                device=self.device,
                stratified=self.stratified,
                var_y=self.y_dict[i],
                const_q=self.const_q,
            )

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval
        loss: dict = {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            lr, prev_lr = {}, {}
            vars_y: dict[int, list[Tensor]] = {}
            vars_x: dict[int, list[Tensor]] = {}
            grads: dict[int, list[Tensor]] = {}
            track_grads: dict[int, list[Tensor]] = {}
            prev_track_grads: dict[int, list[Tensor]] = {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.temp_models[i].train()
                self.temp_models[i].zero_grad()

                self.agent_models[i].train()
                self.agent_models[i].zero_grad()
                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                grads[i] = self.agent_optimizers[i].collect_grad(self.agent_models[i].parameters())

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                self.tol_train_loss += loss[i].item()

                if 0 == self.running_iteration:
                    init_loss = self.criterion(self.temp_models[i](inputs), labels)
                    init_loss.backward()
                    self.agent_optimizers[i].set_y(self.opt.collect_grad(self.temp_models[i].parameters()))

                    lr[i] = self.lr
                    prev_lr[i] = self.lr
                    prev_track_grads[i] = self.agent_optimizers[i].collect_y()
                else:
                    lr[i] = self.agent_optimizers[i].collect_lr()
                    prev_lr[i] = self.agent_optimizers[i].collect_prev_lr()
                    prev_track_grads[i] = self.agent_optimizers[i].collect_prev_track_grad()

                vars_y[i] = self.agent_optimizers[i].collect_y()
                vars_x[i] = self.agent_optimizers[i].collect_x()

            for i in range(self.agents):
                self.agent_optimizers[i].step(
                    _type='x',
                    lr=lr,
                    vars_x=vars_x,
                    vars_y=vars_y,
                    k=self.running_iteration,
                )

                self.temp_models[i].load_state_dict(self.agent_models[i].state_dict())
                self.temp_models[i].train()
                self.temp_models[i].zero_grad()

                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                track_loss = self.criterion(self.temp_models[i](inputs), labels)
                track_loss.backward()
                track_grads[i] = self.opt.collect_grad(self.temp_models[i].parameters())

            for i in range(self.agents):
                self.agent_optimizers[i].step(
                    _type='y',
                    vars_y=vars_y,
                    grads=grads[i],
                    track_grads=track_grads,
                    prev_track_grads=prev_track_grads,

                    lr=lr[i],
                    prev_lr=prev_lr[i],
                    var_x=vars_x[i],
                    k=self.running_iteration,
                )


            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                        self.running_iteration, self.test_agent_models,
                    ]
                )
                log_thread.start()

                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc

class DSGDNTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DSGDN
        self.opt_name = "DSGDN"
        super().__init__(*args, **kwargs)

        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval

        loss, prev_loss = {}, {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, u, lr = {}, {}, {}, {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                self.agent_models[i].train()

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params_grads()

                if self.running_iteration == 0:
                    u[i] = []
                else:
                    u[i] = self.agent_optimizers[i].collect_u()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                self.tol_train_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, u=u, grads=grads)

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()

                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return None

class DADAMTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DADAM
        self.opt_name = "DADAM"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval

        loss, prev_loss = {}, {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, m, v, v_hat, lr = {}, {}, {}, {}, {}, {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                self.agent_models[i].train()

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params_grads()

                if self.running_iteration == 0:
                    m[i] = []
                    v[i] = []
                    v_hat[i] = []
                else:
                    m[i] = self.agent_optimizers[i].collect_m()
                    v[i] = self.agent_optimizers[i].collect_v()
                    v_hat[i] = self.agent_optimizers[i].collect_v_hat()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                self.tol_train_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, m=m[i], v=v[i],
                                              v_hat=v_hat[i])

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                        self.running_iteration, self.test_agent_models],
                )
                log_thread.start()
                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return None

class DAMSGradTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAMSGrad
        self.opt_name = "DAMSGrad"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch, ):
        log_interval = self.log_interval

        loss, prev_loss = {}, {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, m, v, v_hat, u_tilde, lr = {}, {}, {}, {}, {}, {}, {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                self.agent_models[i].train()

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()

                vars[i], grads[i] = self.agent_optimizers[i].collect_params_grads()

                if self.running_iteration == 0:
                    m[i] = []
                    v[i] = []
                    v_hat[i] = []
                    u_tilde[i] = []
                else:
                    m[i] = self.agent_optimizers[i].collect_m()
                    v[i] = self.agent_optimizers[i].collect_v()
                    v_hat[i] = self.agent_optimizers[i].collect_v_hat()
                    u_tilde[i] = self.agent_optimizers[i].collect_u_tilde()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                self.tol_train_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, m=m[i], v=v[i],
                                              v_hat=v_hat[i],
                                              u_tilde=u_tilde)
            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                log_thread.start()

                total_acc, total_count, self.tol_train_loss, = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return None

class ATCDIGingTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = ATCDIGing
        self.opt_name = "ATCDIGing"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def agent_setup(self):
        self.agent_models: dict = {}
        self.test_agent_models: dict = {}
        self.agent_optimizers: dict = {}
        self.temp_models: dict = {}

        self.y_dict: dict = {}
        model = None
        model1 = None

        if self.dataset == 'cifar10':
            model = CIFAR10CNN()
            model1 = CIFAR10CNN()
        elif self.dataset == "imagenet":
            model = resnet.resnet1()
            model1 = resnet.resnet1()

        for i in range(self.agents):
            if i == 0:
                self.agent_models[0] = model
                self.test_agent_models[0] = model1
                self.temp_models[0] = type(model)()
            else:
                self.agent_models[i] = model_copy(self.agent_models[0])
                self.test_agent_models[i] = model_copy(self.test_agent_models[0])
                self.temp_models[i] = model_copy(self.temp_models[0])

            self.agent_models[i].to(self.device)
            self.test_agent_models[i].to(self.device)
            self.temp_models[i].to(self.device)
            self.agent_models[i].train()

            self.y_dict[i] = None
            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                name=self.opt_name,
                device=self.device,
                stratified=self.stratified,
                var_y=self.y_dict[i],
            )

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval
        loss: dict = {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars_y: dict[int, list[Tensor]] = {}
            vars_x: dict[int, list[Tensor]] = {}
            track_grads: dict[int, list[Tensor]] = {}
            prev_track_grads: dict[int, list[Tensor]] = {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.temp_models[i].train()
                self.temp_models[i].zero_grad()

                self.agent_models[i].train()
                self.agent_models[i].zero_grad()
                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
                self.tol_train_loss += loss[i].item()

                if 0 == self.running_iteration:
                    init_loss = self.criterion(self.temp_models[i](inputs), labels)
                    init_loss.backward()
                    self.agent_optimizers[i].set_y(self.opt.collect_grad(self.temp_models[i].parameters()))

                    prev_track_grads[i] = self.agent_optimizers[i].collect_y()
                else:
                    prev_track_grads[i] = self.agent_optimizers[
                        i].collect_prev_track_grad()

                vars_y[i] = self.agent_optimizers[i].collect_y()
                vars_x[i] = self.agent_optimizers[i].collect_x()

            for i in range(self.agents):
                self.agent_optimizers[i].step(
                    _type='x',
                    vars_x=vars_x,
                    vars_y=vars_y,
                )

                self.temp_models[i].load_state_dict(self.agent_models[i].state_dict())
                self.temp_models[i].train()
                self.temp_models[i].zero_grad()

                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                track_loss = self.criterion(self.temp_models[i](inputs), labels)
                track_loss.backward()
                track_grads[i] = self.opt.collect_grad(self.temp_models[i].parameters())

            for i in range(self.agents):
                self.agent_optimizers[i].step(
                    _type='y',
                    vars_y=vars_y,
                    track_grads=track_grads,
                    prev_track_grads=prev_track_grads,
                )

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)

                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                        self.running_iteration, self.test_agent_models,
                    ]
                )
                log_thread.start()

                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()

        return total_acc

class DSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DSGD
        self.opt_name = "DSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        log_interval = self.log_interval

        loss, prev_loss = {}, {}
        total_acc, total_count = .0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, lr = {}, {}, {}

            seed = self.seed
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.agent_optimizers[i].zero_grad()
                self.agent_models[i].train()

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params_grads()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                self.tol_train_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads)

            if 0 == self.running_iteration % log_interval:
                self.tol_train_loss *= (log_interval if self.running_iteration == 0 else 1)

                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                log_thread = threading.Thread(
                    target=self.it_logger,
                    args=[
                        total_acc, total_count, epoch, log_interval, self.tol_train_loss, self.agent_optimizers,
                        self.running_iteration, self.test_agent_models,
                    ]
                )
                log_thread.start()
                total_acc, total_count, self.tol_train_loss = .0, 0, .0
                for i in range(self.agents):
                    self.agent_models[i].train()
        return None