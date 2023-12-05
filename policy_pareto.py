import random
from pathlib import Path
from termcolor import colored

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from argparse import ArgumentParser

from pareto.optim import MINRESKKTSolver
from pareto.optim.policy_hvp_solver import VisionHVPSolver
from pareto.datasets.ensemble import CausalDataset
from pareto.utils import TopTrace
from pareto.networks.policynet import PolicyNet
from pareto.networks.MutualInfor.module import Net

parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='')
parser.add_argument('--cuda_enabled', type=bool, default=True, help='')
parser.add_argument('--cuda_deterministic', type=bool, default=False, help='')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='')

parser.add_argument('--shared', type=bool, default=False, help='')
parser.add_argument('--stochastic', type=bool, default=False, help='')
parser.add_argument('--kkt_momentum', type=float, default=0.0, help='')
parser.add_argument('--create_graph', type=bool, default=False, help='')
parser.add_argument('--grad_correction', type=bool, default=False, help='')
parser.add_argument('--shift', type=float, default=0.0, help='')
parser.add_argument('--tol', type=float, default=1e-5, help='')
parser.add_argument('--damping', type=float, default=0.1, help='')
parser.add_argument('--maxiter', type=int, default=50, help='')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='')
parser.add_argument('--num_steps', type=int, default=100, help='')
parser.add_argument('--verbose', type=bool, default=False, help='')
parser.add_argument('--dataset', type=str, default='jobs', help='dataset name')
parser.add_argument('--sample_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=16, help='')
parser.add_argument('--num_tasks', type=int, default=2, help='number of multi-tasks')
parser.add_argument('--beta1', type=float, default=0.5, help='')
parser.add_argument('--beta2', type=float, default=0.5, help='')

args = parser.parse_args()


def expand(bs, val, device):
    gt = torch.tensor(val)
    gt = gt.unsqueeze(0)
    gt = gt.expand(bs, -1)
    return gt.to(device)


@torch.no_grad()
def evaluate(network, testset, device, criterion, estimator, header=''):
    num_samples = 0
    losses = np.zeros(args.num_tasks)
    network.train(False)

    # 评估所有样本的 loss
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    for X, _, _, _ in testloader:
        batch_size = len(X)
        num_samples += batch_size
        X = X.to(device)
        T = network(X)

        # inputs = torch.cat([X, T], dim=1)
        # _, hat_s, hat_y = estimator(inputs)
        _, hat_s, hat_y = estimator(X, T)

        gt_s = expand(batch_size, testset.data.s_max + 1, device)
        gt_y = expand(batch_size, testset.data.y_max + 1, device)

        loss_s = criterion(gt_s, hat_s).item()
        loss_y = criterion(gt_y, hat_y).item()

        losses_batch = [loss_s, loss_y]
        losses += batch_size * np.array(losses_batch)

    # 计算平均值
    losses /= num_samples

    # 打印评估结果
    loss_msg = '[{}]'.format('/'.join([f'{loss:.6f}' for loss in losses]))
    msgs = [
        f'{header}:' if header else '',
        'loss', colored(loss_msg, 'yellow'),
    ]
    print(' '.join(msgs))
    return losses


def train(start_path, beta):
    # prepare path
    ckpt_name = start_path.name.split('.')[0]
    root_path = Path(__file__).resolve().parent
    dataset_path = root_path / 'long_term' / 'dataset' / 'ensemble' / args.dataset
    scale_path = root_path / 'long_term' / 'policy_pareto' / args.dataset
    seed_folder = "seed_%d" % args.seed
    seed_path = scale_path / seed_folder
    ckpt_path = seed_path / ckpt_name

    policy_path = root_path / 'long_term' / 'policy_initial' / args.dataset / seed_folder / '0.pth'
    estimation_path = root_path / 'long_term' / 'cpmtl' / args.dataset / seed_folder / '0' / 'best.pth'

    if not start_path.is_file():
        raise RuntimeError('Pareto solutions not found.')

    root_path.mkdir(parents=True, exist_ok=True)
    scale_path.mkdir(parents=True, exist_ok=True)
    seed_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda_enabled and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # prepare device
    if args.cuda_enabled and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        if args.cuda_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # prepare dataset
    trainset = CausalDataset(args.dataset, dataset_path, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)

    testset = CausalDataset(args.dataset, dataset_path, train=False)

    train_s_min, train_s_max, train_y_min, train_y_max = trainset.get_min_max()

    # prepare network
    network = PolicyNet(x_dim=trainset.data.x_dim).to(device)
    net_optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # initialize network
    start_ckpt = torch.load(policy_path, map_location='cpu')
    network.load_state_dict(start_ckpt['state_dict'])
    net_optimizer.load_state_dict(start_ckpt['net_optimizer'])

    estimator = Net(x_dim=trainset.data.x_dim).to(device)
    estimation_ckpt = torch.load(estimation_path, map_location='cpu')
    estimator.load_state_dict(estimation_ckpt['state_dict'])

    # prepare HVP solver
    hvp_solver = VisionHVPSolver(train_s_min, train_s_max, train_y_min, train_y_max, network, device, trainloader, estimator, shared=args.shared)
    hvp_solver.set_grad(batch=False)
    hvp_solver.set_hess(batch=True)

    # prepare KKT solver
    kkt_solver = MINRESKKTSolver(
        network, hvp_solver, device,
        stochastic=args.stochastic, kkt_momentum=args.kkt_momentum, create_graph=args.create_graph,
        grad_correction=args.grad_correction, shift=args.shift, tol=args.tol, damping=args.damping, maxiter=args.maxiter)

    # prepare optimizer
    optimizer = Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # first evaluation
    criterion = nn.L1Loss()
    losses = evaluate(network, testset, device, criterion, estimator, f'{ckpt_name}')
    best_metrics = losses

    # prepare utilities
    top_trace = TopTrace(3)
    top_trace.print(losses, show=False)

    beta = beta.to(device)

    # training
    flag = True
    for step in range(1, args.num_steps + 1):
        network.train()
        estimator.eval()
        optimizer.zero_grad()
        kkt_solver.backward(beta, verbose=args.verbose)
        optimizer.step()

        losses = evaluate(network, testset, device, criterion, estimator, f'{ckpt_name}: {step}/{args.num_steps}')

        if losses[0] + losses[1] < best_metrics[0] + best_metrics[1]:
            best_metrics = losses
            flag = False
            ckpt = {
                'state_dict': network.state_dict(),
                'net_optimizer': net_optimizer.state_dict()
            }
            record = {'losses': losses}
            ckpt['record'] = record
            torch.save(ckpt, ckpt_path / 'best.pth')

    # saving
    if flag:
        ckpt = {
            'state_dict': network.state_dict(),
            'net_optimizer': net_optimizer.state_dict()
        }
        record = {'losses': losses}
        ckpt['record'] = record
        torch.save(ckpt, ckpt_path / f'best.pth')

    hvp_solver.close()
    print("=" * 40 + " best " + "=" * 40)
    print(best_metrics)


def cpmtl():
    root_path = Path(__file__).resolve().parent
    seed_folder = "seed_%d" % args.seed
    start_root = root_path / 'long_term' / 'policy_initial' / args.dataset / seed_folder

    # beta = torch.Tensor([0.5, 0.5])
    beta = torch.Tensor([args.beta1, args.beta2])  # beta 就是 loss 权重

    for start_path in sorted(start_root.glob('[0-9]*.pth'), key=lambda x: int(x.name.split('.')[0])):
        # start_path 就是初始帕累托解
        train(start_path, beta)


if __name__ == "__main__":
    cpmtl()
