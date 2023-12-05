import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from termcolor import colored
from pareto.datasets.ensemble import CausalDataset
from pareto.networks.policynet import PolicyNet
from pareto.networks.MutualInfor.module import Net


parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='')
parser.add_argument('--cuda_enabled', type=bool, default=True, help='')
parser.add_argument('--cuda_deterministic', type=bool, default=False, help='')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=500, help='')
parser.add_argument('--sample_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=16, help='')

parser.add_argument('--num_prefs', type=int, default=6, help='number of initial solutions')
parser.add_argument('--num_tasks', type=int, default=2, help='number of multi-tasks')

parser.add_argument('--dataset', type=str, default='jobs', help='dataset name')

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


def train(pref, ckpt_name):
    # prepare path
    root_path = Path(__file__).resolve().parent
    dataset_path = root_path / 'long_term' / 'dataset' / 'ensemble' / args.dataset
    scale_path = root_path / 'long_term' / 'policy_initial' / args.dataset
    seed_folder = "seed_%d" % args.seed
    ckpt_path = scale_path / seed_folder

    estimation_path = root_path / 'long_term' / 'cpmtl' / args.dataset / seed_folder / '0' / 'best.pth'

    root_path.mkdir(parents=True, exist_ok=True)
    scale_path.mkdir(parents=True, exist_ok=True)
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

    # prepare network
    network = PolicyNet(x_dim=trainset.data.x_dim).to(device)
    net_optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # initialize network
    start_ckpt = torch.load(estimation_path, map_location='cpu')
    estimator = Net(x_dim=trainset.data.x_dim).to(device)
    estimator.load_state_dict(start_ckpt['state_dict'])

    # first evaluation
    criterion = nn.L1Loss()
    best_metrics = evaluate(network, testset, device, criterion, estimator, f'{ckpt_name}')

    # save initial state
    if not (ckpt_path / 'random.pth').is_file():
        random_ckpt = {
            'state_dict': network.state_dict(),
            'net_optimizer': net_optimizer.state_dict()
        }
        torch.save(random_ckpt, ckpt_path / 'random.pth')
    random_ckpt = torch.load(ckpt_path / 'random.pth', map_location='cpu')

    network.load_state_dict(random_ckpt['state_dict'])
    net_optimizer.load_state_dict(random_ckpt['net_optimizer'])

    # training
    num_steps = len(trainloader)
    flag = True
    for epoch in range(1, args.num_epochs + 1):
        trainiter = iter(trainloader)
        for i in range(1, num_steps + 1):
            network.train()

            X, _, _, _ = next(trainiter)
            X = X.to(device)

            T = network(X)

            # inputs = torch.cat([X, T], dim=1)
            # _, hat_s, hat_y = estimator(inputs)
            _, hat_s, hat_y = estimator(X, T)

            batch_size = len(X)
            gt_s = expand(batch_size, testset.data.s_max + 1, device)
            gt_y = expand(batch_size, testset.data.y_max + 1, device)

            loss_s = criterion(gt_s, hat_s)
            loss_y = criterion(gt_y, hat_y)
            losses = [loss_s, loss_y]
            loss = sum(w * l for w, l in zip(pref, losses))
            net_optimizer.zero_grad()
            loss.backward()  # retain_graph=True)
            net_optimizer.step()

        if epoch % 1 == 0:
            new_losses = evaluate(network, testset, device, criterion, estimator, f'{epoch}')

            if new_losses[0] + new_losses[1] < best_metrics[0] + best_metrics[1]:
                best_metrics = new_losses
                flag = False
                ckpt = {
                    'state_dict': network.state_dict(),
                    'net_optimizer': net_optimizer.state_dict(),
                    'preference': pref,
                }
                torch.save(ckpt, ckpt_path / f'{ckpt_name}.pth')

    # saving
    if flag:
        ckpt = {
            'state_dict': network.state_dict(),
            'net_optimizer': net_optimizer.state_dict(),
            'preference': pref,
        }
        torch.save(ckpt, ckpt_path / f'{ckpt_name}.pth')


def weighted_sum():
    # 多任务数量为 dim，返回 num_prefs 数量的 weights 分配方案（均匀分配）
    prefs = [(0.5, 0.5)]
    for i, pref in enumerate(prefs):
        # 针对每种 weights 分配方案，都进行训练
        print("=" * 80)
        print("seed: ", args.seed)
        print("initial solutions: ", prefs)
        train(pref, str(i))


if __name__ == '__main__':
    weighted_sum()
