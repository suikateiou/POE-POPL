import torch
import numpy as np
import random
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from termcolor import colored
from pareto.networks.MutualInfor.mi_estimators import CLUB, CLUBSample
from pareto.datasets.ensemble import CausalDataset
from pareto.networks.MutualInfor.module import Net


parser = ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='')
parser.add_argument('--cuda_enabled', type=bool, default=True, help='')
parser.add_argument('--cuda_deterministic', type=bool, default=False, help='')
parser.add_argument('--bs', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=300, help='')
parser.add_argument('--sample_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=16, help='')

parser.add_argument('--num_prefs', type=int, default=6, help='number of initial solutions')
parser.add_argument('--num_tasks', type=int, default=3, help='number of multi-tasks')

parser.add_argument('--dataset', type=str, default='jobs', help='dataset name')

args = parser.parse_args()


@torch.no_grad()
def evaluate(network, dataloader, device, criterion, mi_estimator, header=''):
    num_samples = 0
    losses = np.zeros(args.num_tasks)
    network.train(False)

    # 评估所有样本的 loss
    for X, T, S, Y in dataloader:
        batch_size = len(T)
        num_samples += batch_size
        X = X.to(device)
        T = T.to(device)
        S = S.to(device)
        Y = Y.to(device)

        # inputs = torch.cat([X, T], dim=1)
        # hat_rep, hat_s, hat_y = network(inputs)
        hat_rep, hat_s, hat_y = network(X, T)
        loss_s = criterion(hat_s, S).item()
        loss_y = criterion(hat_y, Y).item()
        net_loss = mi_estimator(hat_rep, T).item()

        losses_batch = [loss_s, loss_y, net_loss]
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
    scale_path = root_path / 'long_term' / 'weighted_sum' / args.dataset
    seed_folder = "seed_%d" % args.seed
    ckpt_path = scale_path / seed_folder

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
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # prepare network
    network = Net(x_dim=trainset.data.x_dim).to(device)
    estimator_name = "CLUB"  # estimator_name in ["NWJ", "MINE", "InfoNCE", "L1OutUB", "CLUB", "CLUBSample"]
    mi_estimator = eval(estimator_name)(args.sample_dim, args.sample_dim, args.hidden_size).cuda()

    net_optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    mi_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=args.lr)
    # net_optimizer = CosineAnnealingLR(net_optimizer, args.num_epochs * len(trainloader))
    # mi_optimizer = CosineAnnealingLR(mi_optimizer, args.num_epochs * len(trainloader))

    criterion = nn.MSELoss()
    mi_est_values = []

    # first evaluation
    best_metrics = evaluate(network, testloader, device, criterion, mi_estimator, f'{ckpt_name}')

    # save initial state
    if not (ckpt_path / 'random.pth').is_file():
        random_ckpt = {
            'state_dict': network.state_dict(),
            'net_optimizer': net_optimizer.state_dict(),
            'mi_optimizer': mi_optimizer.state_dict()
        }
        torch.save(random_ckpt, ckpt_path / 'random.pth')
    random_ckpt = torch.load(ckpt_path / 'random.pth', map_location='cpu')

    network.load_state_dict(random_ckpt['state_dict'])
    net_optimizer.load_state_dict(random_ckpt['net_optimizer'])
    mi_optimizer.load_state_dict(random_ckpt['mi_optimizer'])

    # training
    num_steps = len(trainloader)
    fig_loss_s = []
    fig_loss_y = []
    fig_loss_rep = []
    fig_loss = []
    flag = True
    for epoch in range(1, args.num_epochs + 1):
        trainiter = iter(trainloader)
        for i in range(1, num_steps + 1):
            network.train()
            mi_estimator.eval()

            X, T, S, Y = next(trainiter)
            X = X.to(device)
            T = T.to(device)
            S = S.to(device)
            Y = Y.to(device)

            # inputs = torch.cat([X, T], dim=1)
            # hat_rep, hat_s, hat_y = network(inputs)
            hat_rep, hat_s, hat_y = network(X, T)
            loss_s = criterion(hat_s, S)
            loss_y = criterion(hat_y, Y)
            net_loss = mi_estimator(hat_rep, T)
            # loss = loss_s + loss_y + net_loss
            losses = [loss_s, loss_y, net_loss]
            loss = sum(w * l for w, l in zip(pref, losses))
            net_optimizer.zero_grad()
            loss.backward()  # retain_graph=True)
            net_optimizer.step()

            fig_loss_s.append(loss_s.item())
            fig_loss_y.append(loss_y.item())
            fig_loss_rep.append(net_loss.item())
            fig_loss.append(loss.item())

            for j in range(5):
                mi_estimator.train()
                hat_rep, hat_s, hat_y = network(X, T)
                mi_loss = mi_estimator.learning_loss(hat_rep, T)
                mi_optimizer.zero_grad()
                mi_loss.backward()
                mi_optimizer.step()

            mi_est_values.append(mi_estimator(hat_rep, T).item())

            # if i % 100 == 0:
            #     print("step {}, true MI value {}".format(i, mi_estimator(hat_rep, T).item()))

        if epoch % 1 == 0:
            new_losses = evaluate(network, testloader, device, criterion, mi_estimator, f'{epoch}')

            if new_losses[0] + new_losses[1] < best_metrics[0] + best_metrics[1]:
                best_metrics = new_losses
                flag = False
                ckpt = {
                    'state_dict': network.state_dict(),
                    'net_optimizer': net_optimizer.state_dict(),
                    'mi_optimizer': mi_optimizer.state_dict(),
                    'preference': pref,
                }
                # record = {'losses': best_metrics}
                # ckpt['record'] = record
                torch.save(ckpt, ckpt_path / f'{ckpt_name}.pth')

    # saving
    if flag:
        ckpt = {
            'state_dict': network.state_dict(),
            'net_optimizer': net_optimizer.state_dict(),
            'mi_optimizer': mi_optimizer.state_dict(),
            'preference': pref,
        }
        # record = {'losses': losses}
        # ckpt['record'] = record
        torch.save(ckpt, ckpt_path / f'{ckpt_name}.pth')


def weighted_sum():
    # 多任务数量为 dim，返回 num_prefs 数量的 weights 分配方案（均匀分配）
    w_rep = 0.001
    prefs = [(0.1, 0.9, w_rep), (0.3, 0.7, w_rep), (0.5, 0.5, w_rep), (0.7, 0.3, w_rep), (0.9, 0.1, w_rep)]
    for i, pref in enumerate(prefs):
        # 针对每种 weights 分配方案，都进行训练
        print("=" * 80)
        print("seed: ", args.seed)
        print("initial solutions: ", pref)
        train(pref, str(i))


if __name__ == '__main__':
    weighted_sum()
