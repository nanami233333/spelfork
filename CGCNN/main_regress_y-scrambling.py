import argparse
import os
import shutil
import sys
import time
import warnings
import copy
import random
from random import sample
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import r2_score
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
from torch.utils.data import Dataset

###############################################################################
# 工具/辅助类，与之前相同
###############################################################################
class ScrambledCIFData(Dataset):
    """
    包装 CIFData，将其中的 target 替换为指定的 scrambled_targets。
    假定原始数据集的 __getitem__ 返回 (structure, target, cif_id)
    """
    def __init__(self, original_dataset, scrambled_targets):
        assert len(original_dataset) == len(scrambled_targets), "数据长度不匹配"
        self.original_dataset = original_dataset
        self.scrambled_targets = scrambled_targets

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        structure, _, cif_id = self.original_dataset[idx]
        # 返回时将 target 替换为打乱后的
        return structure, self.scrambled_targets[idx], cif_id

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mae(prediction, target):
    """Compute mean absolute error"""
    return torch.mean(torch.abs(target - prediction))

def class_eval(prediction, target):
    """对分类任务的各种指标：accuracy、precision、recall、fscore、auc"""
    prediction = np.exp(prediction.numpy())  # output是log-softmax时可以这样
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

###############################################################################
# 画图相关函数，与之前相同；这里多了一个可选参数run_index以防多次运行冲突
###############################################################################
def plot_loss_curve(train_losses, val_losses, run_index=None):
    csv_name = 'loss_curve_data.csv'
    png_name = 'loss_curve.png'
    if run_index is not None:
        csv_name = f'loss_curve_data_run_{run_index}.csv'
        png_name = f'loss_curve_run_{run_index}.png'
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([epoch, train_loss, val_loss])
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(png_name)
    plt.close()

def plot_predictions_vs_true_with_r2(targets, predictions, r2, run_index=None):
    csv_name = 'predictions_vs_true_with_r2_data.csv'
    png_name = 'predictions_vs_true_with_r2.png'
    if run_index is not None:
        csv_name = f'predictions_vs_true_with_r2_data_run_{run_index}.csv'
        png_name = f'predictions_vs_true_with_r2_run_{run_index}.png'
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Values', 'Predicted Values'])
        for target, prediction in zip(targets, predictions):
            writer.writerow([target, prediction])
    plt.figure()
    plt.scatter(targets, predictions, label='Predictions vs True Values')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)],
             color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values (R² = {r2:.2f})')
    plt.legend()
    plt.savefig(png_name)
    plt.close()

def plot_residuals(targets, residuals, run_index=None):
    csv_name = 'residuals_data.csv'
    png_name = 'residuals_vs_true.png'
    if run_index is not None:
        csv_name = f'residuals_data_run_{run_index}.csv'
        png_name = f'residuals_vs_true_run_{run_index}.png'
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Values', 'Residuals (True - Predicted)'])
        for target, residual in zip(targets, residuals):
            writer.writerow([target, residual])
    plt.figure()
    plt.scatter(targets, residuals, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('True Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title('Residuals vs True Values')
    plt.legend()
    plt.savefig(png_name)
    plt.close()

def plot_error_distribution(residuals, run_index=None):
    csv_name = 'error_distribution_data.csv'
    png_name = 'error_distribution_with_kde.png'
    if run_index is not None:
        csv_name = f'error_distribution_data_run_{run_index}.csv'
        png_name = f'error_distribution_with_kde_run_{run_index}.png'
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Residuals'])
        for residual in residuals:
            writer.writerow([residual])
    plt.figure()
    sns.histplot(residuals, bins=30, kde=True, edgecolor='k', color='blue', alpha=0.6)
    plt.xlabel('Prediction Error (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution with KDE')
    plt.savefig(png_name)
    plt.close()

###############################################################################
# 训练和验证函数（与之前类似）
###############################################################################
def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()

    model.train()
    end = time.time()

    for i, (input_data, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (
                Variable(input_data[0].cuda(non_blocking=True)),
                Variable(input_data[1].cuda(non_blocking=True)),
                input_data[2].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input_data[3]]
            )
        else:
            input_var = (
                Variable(input_data[0]),
                Variable(input_data[1]),
                input_data[2],
                input_data[3]
            )

        # normalize target (回归)
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # forward & loss
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            # 如果是分类，可以在这里计算 accuracy 等
            pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )

    return losses.avg  # 返回平均loss (或你想要的别的度量)

def validate(val_loader, model, criterion, normalizer, args, test=False, run_index=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()  # 回归时

    # 如果是测试阶段，需要收集预测和真实值进行可视化
    test_targets = []
    test_preds = []
    test_cif_ids = []

    model.eval()
    end = time.time()

    for i, (input_data, target, batch_cif_ids) in enumerate(val_loader):
        # 不计算梯度
        if args.cuda:
            with torch.no_grad():
                input_var = (
                    Variable(input_data[0].cuda(non_blocking=True)),
                    Variable(input_data[1].cuda(non_blocking=True)),
                    input_data[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input_data[3]]
                )
        else:
            with torch.no_grad():
                input_var = (
                    Variable(input_data[0]),
                    Variable(input_data[1]),
                    input_data[2],
                    input_data[3]
                )

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # forward
        output = model(*input_var)
        loss = criterion(output, target_var)

        # record
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            # 分类时可以记录acc, auc...
            losses.update(loss.data.cpu(), target.size(0))
            if test:
                # 同理, test_preds收集正类概率 or logits, test_targets收集标签
                pass

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))

    if test and args.task == 'regression':
        # 计算 R²、画图
        r2 = r2_score(test_targets, test_preds)
        residuals = [t - p for t, p in zip(test_targets, test_preds)]
        plot_predictions_vs_true_with_r2(test_targets, test_preds, r2, run_index=run_index)
        plot_residuals(test_targets, residuals, run_index=run_index)
        plot_error_distribution(residuals, run_index=run_index)
        # 保存预测结果
        with open('test_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for cif_id, t, p in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow([cif_id, t, p])

    if args.task == 'regression':
        print(' * MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' * Loss {loss.avg:.4f}'.format(loss=losses))
        return losses.avg  # 或者返回你想追踪的分类指标

###############################################################################
# “只训练一次” 的主逻辑，可传入现成的 dataset (打乱后用)
###############################################################################
def train_and_evaluate_once(args, seed, run_index=None, external_dataset=None):
    """
    训练 + 测试一次。如果 external_dataset 不为 None，就用它；否则根据 args 去加载 dataset。
    返回测试集上的性能指标（回归为MAE，分类可自定义）。
    """
    if args.task == 'regression':
        best_metric = 1e10
    else:
        best_metric = 0.0

    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)

    train_losses = []
    val_losses = []

    # ============ 1. 加载数据集 ==================
    if external_dataset is not None:
        # 如果外部已经给了 dataset（可能被y-scramble过），直接用它
        dataset = external_dataset
    else:
        # 否则按照原逻辑加载
        dataset = CIFData(*args.data_options, radius=args.radius)

    # 构造 DataLoader
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True
    )

    # ============ 2. Normalizer ================
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        # 选取 500 个样本用来估计 mean, std
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # ============ 3. 构建模型 =================
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=True if args.task == 'classification' else False
    )
    if args.cuda:
        model.cuda()

    # ============ 4. 定义损失函数和优化器 ========
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # ============ 5. checkpoint (可选) ==========
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 学习率调度器
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # ============ 6. 开始训练 ===============
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer, args)
        train_losses.append(train_loss)

        val_loss = validate(val_loader, model, criterion, normalizer, args, test=False)
        val_losses.append(val_loss)
        val_metric = validate(val_loader, model, criterion, normalizer, args, test=False)

        if val_metric != val_metric:  # NaN检查
            print('Exit due to NaN in validation.')
            sys.exit(1)

        scheduler.step()

        # 根据回归/分类决定选优逻辑
        if args.task == 'regression':
            is_best = (val_metric < best_metric)
            best_metric = min(val_metric, best_metric)
        else:
            is_best = (val_metric > best_metric)
            best_metric = max(val_metric, best_metric)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_metric,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # 画训练曲线
    plot_loss_curve(train_losses, val_losses, run_index=run_index)

    # ============ 7. 测试集评估 =============
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_metric = validate(test_loader, model, criterion, normalizer, args, test=True, run_index=run_index)

    return test_metric

###############################################################################
# 这里是 y-scrambling 实验函数
###############################################################################
def y_scrambling_experiment(args, num_scramble_runs=5):
    # 1. 加载原始数据集（不打乱）
    original_dataset = CIFData(*args.data_options, radius=args.radius)
    # 提取所有 target（假定 __getitem__ 返回 (structure, target, cif_id)）
    original_targets = [original_dataset[i][1] for i in range(len(original_dataset))]

    scramble_results = []

    for run_idx in range(num_scramble_runs):
        scrambled_targets = original_targets.copy()
        random.shuffle(scrambled_targets)
        scrambled_dataset = ScrambledCIFData(original_dataset, scrambled_targets)

        seed_for_this_run = args.seed + 1000 * run_idx
        test_metric = train_and_evaluate_once(
            args,
            seed=seed_for_this_run,
            run_index=f'yScramble_{run_idx+1}',
            external_dataset=scrambled_dataset
        )
        scramble_results.append(test_metric)

    scramble_results = np.array(scramble_results)
    mean_score = np.mean(scramble_results)
    std_score = np.std(scramble_results)

    print("\n===== Y-scrambling Results over {} runs =====".format(num_scramble_runs))
    print(f"Scrambled test metric list: {scramble_results}")
    print(f"Average scrambled metric = {mean_score:.4f} ± {std_score:.4f}")
    print("==============================================\n")

    return mean_score, std_score


###############################################################################
# 主函数 main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
    parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                        help='dataset options, starting with the path to root dir, then other options')
    parser.add_argument('--task', choices=['regression', 'classification'],
                        default='regression', help='Task type (default: regression)')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (default: 30)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int, metavar='N',
                        help='milestones for scheduler (default: [100])')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--radius', default=5.0, type=float,
                        help='neighbor search radius (default: 5.0 Å)')

    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                            help='percentage of training data')
    train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                            help='number of training data')
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                            help='percentage of validation data (default 0.1)')
    valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                            help='number of validation data')
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                            help='percentage of test data (default 0.1)')
    test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                            help='number of test data')

    parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                        help='choose an optimizer, SGD or Adam, (default: SGD)')
    parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                        help='number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                        help='number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                        help='number of conv layers')
    parser.add_argument('--n-h', default=1, type=int, metavar='N',
                        help='number of hidden layers after pooling')

    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--num-runs', default=1, type=int,
                        help='Number of repeated runs with different random seeds (default: 1)')

    # y-scrambling 参数：如果大于0，就执行 y-scrambling 实验
    parser.add_argument('--y-scramble-runs', default=0, type=int,
                        help='Perform y-scrambling multiple times (default: 0 means no y-scramble)')

    args = parser.parse_args(sys.argv[1:])
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # ========================
    # 先做正常训练多次(不打乱)
    # ========================
    all_results = []
    for run_idx in range(args.num_runs):
        current_seed = args.seed + run_idx
        print(f"\n=== Normal Run {run_idx+1}/{args.num_runs}, Seed={current_seed} ===")
        test_metric = train_and_evaluate_once(args, seed=current_seed, run_index=(run_idx+1))
        all_results.append(test_metric)

    # 输出多次正常训练的平均结果
    all_results = np.array(all_results)
    mean_score = np.mean(all_results)
    std_score = np.std(all_results)

    if args.task == 'regression':
        print("\n===== Final Results over {} runs (Regression) =====".format(args.num_runs))
        print("MAE list:", all_results)
        print(f"Average MAE = {mean_score:.4f} ± {std_score:.4f}")
    else:
        print("\n===== Final Results over {} runs (Classification) =====".format(args.num_runs))
        print("Metric list:", all_results)
        print(f"Average metric = {mean_score:.4f} ± {std_score:.4f}")

    with open('multi_run_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['RunIndex', 'TestMetric'])
        for i, val in enumerate(all_results):
            writer.writerow([i+1, val])

    # ========================
    # 如果需要做 y-scramble
    # ========================
    if args.y_scramble_runs > 0:
        print(f"\nNow do Y-scrambling for {args.y_scramble_runs} runs...\n")
        y_scrambling_experiment(args, num_scramble_runs=args.y_scramble_runs)

if __name__ == '__main__':
    main()
