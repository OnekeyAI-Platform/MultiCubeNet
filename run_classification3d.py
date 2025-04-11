# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/4/11
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import argparse
import json
import os
import random
import shutil

from onekey_algo.datasets.MultiTaskDataset import MultiTaskDataset3D
from onekey_algo.utils import create_dir_if_not_exists
from onekey_core.core.losses_factory import cox_loss, CoxPHLoss
from onekey_core.models.fusion.MultiCubeNet import MultiCubeNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType,
)
from torch.utils.tensorboard import SummaryWriter

from onekey_algo.datasets import create_classification_dataset
from onekey_algo.utils.about_log import logger, log_long_str
from onekey_core.core import create_lr_scheduler, create_optimizer, create_losses, create_model


def __config_list_or_folder_dataset(records, data_pattern):
    if not isinstance(records, (list, tuple)):
        records = [records]
    if records:
        if all(os.path.exists(r) for r in records) and all(os.path.isdir(r) for r in records):
            return {'records': None, 'ori_img_root': records}
        elif all(os.path.isfile(r) for r in records):
            return {'records': records, 'ori_img_root': data_pattern}
    else:
        return {'records': None, 'ori_img_root': data_pattern}
    raise ValueError(f"records({records}) or data_pattern({data_pattern}) config error!")


def train_one_epoch(model, device, train_loader: DataLoader, criterion, optimizer, lr_scheduler, writer, epoch,
                    log_file, **kwargs):
    model.train()
    epoch_loss = 0
    step = 0
    iters_verbose = kwargs.get('iters_verbose', 1)
    train_dir = kwargs.get('train_dir', './train')
    os.makedirs(train_dir, exist_ok=True)
    train_file_spec = open(os.path.join(train_dir, f'Epoch-{epoch}.csv'), 'w')

    fname_idx = 0
    for idx, bd in enumerate(train_loader):
        step += 1
        samples = bd[3]
        inputs, clf_l, sur_l = bd[0].to(device), bd[1].to(device), bd[2].to(device)
        optimizer.zero_grad()
        model_outputs = model(inputs)
        sur_loss = criterion[-1](model_outputs[-1], sur_l)
        clf_losses = []
        for lidx in range(len(model_outputs) - 1):
            clf_losses.append(criterion[0](model_outputs[lidx], clf_l[:, lidx]))
        if sur_loss is not None and not torch.isnan(sur_loss):
            loss = sur_loss + sum(clf_losses)
            info = [epoch, idx, float(sur_loss), *[float(i_) for i_ in clf_losses]]
            print(','.join(map(str, info)), file=log_file)
        else:
            loss = sum(clf_losses)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        epoch_len = len(train_loader.dataset) // train_loader.batch_size
        if idx % iters_verbose == 0:
            logger.info(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, sur_loss: {sur_loss}, "
                        f"clf_loss: {sum(clf_losses):.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        task_probs = []
        for oidx, outputs in enumerate(model_outputs):
            if oidx == len(model_outputs) - 1:
                task_probs.append(outputs)
            else:
                probs = nn.functional.softmax(outputs, dim=1)
                task_probs.append(probs)
        task_probs = torch.concat(task_probs + [clf_l, sur_l], dim=1)
        for sample, line in zip(samples, task_probs):
            train_file_spec.write('%s,%s\n' % (sample, ','.join(map(str, line.detach().cpu().numpy()))))
        fname_idx += train_loader.batch_size
    epoch_loss /= step
    logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    return epoch_loss


def evaluate(model, device, val_loader, writer, epoch, **kwargs):
    model.eval()
    val_dir = kwargs.get('val_dir', './valid')
    os.makedirs(val_dir, exist_ok=True)
    val_file_spec = open(os.path.join(val_dir, f'Epoch-{epoch}.csv'), 'w')
    fname_idx = 0
    with torch.no_grad():
        # num_correct = 0.0
        # metric_count = 0
        for bd in val_loader:
            samples = bd[3]
            inputs, clf_l, sur_l = bd[0].to(device), bd[1].to(device), bd[2].to(device)
            model_outputs = model(inputs)
            # value = torch.eq(val_outputs.argmax(dim=1), labels)
            # print(val_outputs.argmax(dim=1), val_labels)
            # metric_count += len(value)
            # num_correct += value.sum().item()
            task_probs = []
            for oidx, outputs in enumerate(model_outputs):
                if oidx == len(model_outputs) - 1:
                    task_probs.append(outputs)
                else:
                    probs = nn.functional.softmax(outputs, dim=1)
                    task_probs.append(probs)
            task_probs = torch.concat(task_probs + [clf_l, sur_l], dim=1)
            for sample, line in zip(samples, task_probs):
                val_file_spec.write('%s,%s\n' % (sample, ','.join(map(str, line.detach().cpu().numpy()))))
            fname_idx += val_loader.batch_size
        # metric = num_correct / metric_count
        metric = 0.5
        writer.add_scalar("val_accuracy", metric, epoch + 1)
    return metric


def main(args):
    train_transforms = Compose([ScaleIntensity(), Resize(args.roi_size), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), Resize(args.roi_size), EnsureType()])
    train_datasets = MultiTaskDataset3D(data_pattern=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\images',
                                        clf_label_file=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\clf_label.csv',
                                        sur_label_file=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\survival.csv',
                                        transform=train_transforms, subset='train')
    val_datasets = MultiTaskDataset3D(data_pattern=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\images',
                                      clf_label_file=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\clf_label.csv',
                                      sur_label_file=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI\survival.csv',
                                      transform=val_transforms, subset='all')
    log_long_str('Train:%s\nValid:%s' % (train_datasets, val_datasets))

    # create a training data loader
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.j, pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_loader = DataLoader(val_datasets, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available(),
                            shuffle=False)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiCubeNet([256, 128, 64, 32, 16], in_channels=3, num_clf_classes=[2, 2, 2], num_reg_tasks=1).to(device)
    clf_loss = create_losses('softmax_ce')
    sur_loss = CoxPHLoss()
    optimizer = create_optimizer(args.optimizer, parameters=model.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(train_datasets) // args.batch_size)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    model_dir = create_dir_if_not_exists(os.path.join(args.model_root), add_date=False)
    create_dir_if_not_exists(os.path.join(model_dir, 'runs'))
    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'runs'))
    max_epochs = args.epochs
    log_file = open(os.path.join(model_dir, f'log.csv'), 'w')
    for epoch in range(max_epochs):
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = train_one_epoch(model, device, train_loader, (clf_loss, sur_loss), optimizer, lr_scheduler,
                                     writer=writer, epoch=epoch, iters_verbose=args.iters_verbose,
                                     train_dir=os.path.join(model_dir, 'train'), log_file=log_file)
        epoch_loss_values.append(epoch_loss)
        if (epoch + 1) % args.val_interval == 0:
            metric = evaluate(model, device, val_loader, writer=writer, epoch=epoch,
                              val_dir=os.path.join(model_dir, 'valid'))
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                create_dir_if_not_exists(os.path.join(model_dir, 'viz'))
                torch.save({'state_dict': model.state_dict()},
                           os.path.join(model_dir, 'viz', f"BEST-training-params.pth"))
                shutil.copy(os.path.join(model_dir, 'train', f'Epoch-{epoch}.csv'),
                            os.path.join(model_dir, 'viz', 'BST_TRAIN_RESULTS.csv'))
                shutil.copy(os.path.join(model_dir, 'valid', f'Epoch-{epoch}.csv'),
                            os.path.join(model_dir, 'viz', 'BST_VAL_RESULTS.csv'))
            logger.info(f"current epoch: {epoch + 1} current accuracy: {metric:.4f} "
                        f"best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        torch.save({'state_dict': model.state_dict()},
                   os.path.join(model_dir, 'train', f"params_{epoch}.pth"))
    with open(os.path.join(model_dir, 'viz', f'task.json'), 'w') as task_info_file:
        task_info = {'model_name': args.model_name, 'num_classes': train_datasets.num_classes,
                     'in_channels': 1, 'input_size': args.roi_size}
        print(json.dumps(task_info, ensure_ascii=False, indent=True), file=task_info_file)
    logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


DATA_ROOT = '/Users/zhangzhiwei/Downloads/Skull'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=os.path.join(DATA_ROOT, 'train'), help='Training dataset')
    parser.add_argument('--val', nargs='*', default=os.path.join(DATA_ROOT, 'val'), help='Validation dataset')
    parser.add_argument('--labels_file', default=os.path.join(DATA_ROOT, 'labels.txt'), help='Labels file')
    parser.add_argument('--data_pattern', default=None, nargs='*', help='Where to save origin image data.')
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--max2use', default=None, type=int, help='Maximum number of sample per class to be used!')
    parser.add_argument('--normalize_method', default='imagenet', choices=['-1+1', 'imagenet'],
                        help='Normalize method.')
    parser.add_argument('--model_name', default='resnet18', help='Model name')
    parser.add_argument('--roi_size', default=[96, 96, 96], type=int, nargs='*', help='Model name')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0], help='GPU index to be used!')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--model_root', default=r'E:\20230814-ZhangHongBo\Glioma_OnekeyAI/models_re',
                        help='path where to save')
    parser.add_argument('--val_interval', default=1, type=int, help='val_interval')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--iters_verbose', default=1, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')

    main(parser.parse_args())
