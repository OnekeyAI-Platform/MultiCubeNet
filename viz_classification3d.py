# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/4/11
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import argparse
import os

import matplotlib

matplotlib.use('Agg')
from onekey_algo.datasets.MultiTaskDataset import MultiTaskDataset3D
from onekey_core.models.fusion.MultiCubeNet import MultiCubeNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    Resize,
    ScaleIntensity,
    EnsureType,
)
import matplotlib.pyplot as plt
import monai
from onekey_algo.utils.about_log import log_long_str


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


def main(args):
    train_transforms = Compose([ScaleIntensity(), Resize(args.roi_size), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), Resize(args.roi_size), EnsureType()])
    train_datasets = MultiTaskDataset3D(data_pattern=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\images',
                                        clf_label_file=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\clf_label.csv',
                                        sur_label_file=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\survival.csv',
                                        transform=train_transforms, subset='train')
    val_datasets = MultiTaskDataset3D(data_pattern=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\images',
                                      clf_label_file=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\clf_label.csv',
                                      sur_label_file=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\survival.csv',
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
    state_dict = torch.load(args.retrain, map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    for n, m in model.named_modules():
        print('Feature name:', n, "|| Module:", m)

    target_layer = "l5f.2"
    gradcam = monai.visualize.GradCAM(nn_module=model, target_layers=target_layer)

    for sample_, _, _, sn in val_loader:
        save2 = rf'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\models_re/Grad-CAM/{os.path.basename(sn[0])}'
        if os.path.exists(save2):
            continue
        res_cam = gradcam(x=sample_.to(device), class_idx=None)
        sample_np = sample_.cpu().detach().numpy()
        print(torch.sum(res_cam), sample_np.shape)
        for idx in range(sample_.size()[-1]):
            num_channels = sample_np.shape[1]
            fig, axes = plt.subplots(1, num_channels + 1, figsize=(4 * (num_channels + 1), 4), facecolor='white')
            for i in range(num_channels):
                axes[i].imshow(sample_np[0][i][idx, ...], cmap='gray')
                axes[i].axis('off')
            imshow = axes[num_channels].imshow(res_cam[0][0][idx, ...].cpu().detach().numpy(), cmap='jet')
            axes[num_channels].axis('off')
            cax = fig.add_axes([0.92, 0.17, 0.02, axes[num_channels].get_position().height])
            plt.colorbar(imshow, cax=cax)

            os.makedirs(save2, exist_ok=True)
            plt.savefig(os.path.join(save2, f'{idx}.jpg'), bbox_inches='tight')
            plt.close()


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
    parser.add_argument('--retrain', default=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI\models_re\train/params_8.pth',
                        help='Retrain from path')
    parser.add_argument('--model_root', default=r'Z:\20230814-ZhangHongBo\Glioma_OnekeyAI/models_re',
                        help='path where to save')
    parser.add_argument('--val_interval', default=1, type=int, help='val_interval')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--iters_verbose', default=1, type=int, help='Iters start')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained core or not')

    main(parser.parse_args())
