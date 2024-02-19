import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
from mycifar10 import MY_CIFAR10
from mycifar100 import MY_CIFAR100
from utils.utils_algo import sample_labeled_unlabeled_data
import torchvision.transforms as transforms

mean, std = {}, {}
mean['cifar10'] = [0.4914, 0.4822, 0.4465]
std['cifar10'] = [0.229, 0.224, 0.225]

mean['cifar100'] = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std['cifar100'] = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

mean['cub200'] = [0.485, 0.456, 0.406]
std['cub200'] = [0.229, 0.224, 0.225]

mean['svhn'] = [0.5, 0.5, 0.5]
std['svhn'] = [0.5, 0.5, 0.5]

mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]

def cifar10_dataloaders(data_dir, args):
    # Data loader for test dataset
    test_transform = Compose([
        ToTensor(),
        Normalize(mean['cifar10'], std['cifar10']),
    ])
    cifar10_test_ds = datasets.CIFAR10(data_dir, transform=test_transform, train=False, download=True)
    # print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test_loader = DataLoader(
        cifar10_test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 如果是半监督的设定
    if hasattr(args, 'num_labels'):
        # 读取原始训练集
        ori_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
        base_dir = os.path.dirname(__file__)
        # 进行lb和ulb的划分
        lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, ori_dataset.targets, args.num_classes,
                                                        args.num_labels, base_dir)
        # 构造lb数据集和ulb数据集
        cifar10_train_ds_lb = MY_CIFAR10(data_dir, args, train=True, download=True, rate_partial=args.partial_rate, labeled_idx=lb_idx)
        train_loader_lb = torch.utils.data.DataLoader(
            cifar10_train_ds_lb,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        cifar10_train_ds_ulb = MY_CIFAR10(data_dir, args, train=True, download=True, unlabeled_idx=ulb_idx)
        train_loader_ulb = torch.utils.data.DataLoader(
            cifar10_train_ds_ulb,
            batch_size=args.ratio*args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader_lb, train_loader_ulb, test_loader
    else:
        cifar10_train_ds = MY_CIFAR10(data_dir, args, train=True, download=True, rate_partial=args.partial_rate)
        train_loader = torch.utils.data.DataLoader(
            cifar10_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader, test_loader


def cifar100_dataloaders(data_dir, args):
    # Data loader for test dataset
    test_transform = Compose([
        ToTensor(),
        Normalize(mean['cifar100'], std['cifar100']),
    ])
    cifar10_test_ds = datasets.CIFAR100(data_dir, transform=test_transform, train=False, download=True)
    # print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
    test_loader = DataLoader(
        cifar10_test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 如果是半监督的设定
    if hasattr(args, 'num_labels'):
        # 读取原始训练集
        ori_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)
        base_dir = os.path.dirname(__file__)
        # 进行lb和ulb的划分
        lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, ori_dataset.targets, args.num_classes,
                                                        args.num_labels, base_dir)
        # 构造lb数据集和ulb数据集
        cifar100_train_ds_lb = MY_CIFAR100(data_dir, args, train=True, download=True, rate_partial=args.partial_rate,
                                         labeled_idx=lb_idx)
        train_loader_lb = torch.utils.data.DataLoader(
            cifar100_train_ds_lb,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        cifar100_train_ds_ulb = MY_CIFAR100(data_dir, args, train=True, download=True, unlabeled_idx=ulb_idx)
        train_loader_ulb = torch.utils.data.DataLoader(
            cifar100_train_ds_ulb,
            batch_size=args.ratio * args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader_lb, train_loader_ulb, test_loader
    else:
        cifar100_train_ds = MY_CIFAR100(data_dir, args, train=True, download=True, rate_partial=args.partial_rate)
        train_loader = torch.utils.data.DataLoader(
            cifar100_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        return train_loader, test_loader


def cub200_dataloaders(data_dir, args):
    input_size = 224
    test_transform = Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            ToTensor(),
            Normalize(mean['cub200'], std['cub200']),
    ])
    cub200_test_ds = MY_CUB200(data_dir, args, train=False, test_transform=test_transform, download=True)
    test_loader = DataLoader(
        cub200_test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 如果是半监督的设定
    if hasattr(args, 'num_labels'):
        # 读取原始训练集
        ori_dataset = MY_CUB200(data_dir, args, train=True, download=True)
        base_dir = os.path.dirname(__file__)
        # 进行lb和ulb的划分
        lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, ori_dataset.targets, args.num_classes,
                                                        args.num_labels, base_dir)
        # 构造lb数据集和ulb数据集
        cub200_train_ds_lb = MY_CUB200(data_dir, args, train=True, download=True, rate_partial=args.partial_rate,
                                           labeled_idx=lb_idx)
        train_loader_lb = torch.utils.data.DataLoader(
            cub200_train_ds_lb,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        cub200_train_ds_ulb = MY_CUB200(data_dir, args, train=True, download=True, unlabeled_idx=ulb_idx)
        train_loader_ulb = torch.utils.data.DataLoader(
            cub200_train_ds_ulb,
            batch_size=args.ratio * args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        return train_loader_lb, train_loader_ulb, test_loader



# def svhn_dataloaders(data_dir, args):
#     # Data loader for test dataset
#     test_transform = Compose([
#         ToTensor(),
#         Normalize(mean['svhn'], std['svhn']),
#     ])
#     svhn_test_ds = datasets.SVHN(data_dir, transform=test_transform, split='test', download=True)
#     # print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
#     test_loader = DataLoader(
#         svhn_test_ds,
#         batch_size=64,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
#
#     # 如果是半监督的设定
#     if hasattr(args, 'num_labels'):
#         # 读取原始训练集
#         ori_dataset = datasets.SVHN(root=data_dir, split='train', download=True)
#         base_dir = os.path.dirname(__file__)
#         # 进行lb和ulb的划分
#         lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, ori_dataset.labels, args.num_classes,
#                                                         args.num_labels, base_dir)
#         # 构造lb数据集和ulb数据集
#         svhn_train_ds_lb = MY_SVHN(data_dir, args, split='train', download=True, rate_partial=args.partial_rate,
#                                          labeled_idx=lb_idx)
#         train_loader_lb = torch.utils.data.DataLoader(
#             svhn_train_ds_lb,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True
#         )
#
#         svhn_train_ds_ulb = MY_SVHN(data_dir, args, split='train', download=True, unlabeled_idx=ulb_idx)
#         train_loader_ulb = torch.utils.data.DataLoader(
#             svhn_train_ds_ulb,
#             batch_size=args.ratio * args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True
#         )
#         return train_loader_lb, train_loader_ulb, test_loader
#     else:
#         svhn_train_ds = MY_SVHN(data_dir, args, split='train', download=True, rate_partial=args.partial_rate)
#         train_loader = torch.utils.data.DataLoader(
#             svhn_train_ds,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True
#         )
#         return train_loader, test_loader

# def stl10_dataloaders(data_dir, args):
#     # Data loader for test dataset
#     test_transform = Compose([
#         ToTensor(),
#         Normalize(mean['stl10'], std['stl10']),
#     ])
#     stl10_test_ds = datasets.STL10(data_dir, transform=test_transform, split='test', download=True)
#     # print('Test set -- Num_samples: {0}'.format(len(cifar10_test_ds)))
#     test_loader = DataLoader(
#         stl10_test_ds,
#         batch_size=64,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
#
#     # 如果是半监督的设定
#     if hasattr(args, 'num_labels'):
#         # 读取原始带标签的训练集
#         ori_dataset = datasets.STL10(root=data_dir, split='train', download=True)
#         base_dir = os.path.dirname(__file__)
#         # 进行lb和ulb的划分
#         lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, ori_dataset.labels, args.num_classes,
#                                                         args.num_labels, base_dir)
#         # 构造lb数据集和ulb数据集
#         stl10_train_ds_lb = MY_STL10(data_dir, args, split='train', download=True, rate_partial=args.partial_rate,
#                                          labeled_idx=lb_idx, lb_dataset=True)
#         train_loader_lb = torch.utils.data.DataLoader(
#             stl10_train_ds_lb,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True
#         )
#
#         stl10_train_ds_ulb = MY_STL10(data_dir, args, split='train+unlabeled', download=True, unlabeled_idx=ulb_idx,
#                                      lb_dataset=False)
#         train_loader_ulb = torch.utils.data.DataLoader(
#             stl10_train_ds_ulb,
#             batch_size=args.ratio * args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             drop_last=True
#         )
#         return train_loader_lb, train_loader_ulb, test_loader
#     else:
#         svhn_train_ds = MY_SVHN(data_dir, args, split='train', download=True, rate_partial=args.partial_rate)
#         train_loader = torch.utils.data.DataLoader(
#             svhn_train_ds,
#             batch_size=args.batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True
#         )
#         return train_loader, test_loader