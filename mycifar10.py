from PIL import Image
import os
import os.path
import numpy as np
import pickle
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from augment.cutout import Cutout
from augment.autoaugment_extra import CIFAR10Policy
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip
from sklearn.preprocessing import OneHotEncoder

mean, std = {}, {}
mean['cifar10'] = [0.4914, 0.4822, 0.4465]
std['cifar10'] = [0.229, 0.224, 0.225]

class MY_CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, args, train=True, transform=None, target_transform=None,
                 download=False, rate_partial=0.3, labeled_idx=None, unlabeled_idx=None):

        super(MY_CIFAR10, self).__init__(root, transform=transform,
                                         target_transform=target_transform)

        self.train = train  # training set or test set
        self.args = args
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        # print(len(self.targets))
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = torch.tensor(self.targets)

        self.weak_transform = Compose([
            # ToPILImage(),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            ToTensor(),
            Normalize(mean[self.args.dataset], std[self.args.dataset]),
        ])

        self.strong_transform = Compose([
            # ToPILImage(),
            RandomHorizontalFlip(),
            RandomCrop(32, 4, padding_mode='reflect'),
            CIFAR10Policy(),
            ToTensor(),
            Cutout(n_holes=1, length=16),
            Normalize(mean[self.args.dataset], std[self.args.dataset]),
        ])

        self.rate_partial = rate_partial

        # 如果给出了lb_idx，则构造lb数据集
        if labeled_idx is not None:
            # lb_targets = np.array(self.targets)[labeled_idx]
            # lb_targets_one_hot = F.one_hot(torch.tensor(lb_targets, dtype=torch.int64), num_classes=args.num_classes).type(torch.float32)
            # average_partial_lb_targets, _ = partialize(lb_targets_one_hot, args.partial_rate)
            self.data = self.data[labeled_idx]
            self.targets = self.targets[labeled_idx]
            self.partial_labels = self.generate_partial_labels()
            args.logger.info(f"Use {args.num_labels} labeled partial samples for training")
        # 如果给出了ulb_idx，则构造ulb数据集，ulb数据集把所有标签当做候选标记
        elif unlabeled_idx is not None and len(unlabeled_idx) != 0:
            self.data = self.data[unlabeled_idx]
            # average_partial_lb_targets = torch.ones((len(unlabeled_idx), args.num_classes)) / args.num_classes
            self.partial_labels = torch.ones((len(unlabeled_idx), args.num_classes))
            self.targets = self.targets[unlabeled_idx]
            args.logger.info(f"Use {len(unlabeled_idx)} unlabeled partial samples for training")
        # 常规偏标记设定
        else:
            self.partial_labels = self.generate_partial_labels()
            args.logger.info(f"Use {len(self.data)} labeled partial samples for training")

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, partial_label = self.data[index], self.targets[index], self.partial_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.weak_transform is not None:
            img_weak = self.weak_transform(img)
            img_strong = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labeled_idx is not None or not hasattr(self.args, 'num_labels'):
            return img_weak, target, partial_label, index
        else:
            return img_weak, img_strong, target, partial_label, index



    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            # print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def generate_partial_labels(self):
        if (self.rate_partial != -1):
            def binarize_class(y):
                # y is a tensor
                label = y.reshape(len(y), -1)
                enc = OneHotEncoder(categories='auto')
                enc.fit(label)
                label = enc.transform(label).toarray().astype(np.float32)
                label = torch.from_numpy(label)
                return label

            new_y = binarize_class(self.targets)
            n = len(self.targets)
            c = max(self.targets) + 1
            avgC = 0
            partial_rate = self.rate_partial
            print(partial_rate)
            for i in range(n):
                row = new_y[i, :]
                row[np.where(np.random.binomial(1, partial_rate, c) == 1)] = 1
                while torch.sum(row) == 1:
                    row[np.random.randint(0, c)] = 1
                avgC += torch.sum(row)
                new_y[i] = row

            avgC = avgC / n
            print("Finish Generating Candidate Label Sets:{}!\n".format(avgC))
            new_y = new_y.cpu().numpy()
            return new_y