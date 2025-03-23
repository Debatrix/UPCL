import os.path as osp
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data",
                                               train=True,
                                               download=True)
        test_dataset = datasets.cifar.CIFAR10("./data",
                                              train=False,
                                              download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                             std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data",
                                                train=True,
                                                download=True)
        test_dataset = datasets.cifar.CIFAR100("./data",
                                               train=False,
                                               download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet1k/train/"
        test_dir = "data/imagenet1k/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/imagenet100/train/"
        test_dir = "data/imagenet100/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iTinyImageNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(64),
        transforms.CenterCrop(64),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/tinyimagenet/train/"
        test_dir = "data/tinyimagenet/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iStanfordCars(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(64),
        transforms.CenterCrop(64),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "data/StanfordCars/train/"
        test_dir = "data/StanfordCars/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(
            train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iFood101(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(101).tolist()

    def download_data(self):
        data_dir = "data/food-101/"

        train_dataset = datasets.Food101(root=data_dir,
                                         split='train',
                                         download=True)
        test_dataset = datasets.Food101(root=data_dir,
                                        split='test',
                                        download=True)

        self.train_data = np.array(
            [str(x) for x in train_dataset._image_files])
        self.train_targets = np.array(train_dataset._labels)
        self.test_data = np.array([str(x) for x in test_dataset._image_files])
        self.test_targets = np.array(test_dataset._labels)


class iCUB200(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        data_dir = "data/CUB200-2011/"

        with open(osp.join(data_dir, 'images.txt')) as f:
            images = [
                osp.join(data_dir, 'images',
                         x.strip().split(' ')[-1]) for x in f.readlines()
            ]
        with open(osp.join(data_dir, 'image_class_labels.txt')) as f:
            targets = [
                int(x.strip().split(' ')[-1]) - 1 for x in f.readlines()
            ]

        self.train_data, self.train_targets = [], []
        self.test_data, self.test_targets = [], []

        with open(osp.join(data_dir, 'train_test_split.txt')) as f:
            for i, x in enumerate(f.readlines()):
                if int(x.strip().split(' ')[-1]) == 0:
                    self.train_data.append(images[i])
                    self.train_targets.append(targets[i])
                else:
                    self.test_data.append(images[i])
                    self.test_targets.append(targets[i])
        self.train_data = np.array(self.train_data)
        self.train_targets = np.array(self.train_targets)
        self.test_data = np.array(self.test_data)
        self.test_targets = np.array(self.test_targets)
