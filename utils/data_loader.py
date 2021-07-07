import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from torch.utils.data import Dataset
from utils.fashion_mnist import MNIST, FashionMNIST
import pandas as pd
import os
import PIL

class ZahnerDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #why not: image = read_image(img_path) as in ?
        # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
        # ??? would require: from torchvision.io import read_image
        # instead of PIL
        image = PIL.Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, split='train', download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot,  split='test', download=args.download, transform=trans)
        
    elif args.dataset == 'zahner':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = ZahnerDataset('datasets/zahner/annotations.csv', 'datasets/zahner/data', transform=trans)
        test_dataset = ZahnerDataset('datasets/zahner/annotations.csv', 'datasets/zahner/data', transform=trans)
        
    elif args.dataset == 'zahner_64':
        trans = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, )),
        ])
        train_dataset = ZahnerDataset('datasets/zahner/annotations_64.csv', 'datasets/zahner/data_64', transform=trans)
        test_dataset = ZahnerDataset('datasets/zahner/annotations_64.csv', 'datasets/zahner/data_64', transform=trans)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
