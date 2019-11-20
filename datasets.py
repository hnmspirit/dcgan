import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def load_celeba(dataroot, transform):
    dataset = dset.CelebA(root=dataroot, transform=transform, download=True)
    return dataset


def load_loader(dataset, batch_size, workers):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
    return dataloader
