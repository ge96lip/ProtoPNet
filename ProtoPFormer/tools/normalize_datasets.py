import os
import torch
from medmnist import INFO, Evaluator
from medmnist.dataset import PneumoniaMNIST, BreastMNIST, BloodMNIST, DermaMNIST, OCTMNIST, OrganAMNIST, OrganCMNIST, \
    OrganSMNIST, PathMNIST, RetinaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


class MedMNISTWithIndex:
    def __init__(self, medmnist_dataset):
        self.medmnist_dataset = medmnist_dataset

    def __len__(self):
        return len(self.medmnist_dataset)

    def __getitem__(self, index):
        img, target = self.medmnist_dataset[index]
        return img, target, index


def get_mean_std(loader):
    sum_ = torch.tensor(0.0)
    sum_of_squares = torch.tensor(0.0)
    num_pixels = 0

    for images, _, _ in loader:
        num_pixels += images.numel()
        sum_ += images.sum()
        sum_of_squares += (images ** 2).sum()

    mean = sum_ / num_pixels
    std = (sum_of_squares / num_pixels - mean ** 2).sqrt()

    return mean.item(), std.item()


def get_mean_std_numpy(loader):
    all_images = []
    for images, _, _ in loader:
        all_images.append(images.numpy())
    all_images = np.concatenate(all_images, axis=0)
    mean = np.mean(all_images)
    std = np.std(all_images)
    return mean, std


def main():
    data_dir = r"C:\Users\m1choelz\.medmnist"  # Folder containing .npz files
    output_file = r"C:\Users\m1choelz\.medmnist\mean_std_results.txt"  # Output file

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    datasets = {
        "PneumoniaMNIST": PneumoniaMNIST,
        "BreastMNIST": BreastMNIST,
        "BloodMNIST": BloodMNIST,
        "DermaMNIST": DermaMNIST,
        "OCTMNIST": OCTMNIST,
        "OrganAMNIST": OrganAMNIST,
        "OrganCMNIST": OrganCMNIST,
        "OrganSMNIST": OrganSMNIST,
        "PathMNIST": PathMNIST,
        "RetinaMNIST": RetinaMNIST
    }

    with open(output_file, 'w') as f:
        for dataset_name, dataset_class in datasets.items():
            try:
                dataset = dataset_class(split="train", download=True, transform=data_transforms, as_rgb=True, size=224)
                dataset_with_index = MedMNISTWithIndex(dataset)
                loader = DataLoader(dataset_with_index, batch_size=32, shuffle=True)

                # Debugging: Check the first few samples
                first_batch = next(iter(loader))
                images, targets, indices = first_batch
                print(f"First batch of {dataset_name}:")
                print(images[0])  # Print first image tensor
                print(f"Image range: {images.min().item()} to {images.max().item()}")  # Check range
                print(f"Image mean: {images.mean().item()}, Image std: {images.std().item()}")  # Check mean and std

                mean, std = get_mean_std(loader)
                mean_numpy, std_numpy = get_mean_std_numpy(loader)

                # Print the mean and std for debugging
                print(f"{dataset_name} - Mean (Torch): {mean:.5f}, Std (Torch): {std:.5f}")
                print(f"{dataset_name} - Mean (NumPy): {mean_numpy:.5f}, Std (NumPy): {std_numpy:.5f}")
                f.write(f"{dataset_name} - Mean (Torch): {mean:.5f}, Std (Torch): {std:.5f}\n")
                f.write(f"{dataset_name} - Mean (NumPy): {mean_numpy:.5f}, Std (NumPy): {std_numpy:.5f}\n")
            except Exception as e:
                f.write(f"{dataset_name} - Error: {e}\n")
                print(f"{dataset_name} - Error: {e}")


if __name__ == '__main__':
    main()
