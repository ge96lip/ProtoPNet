import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms
import matplotlib.pyplot as plt

class ULT(Dataset):
    """ Medical Images Dataset.
    Args:
        root (string): Root directory of dataset where directories
            'train' and 'test' are located.
        train (bool, optional): If true, creates dataset from training set, otherwise from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, split='val', train=True, train_ratio=0.8, val_ratio=0.1, transform=None,
                 target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split = split
        self.data_folder = 'train' if self.train else 'test'
        self.classes = ["benign", "malignant", "normal"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"Initializing ULT dataset with root: {self.root}")
        self.samples = self.load_samples()

    def split_data(self, samples):
        random.shuffle(samples)
        num_samples = len(samples)
        num_train = int(num_samples * self.train_ratio)
        num_val = int(num_samples * self.val_ratio)

        train_samples = samples[:num_train]
        val_samples = samples[num_train:num_train + num_val]
        test_samples = samples[num_train + num_val:]
        return train_samples, val_samples, test_samples

    def load_samples(self):
        samples = []
        for class_name in self.classes:
            class_index = self.class_to_idx[class_name]
            class_path = os.path.join(self.root, class_name)
            print(f"Loading images from: {class_path}")
            if not os.path.exists(class_path):
                print(f"Directory does not exist: {class_path}")
                continue
            for image_name in os.listdir(class_path):
                if image_name.endswith('.png') and '_mask' not in image_name:
                    image_path = os.path.join(class_path, image_name)
                    samples.append((image_path, class_index))

        # If the split is specified to handle separate subsets
        if self.split == 'all':
            return samples

        if self.train:
            return self.split_data(samples)[0]
        else:
            if self.split == 'test':
                return self.split_data(samples)[2]
            else:
                return self.split_data(samples)[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the class index of the image.
        """
        image_path, target_class = self.samples[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def stats(self):
        counts = [0] * len(self.classes)
        for _, target_class in self.samples:
            counts[target_class] += 1

        for cls, count in zip(self.classes, counts):
            print(f"{cls}: {count} images")

        total_images = sum(counts)
        print(
            f"{total_images} samples spanning {len(self.classes)} classes (avg {total_images / len(self.classes):.2f} per class)")

        return dict(zip(self.classes, counts))


def build_dataset(is_train, args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if args.data_set == 'ULT':
        dataset = ULT(root=os.path.join(args.data_path, 'ultrasound'), train=is_train,
                      transform=transform)
        nb_classes = 3

    return dataset, nb_classes

class Ult:
    def __init__(self):
        class Args:
            data_set = 'ULT'
            data_path = '/vol/aimspace/projects/practical_SoSe24/interpretable_models/ProtoASNet/data/'  # Replace with the correct path

        self.args = Args()
        self.batch_size = 32

        self.train_dataset, _ = build_dataset(is_train=True, args=self.args)
        self.val_dataset, _ = build_dataset(is_train=False, args=self.args)
        self.test_dataset = ULT(root=os.path.join(self.args.data_path, 'ultrasound'), train=False, split='test', transform=self.train_dataset.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def print_stats(self):
        print("Train Dataset Stats:")
        self.train_dataset.stats()

        print("\nValidation Dataset Stats:")
        self.val_dataset.stats()

        print("\nTest Dataset Stats:")
        self.test_dataset.stats()

    def show_samples(self, loader, title):
        batch = next(iter(loader))
        images, labels = batch
        fig, axes = plt.subplots(1, 5, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        for i in range(5):
            image = images[i].permute(1, 2, 0).numpy()
            image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
            image = (image * 255).astype('uint8')
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(self.train_dataset.classes[labels[i].item()])
        plt.show()

    def display_samples(self):
        self.show_samples(self.train_loader, "Train Samples")
        self.show_samples(self.val_loader, "Validation Samples")
        self.show_samples(self.test_loader, "Test Samples")

def get_loader(data_class, mode):
    data = data_class()

    if mode == "train":
        loader = data.train_loader
    elif mode == "val":
        loader = data.val_loader
    else:
        loader = data.test_loader
    return loader

if __name__ == "__main__":
    ult = Ult()
    ult.print_stats()
    ult.display_samples()

    # Example usage to get a specific loader:
    train_loader = get_loader(Ult, "train")
    val_loader = get_loader(Ult, "val")
    test_loader = get_loader(Ult, "test")

    # Now you can use train_loader, val_loader, and test_loader as needed.
