import os
import medmnist
from medmnist import INFO, Evaluator
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np

# Define the transformation for training and validation 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) #imagenet normalization
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

data_flag = 'bloodmnist' #try different dataset
download = True

BATCH_SIZE = 64

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

class MedMnist:
    # load the data
    train_dataset = DataClass(split='train', transform=transform, download=download, as_rgb=True, size=224)
    val_dataset = DataClass(split='val', transform=transform, download=download, as_rgb=True, size=224)
    test_dataset = DataClass(split='test', transform=transform, download=download, as_rgb=True, size=224)

    # Encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)   

def show_images(data_loader):
    batch = next(iter(data_loader))
    images, labels = batch
    
    plt.figure(figsize=(10, 10))
    grid_size = 4 
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, i + 1)
        image = images[i].permute(1, 2, 0).numpy()  
        image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] 
        image = image.clip(0, 1)
        plt.imshow(image)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()



if __name__ == "__main__":
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    datasets = MedMnist
    print(datasets.train_dataset)
    """
    print("===================")
    show_images(a.train_loader)
   """
    count = [0,0,0,0,0,0,0,0,0,0,0,0]
    for i in datasets.train_dataset:
            count[int(i[1])] += 1
    print(count)

