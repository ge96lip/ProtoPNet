import medmnist
from medmnist import INFO
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


#choose the right mnist in order to get the correct class names.

info = INFO["breastmnist"] 

class_names = info["label"]
local_npz_path = 'Dataset/breastmnist_224.npz' 

output_path = "Dataset/breastmnist_224"

# Paths

# Load dataset
data = np.load(local_npz_path)
train_images, train_labels = data['train_images'], data['train_labels']
val_images, val_labels = data['val_images'], data['val_labels']
test_images, test_labels = data['test_images'], data['test_labels']

# Create directories
for subset in ['train', 'val', 'test']:
    for class_name in class_names.values():
        class_dir = os.path.join(output_path, subset, class_name)
        os.makedirs(class_dir, exist_ok=True)

# Function to save images to appropriate folders
def save_images(images, labels, subset):
    for i in tqdm(range(len(images))):
        class_name = class_names[str(labels[i][0])]
        output_dir = os.path.join(output_path, subset, class_name)
        image_path = os.path.join(output_dir, f'{i}.png')
        plt.imsave(image_path, images[i], cmap='gray')

# Save train images
save_images(train_images, train_labels, 'train')

# Save validation images
save_images(val_images, val_labels, 'val')

# Save test images
save_images(test_images, test_labels, 'test')

print("Dataset organized successfully!")