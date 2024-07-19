import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist import INFO
import medmnist
from src.models.XProtoNet import XProtoNet
from src.models.resnet_features import resnet50_features  # Import the resnet18 features
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os

class GrayscaleToRGB:
    def __call__(self, img):
        return img.convert("RGB")


def get_heatmap(rescaled_occurrence_maps):
    D = len(rescaled_occurrence_maps.shape)
    n_prototypes = rescaled_occurrence_maps.shape[0]
    if D == 3:
        prots_heatmaps = [
            cv2.applyColorMap(np.uint8(255 * rescaled_occurrence_maps[p]), cv2.COLORMAP_TURBO)
            for p in range(n_prototypes)
        ]
    elif D == 4:
        To = rescaled_occurrence_maps.shape[1]
        prots_heatmaps = []
        for p in range(n_prototypes):
            prots_heatmaps.append(
                np.asarray(
                    [
                        cv2.applyColorMap(
                            np.uint8(255 * rescaled_occurrence_maps[p, t]),
                            cv2.COLORMAP_TURBO,
                        )
                        for t in range(To)
                    ]
                )
            )
    prots_heatmaps = np.float32(prots_heatmaps) / 255
    prots_heatmaps = prots_heatmaps[..., ::-1]
    return prots_heatmaps


class evalx:
    def __init__(self, model_path, config, batch_size=64, output_dir='output_images'):
        self.model_path = model_path
        self.model = self.load_model(model_path, config)
        self.transform = self.get_transform()
        self.test_loader = self.get_test_loader(batch_size)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_model(self, pretrained_weights_path, config):
        model = XProtoNet(**config)  # Create an instance of your custom model
        state_dict = torch.load(pretrained_weights_path)
        model_state_dict = state_dict['state_dict']

        # Filter state dictionary to avoid mismatches
        filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)  # Load the trained weights
        model.to(torch.device('cuda'))  # Move the model to GPU
        model.eval()  # Set the model to evaluation mode
        return model

    def get_transform(self):
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    def get_test_loader(self, batch_size):
        data_flag = 'bloodmnist'
        download = True
        info = INFO[data_flag]

        DataClass = getattr(medmnist, info['python_class'])
        test_dataset = DataClass(split='test', transform=self.transform, download=download, size=224)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    def visualize_prototypes(self, image, distances, influences, prediction, image_index):
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0, 1]
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))  # Adjust for 8 subplots
        axs = axs.flatten()

        # Display the base image
        axs[0].imshow(image_np)
        axs[0].set_title(f'Base Image - Prediction: {prediction}')
        axs[0].axis('off')

        # Determine number of prototypes per class
        num_prototypes_per_class = len(influences) // 8

        # Create overlays for each class
        for class_idx in range(8):
            class_influences = influences[class_idx * num_prototypes_per_class:(class_idx + 1) * num_prototypes_per_class]
            top_indices = torch.argsort(class_influences, descending=True)[:3]  # Get top 3 influences for clarity
            combined_heatmap = self.create_combined_heatmap(top_indices, class_influences, distances)
            heatmap = get_heatmap(combined_heatmap[None, ...])[0]
            overlay = self.overlay_heatmap(image_np, heatmap)
            
            axs[class_idx + 1].imshow(overlay)
            axs[class_idx + 1].set_title(f'Overlay for Class {class_idx}')
            axs[class_idx + 1].axis('off')

        save_path = os.path.join(self.output_dir, f'image_{image_index}_prototypes.png')
        plt.savefig(save_path)
        plt.close()


    def create_combined_heatmap(self, top_indices, influences, distances):
        combined_heatmap = np.zeros((224, 224))  # Assuming heatmap size matches resized image size
        for idx in top_indices:
            prototype_idx = idx.item()
            distance_map = distances[prototype_idx].squeeze().cpu().numpy()
            distance_map_resized = F.interpolate(torch.tensor(distance_map).unsqueeze(0).unsqueeze(0),
                                                size=(224, 224), mode='bilinear', align_corners=False).squeeze().numpy()
            influence = influences[prototype_idx].item()
            combined_heatmap += influence * distance_map_resized
        combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min())
        return combined_heatmap

    def overlay_heatmap(self, image_np, heatmap):
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        overlay = 0.5 * image_np + 0.3 * heatmap_resized
        return overlay

    def evaluate(self):
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(torch.device('cuda'))  # Move images to GPU
                labels = labels.squeeze().long().to(torch.device('cuda'))
                outputs, influences, distances = self.model(images)
                predictions = torch.argmax(outputs, dim=1)

                for i in range(len(images)):
                    print(f"Image {i}:")
                    print(f"Image shape: {images[i].shape}")
                    print(f"Output: {outputs[i]}")
                    print(f"Prediction: {predictions[i]}")
                    print(f"Distance shape: {distances[i].shape}")
                    print(f"Distances: {distances[i]}")
                    self.visualize_prototypes(images[i], distances[i], influences[i], predictions[i].item(), i)


if __name__ == '__main__':
    model_path = 'ProtoASNet/logs/Image_ProtoASNet/test_run_10/epoch_21.pth'  # Update with the correct path
    # Define your configuration here
    config = {
        'features': resnet50_features(pretrained=True),
        'num_classes': 8,
        'prototype_shape': (40, 1024, 1, 1),
        'img_size': 224,
        'proto_layer_rf_info': {
            'receptive_field_size': 32,  # Example receptive field size in pixels
            'stride': 1,  # Example stride used in the prototype layer
            'padding': 0,  # Example padding used in the prototype layer
        }
    }
    evaluator = evalx(model_path=model_path, config=config)
    evaluator.evaluate()