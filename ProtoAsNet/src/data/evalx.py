import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from medmnist import INFO
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_curve, auc, accuracy_score, roc_auc_score
import numpy as np
import medmnist
import os
from src.models.XProtoNet import XProtoNet
from src.models.resnet_features import resnet18_features, resnet50_features  # Import the resnet18/50 features
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class evalx:
    def __init__(self, model_path, config, batch_size=64):
        self.model_path = model_path
        self.model = self.load_model(model_path, config)
        self.transform = self.get_transform()
        self.test_loader = self.get_test_loader(batch_size)
    
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_test_loader(self, batch_size):
        data_flag = 'bloodmnist'
        download = True
        BATCH_SIZE = 64
        info = INFO[data_flag]

        DataClass = getattr(medmnist, info['python_class'])
        test_dataset = DataClass(split='test', transform=self.transform, download=download, as_rgb=True, size=224)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    def evaluate(self):
        true_labels = []
        pred_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(torch.device('cuda'))  # Move images to GPU
                labels = labels.squeeze().long().to(torch.device('cuda'))
                outputs, _, _ = self.model(images)
                logits = outputs  # Extract logits from the outputs tuple
                _, predicted = torch.max(logits, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
    
        
        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        all_probs = np.array(all_probs)
        
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        
        # Get the labels 
        labels = ["0", "1", "2", "3", "4", "5", "6", "7"]

       # Calculate the classification report
        class_report = classification_report(true_labels, pred_labels, target_names=labels, zero_division="warn")
        
        # Calculate MCC
        mcc = matthews_corrcoef(true_labels, pred_labels)
        
        # Calculate ROC-AUC for each class
        roc_auc_dict = {}
        for i in range(len(labels)):
            fpr, tpr, _ = roc_curve(true_labels == i, all_probs[:, i])
            roc_auc_dict[labels[i]] = auc(fpr, tpr)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Calculate average ROC-AUC score
        avg_roc_auc = np.mean(list(roc_auc_dict.values()))

        #auc_score = roc_auc_score(true_labels, pred_labels) #for binary        
        auc_score = roc_auc_score(true_labels, all_probs, multi_class="ovr") #multiclass classification

        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print("AUC Score:", auc_score)
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average ROC-AUC: {avg_roc_auc:.4f}")
        print("\nROC-AUC Scores:")
        for label, score in roc_auc_dict.items():
            print(f"{label}: {score:.4f}")

if __name__ == '__main__':
    model_path = 'ProtoASNet/logs/Image_ProtoASNet/test_run_11/epoch_7.pth'  # Update with the correct path
    # Load the saved model
    # Define your configuration here
  
    config = {
    'features': resnet50_features(pretrained=True),
    'num_classes': 8,
    'prototype_shape': (80, 1024, 1, 1), # (6,512,1,1) -> resnet18
    'img_size': 224,
    'proto_layer_rf_info': {
        'receptive_field_size': 32,  # Example receptive field size in pixels
        'stride': 1,  # Example stride used in the prototype layer
        'padding': 0,  # Example padding used in the prototype layer
        }
    }
    evaluator = evalx(model_path=model_path, config=config)
    evaluator.evaluate()
