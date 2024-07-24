# ProtoPNet

This GitHub repository contains the code implementation from the following analysis: [link]

There are three different ProtoPNets namely:
- ProtoPFormer
- ProtoPiPNet
- ProtoASNet

## Getting Started

### Cloning the Repository

First, clone the repository:

```git clone```


### Setting Up the Environment

To ensure everything runs smoothly, please first install the environment as named below:

```conda env create -f environment.yml```

**Note**: If conda is not automatically, it can be manually installed:

```conda remove pytorch torchvision torchaudio```

```conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia```

```pip install timm==0.4.12```


**Note**: This can only be run using CUDA.

## Using ProtoPFormer
### Training a New MedMNIST ProtoPFormer

To train a new MedMNIST ProtoPNet, run the following command:

```./ProtoPFormer/scripts/train.sh model_type batch_size MedMNIST_dataset number_of_prototypes```

- `MedMNIST_dataset` can be any but should be with the 224 image size.
- `number_of_prototypes` should be `number_of_classes x prototypes_per_class`.

### Evaluating the Model

To evaluate the model, run the following command: 

```CUDA_VISIBLE_DEVICES=1 python ./ProtoPFormer/analysis.py --num_classes x --prototype_shape xx --modeldir "/path/to/model/checkpoints" --model "epoch-best.pth" --dataset "name_dataset"```

**Note**: CUDA_VISIBLE_DEVICE=x not needed but the analysis does require a lot of GPU Memory therefore it is recommended choosing an idle GPU 


### Visualization of Prototypes

To visualize the prototypes of the model, run the following command:

```./ProtoPFormer/scripts/eval.sh deit_tiny_patch16_224 MedMNIST_dataset output_view False "path_to_checkpoint" epoch-best.pth```

- `MedMNIST_dataset` should be the dataset the model was trained on.
- `path_to_checkpoint` default: `"output_cosine/dataset/deit_tiny_patch16_224/1028-adamw-0.05-100-protopformer/checkpoints"`.

## Variations of Visualization

### Prototype Visualization

The `main_visualization_v1.py` file allows for different visualization methods.

Closest Patches in Image Space to Learned Prototypes

```
closest_patches_info = get_closest_patches(
    ppnet, ppnet.prototype_vectors, train_loader, patch_size)
path = os.path.join("prototype_visualizations", f"{args.data_set}_{proto_per_category}")
visualize_prototypes(
    closest_patches_info, dataset_train, path, patch_size
)
```

Original Visualization Methods from the Authors
```visualize_main(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the ONE Prototype Most Important for a Class Category
```visualize_most_important_for_label(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the Prototype Influencing the Prediction the Most
```visualize_most_important_for_prediction(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the Top Three Prototypes from Every Class for Every Image from the First Batch
```
visualize_top_three_every_class(
    args, ppnet, use_train_imgs, view_loader, loader, 15
)
```


Learning a New Image Prototype from Feature Space
**Note** This method does not yet work

```
# Initial Patch
initial_patch = torch.rand(3, 16, 16)

# Visualize the Initial Patch
plt.imshow(v2.ToPILImage()(initial_patch))
plt.title("Initial Patch")
plt.show()

# Example Call to Synthesize Prototype
synthesize_prototype(ppnet, prototype_idx=0, patch=initial_patch)
```

## Using ProtoAsNet
--------------------------------------------------------------------------------------------------------
## Contents
- [Introduction](#Introduction)
- [Environment Setup](#Environment-Setup)
- [Train and Test](#Train-and-Test)
- [Description of Files and Folders](#Description-of-Files-and-Folders)


## Introduction 

ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography for medmnist and ultrasound dataset.

--------------------------------------------------------------------------------------------------------
## Environment Setup

1. Place your data in the `data` folder. Download Ultrasound dataset and put it inside 'data' folder, [Link is here](https://drive.google.com/drive/folders/18Pzqy-Zy3_aPRwAVp5UT9GYoLpXTK2rv?usp=sharing)

2. Also, we already trained Bloodmnist model for 3-5-10 prototypes. Also, you can find in the driver and put it inside 'logs/Image_ProtoAsNet' [Link is here](https://drive.google.com/drive/folders/189V5SGVvBsm-8idoU-XEFwRcFXDbgvmQ?usp=sharing)


--------------------------------------------------------------------------------------------------------
## Train and Test

To train the model `cd` to the project folder, then use the command `python main.py` with the arguments described below:

- Before training you need to check

- "src/configs/Ours_ProtoASNet_Image.yml"`: Yaml file containing hyper-parameters for model, experiment, and loss objectives.

- You can change model architecture, epoch, num_classes, and prototype_shape. Default is ResNet50. You NEED to adapt num_classes and prototype_shape according to your chosen dataset. 

- "src/data/as_dataloader" : Implement class labels and dataset (data = MedMnist or data = Ult())

- "src/data/medMnistx.py" python file dataloading for medmnist dataset, you can try different medmnist datasets, batch sizes, normalizations and checking datasets.

- After controlling these files run "main.py" for training. You can follow in backlogs

- "logs/<path-to-save>"` the folder created automatically to save all the trained model checkpoints, evaluations, and visualization of learned prototypes, you can try the last model which is the best validation accuracy.

**Dataset Checking and Evaluation**

*medmnist*

- "src/data/medMnistx.py" python file data loading for medmnist dataset, you can try different medmnist datasets, batch sizes, normalizations and checking datasets.

- "evaluation.py" evaluates the model on the test dataset, and computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

**Visualization of Images**

- "explain_images.py" visualizes the prototypes and their influences on the model's predictions and saves the visualization results in the output_images folder.

- "src/data/explain_images_without_names.py" visualizing the prototypes and their influences on the model's predictions and saving the visualization results without real names just (category 0,1,2..).

**Careful**
- Need to write correct model_path, num_class and prototype shape in "evaluation.py" and "explain_images.py"

*ultrasound*

- "src/data/ultrasoundload.py" python file data loading for ultrasound dataset

- "src/data/ultraeval.py" evaluating the model on the ultrasound test dataset, computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

### outputs 

the important content saved in the logs/save_dir folder are:

- `model_best.pth`: checkpoint of the last model which is the best model based on a metric of ACC

**NOTE**

If you run into the error `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` please try and change the running CUDA device. 
E.g. run: `CUDA_VISIBLE_DEVICES=1 python main.py`

--------------------------------------------------------------------------------------------------------
## Description of files and folders

### logs
Once you run the system, it will contain the saved models, logs, and evaluation results (visualization of explanations, etc)

### pretrained_models
When training is done for the first time, pretrained backbone models are saved here automatically.

### src
- `agents/`: folder containing agent classes for each of the architectures. contains the main framework for the training process
- `configs/`: folder containing the yaml files containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations.
- `data/`: folder for dataset, dataloader classes, evaluations and explanations
- `loss/`: folder for loss functions
- `models/`: folders for model architectures
- `utils/`: folder for some utility scripts and local explanation 

## Using PIPNET

README:
The code submitted is mainly similar to the original publication. However it has been adapted for the medmnist dataset. See following instructions in order to run the code.

- DATASET
  The way the script has been formulated is that every folder has to be downloaded and organized according to the splits from medmnist. In the beginning of the main function, please specify where you want the folder with the images to be and specificy where the .npz file is. Otherwise the process is automatic and running the code will start off with running and creating the dataset.

  See the data.py and data_test.py files for changing the names of the paths themselves. The script is now adapted for the breastmnist and everything works accordingly to that. If there is another dataset used, change the paths. The data.py and data_test.py need to have the correct paths. See the current paths to understand.

- Run
  When running the code, copy the .txt from the used_arguments folder. Paste them in the terminal. It is necessary that the you are located inside the PIPNet. The necessary specifications are in there. For more information see args.py for what each mean. Change the args to you liking. However remember to always create a new folder for every run.

- Analysis
  When you have ran the entire code in you output you will get several files. The ones added are the visualisations of the prototypes for the best model and the visualisations for the test images for the best model. With similar structure as the folders for the validation set. For more information see the old read me. The folder with all the rpedictions and their explanations are called "visualisation_results". The best model is automatically produced after the run is completed.

For more information see the old readme. This is a condensed version. The old read me is in the PIPNet folder.
