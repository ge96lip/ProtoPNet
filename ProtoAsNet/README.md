# ProtoASNet
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

1. Clone the repo

```bash
git clone https://github.com/Tugcannn/ProtoAsNet.git
cd ProtoASNet
```
2. Place your data in the `data` folder. Download Ultrasound dataset and put it inside 'data' folder, [Link is here](https://drive.google.com/drive/folders/18Pzqy-Zy3_aPRwAVp5UT9GYoLpXTK2rv?usp=sharing)

3. Also, we already trained Bloodmnist model for 3-5-10 prototypes. Also, you can find in the driver and put it inside 'logs/Image_ProtoAsNet' [Link is here](https://drive.google.com/drive/folders/189V5SGVvBsm-8idoU-XEFwRcFXDbgvmQ?usp=sharing)

4. Python library dependencies can be installed using:

```bash
pip install --upgrade pip
pip install torch torchvision  
pip install pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab imageio array2gif moviepy scikit-image scikit-learn torchmetrics termplotlib
pip install -e .
# sanity check 
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```


--------------------------------------------------------------------------------------------------------
## Train and Test

To train the model `cd` to the project folder, then use the command `python main.py` with the arguments described below:

- Before training you need to check

- "src/configs/Ours_ProtoASNet_Image.yml"`: yaml file containing hyper-parameters for model, experiment, loss objectives.

- You can change model architecture, epoch, num_classes and prototype_shape.

- "src/data/as_dataloader" : Implement class labels and dataset (data = MedMnist or data = Ult())

- After control these files run "main.py". You can follow backlogs

- "logs/<path-to-save>"` the folder to save all the trained model checkpoints, evaluations, and visualization of learned prototypes, you can try last model which is best validation accuracy.

**Dataset Checking and Evaluation**

*medmnist*

- "src/data/medMnistx.py" python file dataloading for medmnist dataset, you can try different medmnist datasets, batch sizes, normalizations and checking datasets.

- "src/data/evalx.py" evaluating the model on the test dataset, Computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

- "src/data/explain_images.py" visualizing the prototypes and their influences on the model's predictions and saving the visualization results.

- "src/data/explain_images_without_names.py" visualizing the prototypes and their influences on the model's predictions and saving the visualization results without real names just (category 0,1,2..).

**Carefull**
- Need to write correct model_path, num_class and prototype shape in "evalx.py" and "explain_images.py"
- Use for ex: test_run_01 and epoch_30.pth 

*ultrasound*

- "src/data/ultrasoundload.py" python file dataloading for ultrasound dataset

- "src/data/ultraeval.py" evaluating the model on the ultrasound test dataset, Computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

### outputs 

the important content saved in save_dir folder are:

- `model_best.pth`: checkpoint of the last model which is best model based on a metric of ACC


--------------------------------------------------------------------------------------------------------
## Description of files and folders

### logs
Once you run the system, it will contain the saved models, logs, and evaluation results (visualization of explanations, etc)

### pretrained_models
When training is done for the first time, pretrained backbone models are saved here.

### src
- `agents/`: folder containing agent classes for each of the architectures. contains the main framework for the training process
- `configs/`: folder containing the yaml files containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations.
- `data/`: folder for dataset, dataloader classes, evaluations and explanations
- `loss/`: folder for loss functions
- `models/`: folders for model architectures
- `utils/`: folder for some utility scripts and local explanation 



