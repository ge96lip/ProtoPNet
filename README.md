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
**Note**: If conda is not automatically installed do this: 
```conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia```


**Note**: This can only be run using CUDA.

## Using ProtoPFormer
### Training a New MedMNIST ProtoPFormer

To train a new MedMNIST ProtoPNet, run the following command:
```./ProtoPFormer/scripts/train.sh model_type batch_size MedMNIST_dataset number_of_prototypes```

- `MedMNIST_dataset` can be any but should be with the 224 image size.
- `number_of_prototypes` should be `number_of_classes x prototypes_per_class`.

### Evaluating the Model

To evaluate the model, run the following command:

```./ProtoPFormer/scripts/eval.sh deit_tiny_patch16_224 MedMNIST_dataset output_view False "path_to_checkpoint" epoch-best.pth```


- `MedMNIST_dataset` should be the dataset the model was trained on.
- `path_to_checkpoint` default: `"output_cosine/dataset/deit_tiny_patch16_224/1028-adamw-0.05-100-protopformer/checkpoints"`.

## Visualization

### Prototype Visualization

The `main_visualization_v1.py` file allows for different visualization methods.

#### Closest Patches in Image Space to Learned Prototypes

```
closest_patches_info = get_closest_patches(
    ppnet, ppnet.prototype_vectors, train_loader, patch_size
)
path = os.path.join("prototype_visualizations", f"{args.data_set}_{proto_per_category}")
visualize_prototypes(
    closest_patches_info, dataset_train, path, patch_size
)```

Original Visualization Methods from the Authors
```visualize_main(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the ONE Prototype Most Important for a Class Category
```visualize_most_important_for_label(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the Prototype Influencing the Prediction the Most
```visualize_most_important_for_prediction(args, ppnet, use_train_imgs, view_loader, loader)```

Visualize the Top Three Prototypes from Every Class for Every Image from the First Batch
visualize_top_three_every_class(
    args, ppnet, use_train_imgs, view_loader, loader, 15
)


Learning a New Image Prototype from Feature Space
**Note** This method does not yet work
# Initial Patch
initial_patch = torch.rand(3, 16, 16)

# Visualize the Initial Patch
plt.imshow(v2.ToPILImage()(initial_patch))
plt.title("Initial Patch")
plt.show()

# Example Call to Synthesize Prototype
synthesize_prototype(ppnet, prototype_idx=0, patch=initial_patch)


