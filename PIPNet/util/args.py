import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args():
    parser = argparse.ArgumentParser(description='Train a PIP-Net')
    parser.add_argument('--dataset', type=str, help='Dataset on which PIP-Net should be trained')
    parser.add_argument('--validation_size', type=float, help='Split between training and validation set.')
    parser.add_argument('--net', type=str, help='Base network used as backbone of PIP-Net.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_pretrain', type=int, help='Batch size for pretraining the prototypes.')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--epochs_pretrain', type=int, help='Number of pre-training epochs.')
    parser.add_argument('--optimizer', type=str, help='Type of optimizer.')
    parser.add_argument('--lr', type=float, help='Learning rate for training the prototypes to classes.')
    parser.add_argument('--lr_block', type=float, help='Learning rate for the last convolutional layers of the backbone.')
    parser.add_argument('--lr_net', type=float, help='Learning rate for the backbone network.')
    parser.add_argument('--weight_decay', type=float, help='Weight decay rate used in the optimizer.')
    parser.add_argument('--disable_cuda', action='store_true', help='Flag to disable CUDA.')
    parser.add_argument('--log_dir', type=str, help='Directory where training progress and logs will be saved.')
    parser.add_argument('--num_features', type=int, help='Number of prototypes.')
    parser.add_argument('--image_size', type=int, help='Size to which input images are resized.')
   # parser.add_argument('--state_dict_dir_net', type=str, help='The directory containing a state dict with a pretrained PIP-Net.')
    parser.add_argument('--freeze_epochs', type=int, help='Number of epochs to freeze the pretrained features.')
    parser.add_argument('--dir_for_saving_images', type=str, help='Directory for saving output images and prototypes.')
    parser.add_argument('--disable_pretrained', action='store_true', help='Disable using pretrained weights.')
    parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss based on class balance.')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility.')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs for training.')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading.')
    parser.add_argument('--bias', action='store_true', help='Include a trainable bias in the linear classification layer.')
    parser.add_argument('--extra_test_image_folder', type=str, default='./experiments', help='Folder with images that PIP-Net will predict and explain, that are not in the training or test set. E.g. images with 2 objects or OOD image. Images should be in subfolder. E.g. images in ./experiments/images/, and argument --./experiments')

    args = parser.parse_args()
    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    
def get_optimizer_nn(net, args: argparse.Namespace) -> torch.optim.Optimizer:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name,param in net.module._net.named_parameters():
            if 'layer4.2' in name:
                params_to_train.append(param)
            elif 'layer4' in name or 'layer3' in name:
                params_to_freeze.append(param)
            elif 'layer2' in name:
                params_backbone.append(param)
            else: #such that model training fits on one gpu. 
                param.requires_grad = False
                # params_backbone.append(param)
    
    elif 'convnext' in args.net:
        print("chosen network is convnext", flush=True)
        for name,param in net.module._net.named_parameters():
            if 'features.7.2' in name: 
                params_to_train.append(param)
            elif 'features.7' in name or 'features.6' in name:
                params_to_freeze.append(param)
            # CUDA MEMORY ISSUES? COMMENT LINE 202-203 AND USE THE FOLLOWING LINES INSTEAD
            # elif 'features.5' in name or 'features.4' in name:
            #     params_backbone.append(param)
            # else:
            #     param.requires_grad = False
            else:
                params_backbone.append(param)
    else:
        print("Network is not ResNet or ConvNext.", flush=True)     
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": net.module._add_on.parameters(), "lr": args.lr_block*10., "weight_decay_rate": args.weight_decay}]
            
    paramlist_classifier = [
            {"params": classification_weight, "lr": args.lr, "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, "lr": args.lr, "weight_decay_rate": 0},
    ]
          
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(paramlist_net,lr=args.lr,weight_decay=args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(paramlist_classifier,lr=args.lr,weight_decay=args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    else:
        raise ValueError("this optimizer type is not implemented")

