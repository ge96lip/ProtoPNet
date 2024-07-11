import matplotlib.pyplot as plt
import sys
import os
# Add the ProtoPFormer directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ProtoPFormer')))


import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize

from tools.datasets import build_dataset_noaug
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.utils.data
import protopformer
import argparse

class Args:
    def __init__(self, num_classes, batch_size, num_workers, pin_mem, drop_last, base_architecture, img_size, reserve_layers, reserve_token_nums, input_size, dataset, model, modeldir, prototype_shape):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_mem = pin_mem
        self.drop_last = drop_last
        self.base_architecture = base_architecture
        self.img_size = img_size
        self.reserve_layers = [reserve_layers]
        self.reserve_token_nums = [reserve_token_nums]
        self.input_size = input_size
        self.model = model
        self.num_classes = num_classes
        self.data_set = dataset
        self.data_path = "..\\datasets"
        self.modeldir = modeldir
        self.prototype_shape = prototype_shape
        if model == "deit_small_patch16_224":
            self.reserve_layer_idx=11
        elif model == "deit_tiny_patch16_224":
            self.reserve_layer_idx=11
        elif model == "cait_xxs24_224":
            self.reserve_layer_idx=1

# Define a function to calculate MCC
def calculate_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def calculate_auc(true_labels, predicted_probs, num_classes):
    true_labels = torch.nn.functional.one_hot(torch.tensor(true_labels), num_classes=num_classes).numpy()
    auc = roc_auc_score(true_labels, predicted_probs, multi_class='ovr')
    return auc

def calculate_auc_med(true_labels, predicted_probs, num_classes=2):
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels)

    true_labels = true_labels.view(-1).long()
    true_labels_one_hot = torch.nn.functional.one_hot(true_labels, num_classes=num_classes).numpy()

    if isinstance(predicted_probs, list):
        predicted_probs = np.array(predicted_probs)

    if predicted_probs.shape[1] != num_classes:
        raise ValueError(
            f"predicted_probs should have shape (n_samples, {num_classes}), but got shape {predicted_probs.shape}")

    auc_scores = []
    for i in range(num_classes):
        class_predicted_probs = predicted_probs[:, i]

        if np.unique(true_labels_one_hot[:, i]).size > 1:
            auc = roc_auc_score(true_labels_one_hot[:, i], class_predicted_probs)
            auc_scores.append(auc)
        else:
            print(f"Skipping AUC calculation for class {i}: Not enough classes present")

    if auc_scores:
        return np.mean(auc_scores)
    else:
        return None

def calculate_accuracy(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

def mathew_analysis(args):
    dataset_val, _ = build_dataset_noaug(is_train=False, args=args)

    test_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False
    )

    load_model_path = os.path.join(args.modeldir, args.model[0])

    ppnet = protopformer.construct_PPNet(base_architecture=args.base_architecture,
                                         pretrained=True, img_size=args.img_size,
                                         prototype_shape=args.prototype_shape,
                                         num_classes=args.num_classes,
                                         reserve_layers=args.reserve_layers,
                                         reserve_token_nums=args.reserve_token_nums,
                                         use_global=True,
                                         use_ppc_loss=True,
                                         global_coe=0.5,
                                         global_proto_per_class=3,
                                         prototype_activation_function='log',
                                         add_on_layers_type='regular')
    print('load model from ' + load_model_path)
    load_model = torch.load(load_model_path, map_location='cuda:0')
    checkpoint = torch.load(load_model_path)
    state_dict = checkpoint['model']

    if 'model' in load_model.keys():
        print("load correct")
        ppnet.load_state_dict(state_dict, strict=False)
    else:
        ppnet.load_state_dict(load_model, strict=False)
    ppnet = ppnet.cuda()
    ppnet.eval()

    num_runs = 5
    mcc_scores, accuracy_scores, auc_scores = [], [], []

    for _ in range(num_runs):
        true_labels, predicted_labels, predicted_probs = [], [], []

        with torch.no_grad():
            for data, target, _ in test_loader:
                data, target = data.cuda(), target.cuda()
                output = ppnet(data)[0]
                probabilities = torch.softmax(output, dim=1)

                _, predicted = torch.max(output, dim=1)

                true_labels.extend(target.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                predicted_probs.extend(probabilities.cpu().numpy())

        mcc = calculate_mcc(true_labels, predicted_labels)
        accuracy = calculate_accuracy(true_labels, predicted_labels)
        auc = calculate_auc_med(true_labels, predicted_probs, num_classes=args.num_classes)

        mcc_scores.append(mcc)
        accuracy_scores.append(accuracy)
        auc_scores.append(auc)

    mean_mcc = sum(mcc_scores) / num_runs
    mean_accuracy = sum(accuracy_scores) / num_runs
    mean_auc = sum(auc_scores) / num_runs

    print(f"Mean test MCC over {num_runs} runs: {mean_mcc}")
    print(f"Mean test Accuracy over {num_runs} runs: {mean_accuracy}")
    print(f"Mean test AUC over {num_runs} runs: {mean_auc}")

def main():
    parser = argparse.ArgumentParser(description='ProtoPFormer Analysis Script')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--prototype_shape', type=int, required=True, help='Prototype shape (1 integers)')
    parser.add_argument('--modeldir', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--batch_size', type=int, default=60, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--pin_mem', type=bool, default=True, help='Pin memory')
    parser.add_argument('--drop_last', type=bool, default=True, help='Drop last batch')
    parser.add_argument('--base_architecture', type=str, default='deit_tiny_patch16_224', help='Base architecture')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--reserve_layers', type=int, default=11, help='Reserve layers')
    parser.add_argument('--reserve_token_nums', type=int, default=81, help='Reserve token numbers')
    parser.add_argument('--input_size', type=int, default=224, help='Input size')
    parser.add_argument('--dataset', type=str, default="BloodMNIST", help='Name of Dataset')

    args = parser.parse_args()

    args = Args(
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        drop_last=args.drop_last,
        base_architecture=args.base_architecture,
        img_size=args.img_size,
        reserve_layers=args.reserve_layers,
        reserve_token_nums=args.reserve_token_nums,
        input_size=args.input_size,
        model=[args.model],
        modeldir=args.modeldir,
        prototype_shape=[args.prototype_shape, 192, 1, 1], 
        dataset=args.dataset
    )

    mathew_analysis(args)

if __name__ == '__main__':
    main()
