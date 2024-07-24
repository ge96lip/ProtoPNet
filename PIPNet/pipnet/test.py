from tqdm import tqdm
import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score

@torch.no_grad()
def test_pipnet(net, test_loader: DataLoader) -> None:
    # Determine the device from the model
    device = next(net.parameters()).device
    
    # Ensure the model is in evaluation mode
    net.eval()
    
    # Initialize information dictionary
    info = dict()
    
    # Build a confusion matrix
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)

    global_top1acc = 0.
    global_top5acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader), total=len(test_loader), desc='Evaluating', mininterval=5., ncols=0)
    
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.))
            _, pooled, out = net(xs, inference=True)
            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred_scores = torch.amax(F.softmax(torch.log1p(out ** net.module._classification.normalization_multiplier), dim=1), dim=1)
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
            repeated_weight = net.module._classification.weight.unsqueeze(1).repeat(1, pooled.shape[0], 1)
            sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(pooled * repeated_weight), 1e-3).float(), dim=2).float()
            local_size = torch.count_nonzero(torch.gt(torch.relu((pooled * repeated_weight) - 1e-3).sum(dim=1), 0.).float(), dim=1).float()
            local_size_total += local_size.sum().item()
            
            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(sim_scores_anz, dim=0, index=ys_pred), 0)
            global_sim_anz += correct_class_sim_scores_anz.sum().item()
            
            almost_nz = torch.count_nonzero(torch.gt(torch.abs(pooled), 1e-3).float(), dim=1).float()
            global_anz += almost_nz.sum().item()
            
            cm_batch = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)
            for y_pred, y_true in zip(ys_pred, ys):
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
            acc = acc_from_cm(cm_batch)
            test_iter.set_postfix_str(f'SimANZCC: {correct_class_sim_scores_anz.mean().item():.2f}, ANZ: {almost_nz.mean().item():.1f}, LocS: {local_size.mean().item():.1f}, Acc: {acc:.3f}', refresh=False)

            top1accs, top5accs = topk_accuracy(out, ys, topk=[1, 5])
            global_top1acc += torch.sum(top1accs).item()
            global_top5acc += torch.sum(top5accs).item()
            y_preds += ys_pred_scores.detach().tolist()
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()
        
        del out
        del pooled
        del ys_pred
        
    print("PIP-Net abstained from a decision for", abstained.item(), "images", flush=True)
    info['num non-zero prototypes'] = torch.gt(net.module._classification.weight, 1e-3).any(dim=0).sum().item()
    print("Sparsity ratio: ", (torch.numel(net.module._classification.weight) - torch.count_nonzero(torch.nn.functional.relu(net.module._classification.weight - 1e-3)).item()) / torch.numel(net.module._classification.weight), flush=True)
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['top1_accuracy'] = global_top1acc / len(test_loader.dataset)
    info['top5_accuracy'] = global_top5acc / len(test_loader.dataset)
    info['almost_sim_nonzeros'] = global_sim_anz / len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total / len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz / len(test_loader.dataset)

    if net.module._num_classes == 2:
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        print("TP: ", tp, "FN: ", fn, "FP:", fp, "TN:", tn, flush=True)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("Confusion matrix: ", cm, flush=True)
        try:
            for classname, classidx in test_loader.dataset.class_to_idx.items():
                if classidx == 0:
                    print("Accuracy positive class (", classname, classidx, ") (TPR, Sensitivity):", tp / (tp + fn))
                elif classidx == 1:
                    print("Accuracy negative class (", classname, classidx, ") (TNR, Specificity):", tn / (tn + fp))
        except ValueError:
            pass
        print("Balanced accuracy: ", balanced_accuracy_score(y_trues, y_preds_classes), flush=True)
        print("Sensitivity: ", sensitivity, "Specificity: ", specificity, flush=True)
        print("F1 Score: ", f1_score(y_trues, y_preds_classes), flush=True)
        try:
            print("AUC macro: ", roc_auc_score(y_trues, y_preds, average='macro'), flush=True)
            print("AUC weighted: ", roc_auc_score(y_trues, y_preds, average='weighted'), flush=True)
        except ValueError:
            pass
    else:
        print("Top-5 Accuracy: ", global_top5acc / len(test_loader.dataset))

    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
