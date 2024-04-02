import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

def generate_loss_mask(labels, device="cpu"):
    result = []
    
    for label in labels:
        l = label.shape[0]
        k = label.shape[1]
        mask_ = torch.zeros(l, k)
        
        for i in range(l):
            non_zero = torch.count_nonzero(label[i, :]).item()
            if non_zero > 0:
                mask_[i] = torch.ones(k)
            else:
                mask_[i] = torch.zeros(k)

        for j in range(k):
            non_zero = torch.count_nonzero(label[:, j]).item()
            if non_zero > 0:
                mask_[:, j] = torch.ones(l)
            else:
                mask_[:, j] = torch.zeros(l)
        
        result.append(mask_.to(device))
    
    return result


def calculate_prf(predictions, labels):
    mean_f = 0
    mean_p = 0
    mean_r = 0
    for pred, label in zip(predictions, labels):
        label = torch.flatten(label)
        pred = torch.flatten(pred)
        p, r, f, _ = precision_recall_fscore_support(label, pred, average='binary')
        mean_f += f
        mean_p += p
        mean_r += r
    
    return mean_p / len(labels), mean_r / len(labels), mean_f / len(labels)