import numpy as np
import utils

from torch import nn
import torch
import torch.nn.functional as F
from metrics import AllInOneMeter
import time
import torchvision.transforms as transforms


def validation_binary(model: nn.Module, criterion, valid_loader, device, num_classes=None):
    with torch.no_grad():
        #model.eval()
        #losses1 = []
        #losses2 = []
        #jaccard = []
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        meter = AllInOneMeter()
        start_time = time.time()
        w1 = 1.0
        w2 = 0.5
        w3 = 0.5
        for valid_image, valid_mask, valid_mask_ind in valid_loader:
            valid_image = valid_image.to(device)  # [N, 1, H, W]
            valid_mask  = valid_mask.to(device).type(torch.cuda.FloatTensor)
            valid_image = valid_image.permute(0, 3, 1, 2)
            valid_mask  = valid_mask.permute(0, 3, 1, 2)
            valid_mask_ind = valid_mask_ind.to(device).type(torch.cuda.FloatTensor)

            outputs, outputs_mask_ind1, outputs_mask_ind2 = model(valid_image)
            valid_prob = F.sigmoid(outputs)
            valid_mask_ind_prob1 = F.sigmoid(outputs_mask_ind1)
            valid_mask_ind_prob2 = F.sigmoid(outputs_mask_ind1)
            #loss = criterion(outputs, valid_mask)
            loss1 = criterion(outputs, valid_mask)
            #loss1 = F.binary_cross_entropy_with_logits(outputs, valid_mask)
            #loss2 = nn.BCEWithLogitsLoss()(outputs_mask_ind1, valid_mask_ind)
            # weight = torch.ones_like(valid_mask_ind)
            # weight[:, 0] = weight[:, 0] * 1
            # weight[:, 1] = weight[:, 1] * 14
            # weight[:, 2] = weight[:, 2] * 14
            # weight[:, 3] = weight[:, 3] * 4
            # weight[:, 4] = weight[:, 4] * 4
            # weight = weight * valid_mask_ind + 1
            # weight = weight.to(device).type(torch.cuda.FloatTensor)
            loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, valid_mask_ind)
            loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, valid_mask_ind)
            loss = loss1 * w1 + loss2 * w2 + loss3 * w3

            #losses1.append(loss1.item())
            #losses2.append(loss2.item())
            #jaccard += [get_jaccard(valid_mask, (valid_prob > 0).float()).item()]
            meter.add(valid_prob, valid_mask, valid_mask_ind_prob1, valid_mask_ind_prob2, valid_mask_ind,
                      loss1.item(), loss2.item(), loss3.item(), loss.item())

        valid_metrics = meter.value()
        epoch_time = time.time() - start_time
        valid_metrics['epoch_time'] = epoch_time

        ### be careful: image, mask, prob are variables
        ### if you return them directly, the memory will blow up
        #metrics = {'valid_loss1': valid_loss1, 'valid_loss2': valid_loss2, 'valid_jaccard': valid_jaccard,
         #          'valid_image':valid_image.data, 'valid_mask':valid_mask.data, 'valid_prob':valid_prob.data}
        valid_metrics['image'] = valid_image.data
        valid_metrics['mask'] = valid_mask.data
        valid_metrics['prob'] = valid_prob.data
    return valid_metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()



def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices