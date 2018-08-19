from torchnet.meter import AUCMeter
import torch
import numpy as np


class AllInOneMeter(object):
    """
    All in one meter: AUC
    """

    def __init__(self):
        #super(AllInOneMeter, self).__init__()
        self.out1auc1 = AUCMeter()
        self.out1auc2 = AUCMeter()
        self.out1auc3 = AUCMeter()
        self.out1auc4 = AUCMeter()
        self.out1auc5 = AUCMeter()
        self.out2auc1 = AUCMeter()
        self.out2auc2 = AUCMeter()
        self.out2auc3 = AUCMeter()
        self.out2auc4 = AUCMeter()
        self.out2auc5 = AUCMeter()
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss = []
        self.jaccard = []
        #self.nbatch = 0
        self.epsilon = 1e-15
        self.intersection = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.reset()

    def reset(self):
        #self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        #self.targets = torch.LongTensor(torch.LongStorage()).numpy()
        self.out1auc1.reset()
        self.out1auc2.reset()
        self.out1auc3.reset()
        self.out1auc4.reset()
        self.out1auc5.reset()
        self.out2auc1.reset()
        self.out2auc2.reset()
        self.out2auc3.reset()
        self.out2auc4.reset()
        self.out2auc5.reset()
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss = []
        self.jaccard = []
        self.intersection = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([5], dtype=torch.float, device='cuda:0')
        #self.nbatch = 0


    def add(self, mask_prob, true_mask, mask_ind_prob1, mask_ind_prob2, true_mask_ind, loss1,loss2,loss3,loss):
        self.out1auc1.add(mask_ind_prob1[:, 0].data, true_mask_ind[:, 0].data)
        self.out1auc2.add(mask_ind_prob1[:, 1].data, true_mask_ind[:, 1].data)
        self.out1auc3.add(mask_ind_prob1[:, 2].data, true_mask_ind[:, 2].data)
        self.out1auc4.add(mask_ind_prob1[:, 3].data, true_mask_ind[:, 3].data)
        self.out1auc5.add(mask_ind_prob1[:, 4].data, true_mask_ind[:, 4].data)
        self.out2auc1.add(mask_ind_prob2[:, 0].data, true_mask_ind[:, 0].data)
        self.out2auc2.add(mask_ind_prob2[:, 1].data, true_mask_ind[:, 1].data)
        self.out2auc3.add(mask_ind_prob2[:, 2].data, true_mask_ind[:, 2].data)
        self.out2auc4.add(mask_ind_prob2[:, 3].data, true_mask_ind[:, 3].data)
        self.out2auc5.add(mask_ind_prob2[:, 4].data, true_mask_ind[:, 4].data)
        self.loss1.append(loss1)
        self.loss2.append(loss2)
        self.loss3.append(loss3)
        self.loss.append(loss)
        #self.nbatch += true_mask.shape[0]
        y_pred = (mask_prob>0.3).type(true_mask.dtype)
        y_true = true_mask
        self.intersection += (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.union += y_true.sum(dim=-2).sum(dim=-1).sum(dim=0) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=0)


    def value(self):
        jaccard_array = (self.intersection / (self.union - self.intersection + self.epsilon))
        #jaccard_array = jaccard_array.data.cpu().numpy()
        jaccard = jaccard_array.mean()
        metrics = {'out1auc1':self.out1auc1.value()[0], 'out1auc2':self.out1auc2.value()[0],
                   'out1auc3':self.out1auc3.value()[0], 'out1auc4':self.out1auc4.value()[0],
                   'out1auc5':self.out1auc5.value()[0],
                   'out2auc1': self.out2auc1.value()[0], 'out2auc2': self.out2auc2.value()[0],
                   'out2auc3': self.out2auc3.value()[0], 'out2auc4': self.out2auc4.value()[0],
                   'out2auc5': self.out2auc5.value()[0],
                   'loss1':np.mean(self.loss1), 'loss2':np.mean(self.loss2),
                   'loss3':np.mean(self.loss3), 'loss':np.mean(self.loss),
                   'jaccard':jaccard.item(), 'jaccard1':jaccard_array[0].item(),'jaccard2':jaccard_array[1].item(),
                   'jaccard3':jaccard_array[2].item(), 'jaccard4':jaccard_array[3].item(),'jaccard5':jaccard_array[4].item(),
                   }
        for mkey in metrics:
            metrics[mkey] = round(metrics[mkey], 4)
        return metrics