import torch
import numpy as np
from matplotlib import pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class AverageMeterList(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=4):
        self.length = length
        self.reset()

    def reset(self):
        self.val = np.zeros(self.length)
        self.avg = np.zeros(self.length)
        self.sum = np.zeros(self.length)
        self.count = np.zeros(self.length)

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
            n = n.cpu().numpy()
        else:
            val = np.array(val)
            n = np.array(n)
        assert val.shape[0] == self.length
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count.clip(1, None)

class FusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, output, label):
        length = output.shape[0]
        for i in range(length):
            self.matrix[output[i], label[i]] += 1

    def get_rec_per_class(self):
        rec = np.array(
            [
                self.matrix[i, i] / self.matrix[:, i].sum()
                for i in range(self.num_classes)
            ]
        )
        rec[np.isnan(rec)] = 0
        return rec

    def get_pre_per_class(self):
        pre = np.array(
            [
                self.matrix[i, i] / self.matrix[i, :].sum()
                for i in range(self.num_classes)
            ]
        )
        pre[np.isnan(pre)] = 0
        return pre

    def get_accuracy(self):
        acc = (
            np.sum([self.matrix[i, i] for i in range(self.num_classes)])
            / self.matrix.sum()
        )
        return acc

    def plot_confusion_matrix(self, normalize = False, cmap=plt.cm.Blues, save_path=None):

        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = self.matrix.T

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.arange(self.num_classes), yticklabels=np.arange(self.num_classes),
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        if self.num_classes < 10:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if save_path is None:
            return fig
        else:
            plt.savefig(save_path)


def accuracy(output, label):
    cnt = label.shape[0]
    true_count = (output == label).sum()
    now_accuracy = true_count / cnt
    return now_accuracy, cnt

def accuracy_shot(output, label, shot_cate):
    true_count = output == label
    
    true_few = true_count[shot_cate == 0].sum()
    true_medium = true_count[shot_cate == 1].sum()
    true_many = true_count[shot_cate == 2].sum()
    true_all = true_count.sum()
    
    count_few = (shot_cate == 0).sum().float()
    count_medium = (shot_cate == 1).sum().float()
    count_many = (shot_cate == 2).sum().float()
    count_all = label.shape[0]
    
    count = torch.tensor([count_few, count_medium, count_many, count_all])
    acc = torch.stack([true_few, true_medium, true_many, true_all]).float().to(shot_cate.device) / count.clamp(1, None)
    
    return acc, count


def accuracy_shot_perclass(output, label, shot_cate):
    shot_acc, shot_count = accuracy_shot(output, label, shot_cate)
    acc_perclass = []
    true_predict = label == output
    for l in label.unique():
        true_l = label == l
        true_l_predict = true_predict[true_l]
        # acc_perclass.append(true_l_predict.sum() / float(true_l.sum()) )
        try:
            acc_perclass.append(true_l_predict.sum() / max(1, (output == l).sum()))
        except RuntimeError:
            acc_perclass.append(torch.true_divide(true_l_predict.sum(), max(1, (output == l).sum())))

    # print(acc_perclass)
    acc_perclass = torch.stack(acc_perclass).float().mean()
    return shot_acc, shot_count, acc_perclass