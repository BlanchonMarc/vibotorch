import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import jaccard_similarity_score
from torchvision import transforms
from PIL import Image


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.counter = [None, None, None, None]  # TP FP TN FN

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                     lp.flatten(),
                                                     self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist)
        iu = iu / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,
                }, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class evaluation(object):
    def __init__(self, n_classes, lr, modelstr):
        self.n_classes = n_classes
        self.C = np.zeros((n_classes, n_classes))
        self.FalseP = []
        self.FalseN = []
        self.TrueP = []
        self.TrueN = []
        self.prec = []
        self.rec = []
        self.f1score = []
        self.jaccard = []
        self.overallAcc = []
        self.MeanAcc = []
        self.IoU = []

        self.saving_param = -100

        self.f = open("log.txt", "a")
        self.f.write("##################################################")
        self.f.write("Training " + modelstr)
        self.f.write("Leaning Rate" + str(lr))
        self.f.write("##################################################")

    def __call__(self, gt, pred):
        labels = np.asarray([x for x in range(self.n_classes)])
        self.C = confusion_matrix(gt.ravel(), pred.ravel(), labels=labels)

        FalseP = self.C.sum(axis=0) - np.diag(self.C)
        FalseN = self.C.sum(axis=1) - np.diag(self.C)
        TrueP = np.diag(self.C)
        self.FalseP.append(FalseP)
        self.FalseN.append(FalseN)
        self.TrueP.append(TrueP)
        self.TrueN.append(self.C.sum() - (FalseP + FalseN + TrueP))

        self.prec.append(precision_score(gt, pred, labels=labels,
                                         average='micro'))

        self.rec.append(recall_score(gt, pred, labels=labels, average='micro'))

        self.f1score.append(f1_score(gt, pred, labels=labels, average='micro'))

        self.jaccard.append(jaccard_similarity_score(gt, pred))

        self.overallAcc.append(np.diag(self.C).sum() / self.C.sum())

        self.MeanAcc.append(np.nanmean(np.diag(self.C) / self.C.sum(axis=1)))

        self.IoU.append(np.nanmean(np.diag(self.C) / (self.C.sum(axis=0) + self.C.sum(axis=1) - np.diag(self.C))))

    def estimate(self, epoch, max_epoch, model, optim):
        self.FalseP = np.mean(self.FalseP)
        self.FalseN = np.mean(self.FalseN)
        self.TrueP = np.mean(self.TrueP)
        self.TrueN = np.mean(self.TrueN)
        self.prec = np.mean(self.prec)
        self.rec = np.mean(self.rec)
        self.f1score = np.mean(self.f1score)
        self.jaccard = np.mean(self.jaccard)
        self.overallAcc = np.mean(self.overallAcc)
        self.MeanAcc = np.mean(self.MeanAcc)
        self.IoU = np.mean(self.IoU)

        self.f.write("Epoch [" + str(epoch + 1) + " / " + str(max_epoch) + "]")
        self.f.write("False Positive: " + str(self.FalseP))
        self.f.write("False Negative: " + str(self.FalseN))
        self.f.write("True Positive: " + str(self.TrueP))
        self.f.write("True Negative: " + str(self.TrueN))
        self.f.write("Precision: " + str(self.prec))
        self.f.write("Recall: " + str(self.rec))
        self.f.write("F1 Score: " + str(self.f1score))
        self.f.write("Jaccard: " + str(self.jaccard))
        self.f.write("Overall Accuracy: " + str(self.overallAcc))
        self.f.write("Mean Accuracy: " + str(self.MeanAcc))
        self.f.write("IoU: " + str(self.IoU))

        if self.IoU >= self.saving_param:
            self.saving_param = self.IoU

            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optim.state_dict(), }
            torch.save(state,
                       "{}_{}_best_model.pkl".format('segnet', 'Camvid'))



    def reset(self):
        self.C = np.zeros((self.n_classes, self.n_classes))
        self.FalseP = []
        self.FalseN = []
        self.TrueP = []
        self.TrueN = []
        self.prec = []
        self.rec = []
        self.f1score = []
        self.jaccard = []
        self.overallAcc = []
        self.MeanAcc = []
        self.IoU = []




if __name__ == '__main__':

    n_classes = 12
    scoring = evaluation(n_classes)
    first = torch.IntTensor(5, n_classes, 256, 256)
    first.random_(0, n_classes)

    second = torch.IntTensor(5, 256, 256)
    second.random_(0, n_classes)

    pred = first.max(1)[1].numpy()
    print(pred.shape)
    groundtruth = second.numpy()
    print(groundtruth.shape)

    a = scoring(groundtruth.ravel(), pred.ravel())
