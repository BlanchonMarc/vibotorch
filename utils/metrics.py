import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import jaccard_similarity_score
from torchvision import transforms
from PIL import Image


class evaluation(object):
    """Object that allow computation and comparison of metrics"""
    def __init__(self, n_classes, lr, modelstr, textfile):
        """Initialization of confusion matrix and metrics"""
        self.n_classes = n_classes
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

        self.saving_param = -100

        counter = len(glob.glob1("trained_models/", "*.pkl"))
        self.textsave = "trained_models/" + "model" + str(counter) + " .pkl"

        with open(self.textsave, 'w'):
            pass
        with open(textfile, 'w'):
            pass
        self.f = open(textfile, "a")
        self.f.write("\n##################################################\n")
        self.f.write("Training " + modelstr + "\n")
        self.f.write("Leaning Rate " + str(lr) + "\n")
        self.f.write("##################################################\n")

    def __call__(self, gt, pred):
        """Compute all the metrics accordind to the two images given"""
        labels = np.asarray([x for x in range(self.n_classes)])
        self.C = confusion_matrix(gt, pred, labels=labels)

        FalseP = self.C.sum(axis=0) - np.diag(self.C)
        FalseN = self.C.sum(axis=1) - np.diag(self.C)
        TrueP = np.diag(self.C)
        self.FalseP.append(FalseP)
        self.FalseN.append(FalseN)
        self.TrueP.append(TrueP)
        self.TrueN.append(self.C.sum() - (FalseP + FalseN + TrueP))

        self.prec.append(precision_score(gt, pred, labels=labels,
                                         average='weighted'))

        self.rec.append(recall_score(gt, pred, labels=labels,
                                     average='weighted'))

        self.f1score.append(f1_score(gt, pred, labels=labels,
                                     average='weighted'))

        self.jaccard.append(jaccard_similarity_score(gt, pred, normalize=False))

        self.overallAcc.append(np.diag(self.C).sum() / self.C.sum())

        self.MeanAcc.append(np.nanmean(np.diag(self.C) / self.C.sum(axis=1)))

        self.IoU.append(np.nanmean(np.diag(self.C) / (
            self.C.sum(axis=0) + self.C.sum(axis=1) - np.diag(self.C))))

        self.C = np.zeros((self.n_classes, self.n_classes))

    def estimate(self, epoch, max_epoch, model, optim):
        """Estimation of the desired param and print in file all metrics"""
        self.FalseP = np.float32(np.mean(self.FalseP))
        self.FalseN = np.float32(np.mean(self.FalseN))
        self.TrueP = np.float32(np.mean(self.TrueP))
        self.TrueN = np.float32(np.mean(self.TrueN))
        self.prec = np.float32(np.mean(self.prec))
        self.rec = np.float32(np.mean(self.rec))
        self.f1score = np.float32(np.mean(self.f1score))
        self.jaccard = np.float32(np.mean(self.jaccard))
        self.overallAcc = np.float32(np.mean(self.overallAcc))
        self.MeanAcc = np.float32(np.mean(self.MeanAcc))
        self.IoU = np.float32(np.mean(self.IoU))

        self.f.write("Epoch [" + str(epoch + 1) + " / " + str(
            max_epoch) + "]\n")
        self.f.write("False Positive: " + str(self.FalseP) + "\n")
        self.f.write("False Negative: " + str(self.FalseN) + "\n")
        self.f.write("True Positive: " + str(self.TrueP) + "\n")
        self.f.write("True Negative: " + str(self.TrueN) + "\n")
        self.f.write("Precision: " + str(self.prec) + "\n")
        self.f.write("Recall: " + str(self.rec) + "\n")
        self.f.write("F1 Score: " + str(self.f1score) + "\n")
        self.f.write("Jaccard: " + str(self.jaccard) + "\n")
        self.f.write("Overall Accuracy: " + str(self.overallAcc) + "\n")
        self.f.write("Mean Accuracy: " + str(self.MeanAcc) + "\n")
        self.f.write("IoU: " + str(self.IoU) + "\n")
        self.f.write("##################################################\n")

        if self.IoU >= self.saving_param:
            self.saving_param = self.IoU

            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optim.state_dict(), }
            torch.save(state, self.textsave)

    def reset(self):
        """Reset the object parameters"""
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

    def close(self):
        """Close the openned file properly"""
        self.f.close()

    def print_major_metric(self):
        """Print the desired parameters in terminal"""
        print("[MIoU : %f ; F1 : %f]" % (self.IoU, self.f1score))
