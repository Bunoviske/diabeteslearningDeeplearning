from sklearn.metrics import confusion_matrix
from fastai.basics import *

def acc_segmentation(input, target):
    void_code = 0
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

class IOU(AvgMetric):
    "Intersection over Union Metric"
    def __init__(self, class_index, class_label, axis, ignore_index=-1): store_attr('axis,class_index,class_label,ignore_index')
    def accumulate(self, learn):
        pred, targ = learn.pred.argmax(dim=self.axis), learn.y
        intersec = ((pred == targ) & (targ == self.class_index)).sum().item()
        union = (((pred == self.class_index) | (targ == self.class_index)) & (targ != self.ignore_index)).sum().item()
        if union: self.total += intersec
        self.count += union
  
    @property
    def name(self): return self.class_label


class MIOU(AvgMetric):
    "Mean Intersection over Union Metric"
    def __init__(self, classes_index, axis): store_attr()

    def accumulate(self, learn):
        pred, targ = learn.pred.argmax(dim=self.axis).cpu(), learn.y.cpu()
        pred, targ = pred.flatten().numpy(), targ.flatten().numpy()
        self.total += confusion_matrix(targ, pred, self.classes_index)

    @property
    def value(self): 
        conf_matrix = self.total
        per_class_TP = np.diagonal(conf_matrix).astype(float)
        per_class_FP = conf_matrix.sum(axis=0) - per_class_TP
        per_class_FN = conf_matrix.sum(axis=1) - per_class_TP
        iou_index = per_class_TP / (per_class_TP + per_class_FP + per_class_FN)
        iou_index = np.nan_to_num(iou_index)
        mean_iou_index = (np.mean(iou_index))    

        return mean_iou_index

    @property
    def name(self): return 'miou'