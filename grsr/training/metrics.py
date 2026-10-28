"""Metrics used by downstream dense-mask evaluation.

Scikit-image is optional: the M environment already provides SciPy, so a
small compatible connected-component fallback keeps the metric definition
reproducible without an extra binary dependency.
"""

import numpy as np
import torch

try:
    from skimage import measure  # type: ignore
except ImportError:  # pragma: no cover - used on the pointtool environment
    from scipy import ndimage

    class _Region:
        def __init__(self, area, centroid):
            self.area = area
            self.centroid = centroid

    class _MeasureFallback:
        @staticmethod
        def label(array, connectivity=2):
            structure = np.ones((3, 3), dtype=np.uint8)
            if connectivity == 1:
                structure = ndimage.generate_binary_structure(2, 1)
            labels, _ = ndimage.label(np.asarray(array).astype(bool), structure=structure)
            return labels

        @staticmethod
        def regionprops(labels):
            labels = np.asarray(labels)
            regions = []
            for index in range(1, int(labels.max()) + 1):
                coordinates = np.argwhere(labels == index)
                if coordinates.size:
                    regions.append(_Region(len(coordinates), coordinates.mean(axis=0)))
            return regions

    measure = _MeasureFallback()
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  # bins: number of discrete score thresholds on the ROC curve
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        n = self.bins + 1
        self.tp_arr   = np.zeros([n])
        self.pos_arr  = np.zeros([n])
        self.fp_arr   = np.zeros([n])
        self.neg_arr  = np.zeros([n])
        self.class_pos= np.zeros([n])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_pixels = 256 * 256
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            labelss = np.array((labels).cpu()).astype('int64') # P
            predits = np.squeeze(predits)
            labelss = np.squeeze(labelss)
            if predits.ndim != 2 or labelss.ndim != 2:
                raise ValueError(
                    f"PD_FA expects 2D masks, got pred {predits.shape}, label {labelss.shape}"
                )
            if predits.shape != labelss.shape:
                raise ValueError(
                    f"PD_FA shape mismatch: pred {predits.shape}, label {labelss.shape}"
                )
            self.image_pixels = int(predits.size)

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = [int(region.area) for region in coord_image]
            self.image_area_match = []
            self.distance_match = []
            matched_indices = set()

            # Match each ground-truth component to the nearest still-unmatched
            # prediction within the three-pixel centroid tolerance.  The old
            # implementation deleted items from ``coord_image`` while
            # iterating, which could skip components and also mishandle equal
            # area values.
            for region_label in coord_label:
                centroid_label = np.asarray(region_label.centroid)
                candidates = []
                for index, region_image in enumerate(coord_image):
                    if index in matched_indices:
                        continue
                    distance = float(np.linalg.norm(
                        np.asarray(region_image.centroid) - centroid_label
                    ))
                    if distance < 3:
                        candidates.append((distance, index))
                if candidates:
                    distance, index = min(candidates)
                    matched_indices.add(index)
                    self.distance_match.append(distance)
                    self.image_area_match.append(self.image_area_total[index])

            self.dismatch = [
                area for index, area in enumerate(self.image_area_total)
                if index not in matched_indices
            ]
            self.FA[iBin] += np.sum(self.dismatch)
            self.PD[iBin] += len(self.distance_match)

    def get(self,img_num):

        Final_FA =  self.FA / (self.image_pixels * img_num)
        Final_PD = np.divide(self.PD, self.target, out=np.zeros_like(self.PD), where=self.target > 0)

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])
        self.target = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
