import numpy as np
import imageio
import glob


def SimpleWeightComputation(labels_path, n_classes):
    """Compute weight in a basic way, Parity in Dataset"""
    lbl_pths = glob.glob(labels_path)
    # lbl_container = []
    pix_per_class = [0] * 12
    size = 0
    for lbl_pth in lbl_pths:
        lbl = imageio.imread(lbl_pth)
        lbl = np.array(lbl, dtype=np.int8)
        # lbl_container.append(lbl)
        for indx in range(lbl.shape[0]):
            for indy in range(lbl.shape[1]):
                pix_per_class[lbl[indx][indy]] += 1

        size = lbl.size

    pixel_per_dataset = size * len(lbl_pths)

    freq = np.asarray([(x / pixel_per_dataset) for x in pix_per_class])
    # print('Initialweights: ' + str(freq))
    return freq


def InvertSimpleWeightComputation(labels_path, n_classes):
    """Invert the parity accross the dataset for all classes"""
    freq = SimpleWeightComputation(labels_path, n_classes)

    invertweight = [1 - x for x in freq]

    # print('Initialweights: ' + str(invertweight))
    return invertweight


def SimpleMedianWeightComputation(labels_path, n_classes):
    """Compute the median of frequency of appearance on frequency"""
    freq = SimpleWeightComputation(labels_path, n_classes)
    weight = np.median(freq) / freq
    # print(weight)
    return weight
    # weight_inter = weight / weight.sum()
    # print(weight_inter)
    # print(weight_inter.sum())


def NormalizedSimpleMedianWeightComputation(labels_path, n_classes):
    """Normalized version ( all sum to 1 ) of median computation"""
    freq = SimpleWeightComputation(labels_path, n_classes)
    weight = np.median(freq) / freq

    weight = weight / weight.sum()

    # print(weight)
    return weight


def WeightComputation(labels_path, n_classes):
    """Compute the weight according to the appearance in dataset"""
    lbl_pths = glob.glob(labels_path)
    # lbl_container = []
    pix_per_class = [0] * 12
    size = 0
    n_im_cls = [0] * 12
    cls_checker = [x for x in range(n_classes)]
    for lbl_pth in lbl_pths:
        lbl = imageio.imread(lbl_pth)
        lbl = np.array(lbl, dtype=np.int8)

        is_present = [True if x in lbl else False for x in cls_checker]
        for ind in range(n_classes):
            if is_present[ind]:
                n_im_cls[ind] = n_im_cls[ind] + 1
        # lbl_container.append(lbl)
        for indx in range(lbl.shape[0]):
            for indy in range(lbl.shape[1]):
                pix_per_class[lbl[indx][indy]] += 1

        size = lbl.size

    pixels_per_im_cls = [x * size for x in n_im_cls]

    freq = np.asarray([(pix_per_class[x] / pixels_per_im_cls[x])
                       for x in range(len(pix_per_class))])
    # print('Initialweights: ' + str(freq))
    return freq


def InvertWeightComputation(labels_path, n_classes):
    """Invert the weight proportion for each classes ( 1 - X )"""
    weight = WeightComputation(labels_path, n_classes)

    invertweight = [1 - x for x in weight]

    # print('Initialweights: ' + str(invertweight))
    return invertweight


def WeightComputationMedian(labels_path, n_classes):
    """Median of appearance on appearance proportional appearance in image"""
    freq = WeightComputation(labels_path, n_classes)
    weight = np.median(freq) / freq
    return weight


def NormalizedWeightComputationMedian(labels_path, n_classes):
    """Normalized version of the WeightComputationMedian ( sum = 1 )"""
    freq = WeightComputation(labels_path, n_classes)
    weight = np.median(freq) / freq
    weight = weight / weight.sum()
    return weight
