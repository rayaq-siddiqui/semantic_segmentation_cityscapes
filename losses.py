from sklearn.utils.class_weight import compute_class_weight
import segmentation_models as sm
from utils import LoadImage
import numpy as np
import os


def comp_class_weights(km):
    files = os.listdir("data/cityscapes/train")[0:100]

    class_count = []
    for file in files:
        img, seg = LoadImage(file)
        cc = classCount(seg, km)
        class_count.append(cc)

    class_count = np.array(class_count)
    class_count = class_count.reshape((class_count.shape[0]*class_count.shape[1]))

    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_count), 
        y=class_count
    )

    return cw


def classCount(seg, km):
    s = seg.reshape((seg.shape[0]*seg.shape[1],3))
    s = km.predict(s)
    return s


def total_loss_fn(km=None):
    # class_weight = comp_class_weights(km)
    class_weight = [0.47913359, 0.25554373, 0.86917846, 0.25535485, 1.38005577,
        3.63383441, 2.06719598, 5.8991952 , 1.37632189, 6.4732284 ,
        6.97466915, 6.14008263, 2.44706197, 2.91630711, 2.9314096 ]

    # dice_loss = sm.losses.DiceLoss(class_weights=class_weight)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    return total_loss
