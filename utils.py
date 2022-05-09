from PIL import Image
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
import torch

def LoadImage(name, path, rotation=0.0, flip=False, 
                cut_bottom=58,size=(256, 200)):
    """
    Loads image
    """
    img = Image.open(str(path)+"/"+str(name))
    img = np.array(img)
    seg = img[:-cut_bottom, 256:]
    img = img[:-cut_bottom, 0:256]

    for i in range(3):
        zimg = img[:,:,i]
        zimg = cv2.equalizeHist(zimg)
        img[:,:,i] = zimg

    img = Image.fromarray(img).resize(size)
    seg = Image.fromarray(seg).resize(size)
    img = img.rotate(rotation)
    seg = seg.rotate(rotation)

    img = np.array(img)
    seg = np.array(seg)

    if flip:
        img = img[:,::-1,:]
        seg = seg[:,::-1,:]

    #seg = np.round(seg/255.0)

    return img/255, seg/255


def CreateKMeans(num_clusters=15):
    files = os.listdir("data/cityscapes/train")[0:10]
    colors = []
    for file in files:
        img, seg = LoadImage(name=file, path="data/cityscapes/train")
        colors.append(seg.reshape(seg.shape[0]*seg.shape[1], 3))
    colors = np.array(colors)
    colors = colors.reshape((colors.shape[0]*colors.shape[1],3))
    km = KMeans(num_clusters)
    km.fit(colors)
    return km


def CheckAccuracy(loader, model, numLabels):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.permute(0, 3, 1, 2).float()
            y = y.permute(0, 3, 1, 2).float()

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_coeff = DiceCoefMulti(y, preds, numLabels)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_coeff/len(loader)}")
    model.train()


def DiceCoef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def DiceCoefMulti(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += DiceCoef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average