#evaluation of autoencoder
#plot ROC curve and calculate AUC_ROC score, jaccard similaruty score, f1 score, global accuracy, precificity, sensitivity and precision
from My_preprocessing import *
from glob import glob
from PIL import Image
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import os
import cv2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

batchsize = 1
input_shape = (584,568)


def batch(iterable, n=batchsize):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def read_input(path):
    x = np.array(Image.open(path))/255.
    return x


def read_gt(path):
    x = np.array(Image.open(path))
    return x[..., np.newaxis]/np.max(x)

if __name__ == '__main__':
    model_name = "baseline_unet_aug_do_0.1_activation_ReLU_"


    val_data = list(zip(sorted(glob('/home/stud1/PycharmProjects/my_model/DRIVE/test/images/*.tif')),
                          sorted(glob('/home/stud1/PycharmProjects/my_model/DRIVE/test/2nd_manual/*.gif')),
                        sorted(glob('/home/stud1/PycharmProjects/my_model/DRIVE/test/mask/*.gif'))))

    try:
        os.makedirs("../output/"+model_name+"test/", exist_ok=True)
    except:
        pass

    model = my_autoencoder()

    dataset_path = "./Save_Drive_dataset/"
    path = dataset_path +"thebesttry.hdf5"

    model.load_weights(path, by_name=True)

    gt_list = []
    pred_list = []

    for batch_files in tqdm(batch(val_data), total=len(val_data)//batchsize):

        imgs = [resize(read_input(image_path[0]), input_shape) for image_path in batch_files]
        seg = [read_gt(image_path[1]) for image_path in batch_files]
        mask = [read_gt(image_path[2]) for image_path in batch_files]

        imgs = np.array(imgs)
        imgs = rgb2gray(imgs)

        train_imgs_normalized = dataset_normalized(imgs)
        # MORPHOLOGICAL OPERATION
        morph_imgs = morph(train_imgs_normalized)
        # HISTOGRAM EQUALIZATION
        data_equalized = image_histogram_equalization(morph_imgs)
        # REDUCE TO 0-1 RANGE
        imgs = data_equalized / 255.

        pred = model.predict(imgs)

        pred_all = (pred)

        pred = np.clip(pred, 0, 1)

        for i, image_path in enumerate(batch_files):

            pred_ = pred[i, :, :, 0]

            pred_ = resize(pred_, (584, 565))

            mask_ = mask[i]
            gt_ = (seg[i]>0.5).astype(int)

            gt_flat = []
            pred_flat = []

            for p in range(pred_.shape[0]):
                for q in range(pred_.shape[1]):
                    if mask_[p,q]>0.5: # Inside the mask pixels only
                        gt_flat.append(gt_[p,q])
                        pred_flat.append(pred_[p,q])

            print(pred_.size, len(gt_list))

            gt_list += gt_flat
            pred_list += pred_flat

            pred_ = 255.*(pred_ - np.min(pred_))/(np.max(pred_)-np.min(pred_))

            image_base = image_path[0].split("/")[-1]

            cv2.imwrite("../output/"+model_name+"test/"+image_base, pred_)

    print(len(gt_list), len(pred_list))


fpr, tpr, thresholds = roc_curve((gt_list), pred_list)
auc_roc = roc_auc_score(gt_list, pred_list)
print("AUC ROC : ", auc_roc)
roc_curve = plt.figure()
plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % auc_roc)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig("ROC.png")

y_scores = np.asarray(pred_list)
threshold_confusion = 0.5
print("Confusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0

jaccard_index = jaccard_similarity_score(gt_list, y_pred, normalize=True)
print "\nJaccard similarity score: " +str(jaccard_index)

F1_score = f1_score(gt_list, y_pred, labels=None, average='binary', sample_weight=None)
print "\nF1 score (F-measure): " +str(F1_score)

confusion = confusion_matrix(gt_list, y_pred)
print confusion
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print "Global Accuracy: " +str(accuracy)
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print "Specificity: " +str(specificity)
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print "Sensitivity: " +str(sensitivity)
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print "Precision: " +str(precision)