#Preprocessing steps, define autoencoder and training
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import h5py
from PIL import Image
import cv2
from random import randint
import random
from skimage.transform import rotate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Dropout, LeakyReLU, initializers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import backend as K

def auc_roc(y_true, y_pred):
 value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
 metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
 for v in metric_vars:
    tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
 with tf.control_dependencies([update_op]):
  value = tf.identity(value)
 return value

def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return dice

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def my_autoencoder():
    initializer = initializers.he_uniform()

    input_img = Input(shape=(None,None, 1))

    x = Conv2D(8, (3, 3), activation='relu',kernel_initializer=initializer, padding='same')(input_img)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Conv2D(32, (3, 3), activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3),strides=(2,2),activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization()(x)


    x = Conv2D(256, (3, 3),  activation='relu', kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)


    x = Conv2D(512, (3,3), activation='relu', kernel_initializer=initializer ,padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Dropout(0.4)(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2DTranspose(512, (3,3), activation='relu',kernel_initializer=initializer, padding='same')(encoded)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Dropout(0.4)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(256, (3, 3), activation='relu', kernel_initializer=initializer,padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Conv2DTranspose(256, (3, 3), activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = Dropout(0.4)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2DTranspose(32, (3, 3), activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(32, (3,3), activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = (LeakyReLU(alpha=0.1))(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", metrics = ['acc',auc_roc],loss='binary_crossentropy')
    autoencoder.summary()
    return autoencoder

def morph(imgs):
    imgs_morph = np.empty(imgs.shape)
    kernel1 = np.ones((3,3), np.uint8)
    for i in range(imgs.shape[0]):
        image = cv2.erode(imgs[i],kernel1,iterations = 1)
        image = np.reshape(image,(imgs.shape[1],imgs.shape[2],1))
        imgs_morph[i] = image
    return imgs_morph

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)
    assert (rgb.shape[3]==3)
    bn_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1],rgb.shape[2],1))
    return bn_imgs

def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[3]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0)
    print ("ground truth images are correctly withih pixel value range 0-255 (black-white)")
    groundTruth = np.reshape(groundTruth,(Nimgs,height,width,1))
    assert(groundTruth.shape == (Nimgs,height,width,1))
    return imgs, groundTruth

def image_histogram_equalization(images, number_bins=256):
    hist_images = np.empty(images.shape)
    # get image histogram
    for i in range(images.shape[0]):
        image_histogram, bins = np.histogram(images[i].flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize
        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(images[i].flatten(), bins[:-1], cdf)
        hist_images[i] = image_equalized.reshape(images[i].shape)

    return hist_images
input_shape = (48,48)

def extract_patches(image_array, gt_array, number_of_patches):
    final_image_patches = np.empty((number_of_patches,input_shape[0],input_shape[1],1))
    final_gt_patches = np.empty(final_image_patches.shape)
    for i in range(number_of_patches-1):
        random_image = randint(0, image_array.shape[0]-1)
        img_patch, gt_patch = random_crop(image_array[random_image], gt_array[random_image])
        final_image_patches[i] = img_patch
        final_gt_patches[i] = gt_patch
    return final_image_patches,final_gt_patches


def random_crop(img, mask, crop_size=input_shape[0]):
    imgheight= img.shape[0]
    imgwidth = img.shape[1]
    i = randint(0, imgheight-crop_size)
    j = randint(0, imgwidth-crop_size)

    return img[i:(i+crop_size), j:(j+crop_size), :], mask[i:(i+crop_size), j:(j+crop_size)]

def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

if __name__ == '__main__':
#GET DATA

#train
 original_imgs_train = "/home/stud1/PycharmProjects/my_model/DRIVE/training/images/"
 groundTruth_imgs_train = "/home/stud1/PycharmProjects/my_model/DRIVE/training/1st_manual/"
 dataset_path = "./Save_Drive_dataset/"
 Nimgs = 20
 channels = 3
 height = 584
 width = 565
 imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,"train")
 #images=(20,584,565,3) and groundtruth=(20,584,565,1)
 #GRAYSCALE CONV
 train_imgs = rgb2gray(imgs_train)
 #images=(20,584,565,1)
 #NORMALIZATION
 train_imgs_normalized = dataset_normalized(train_imgs)
 #MORPHOLOGICAL OPERATION
 morph_imgs = morph(train_imgs_normalized)
 #HISTOGRAM EQUALIZATION
 data_equalized = image_histogram_equalization(morph_imgs)
 #REDUCE TO 0-1 RANGE
 train_imgs_normalized = data_equalized/255.
 groundTruth_train = groundTruth_train/255.
 write_hdf5(train_imgs_normalized, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
 write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
 #SHOW IMAGE AFTER PREPROCESSING
 #print(train_imgs_normalized.shape) #--> (20,584,565,1)
 #example = train_imgs_normalized[0]
 #example = np.reshape(example, (example.shape[0],example.shape[1]))
 #im = Image.fromarray((example * 255).astype(np.uint8))
 #im.show()

 #CREATE RANDOM PATCHES
 number_of_patches = 100000
 image_patches, groundTruth_patches = extract_patches(train_imgs_normalized,groundTruth_train,number_of_patches)
 print(image_patches.shape)
 #print(groundTruth_patches.shape)
 patch = image_patches[0]
 patch = np.reshape(patch,(patch.shape[0], patch.shape[1]))
 patch = Image.fromarray((patch * 255).astype(np.uint8))
 patch.show()

 patch1 = groundTruth_patches[0]
 patch1 = np.reshape(patch1,(patch1.shape[0], patch1.shape[1]))
 patch1 = Image.fromarray((patch1 * 255).astype(np.uint8))
 patch1.show()

 #AUGMENTATION

 transformations = {'rotate anticlockwise': anticlockwise_rotation,
                      'rotate clockwise': clockwise_rotation,
                      'horizontal flip': h_flip,
                      'vertical flip': v_flip,
                 }
 images_to_generate=200000
 i=1
 aug_images = []
 aug_gt = []
 while i <= images_to_generate:
    r = random.randint(0,len(image_patches)-1)
    original_image = image_patches[r]
    gt_original_image = groundTruth_patches[r]
    transformed_image = None
    gt_transformed_image = None
    n = 0
    transformation_count=0
    while n <= transformation_count:
        key = random.choice(list(transformations))  # randomly choosing method to call
        transformed_image = transformations[key](original_image)
        gt_transformed_image = transformations[key](gt_original_image)
        aug_images.append(transformed_image)
        aug_gt.append(gt_transformed_image)
        n = n + 1
    i = i + 1

 aug_images = np.array(aug_images)
 aug_gt = np.array(aug_gt)
 print(aug_gt.shape)
 #FINAL SHAPE (200000,48,48,1)

 #DEFINE THE MODEL & TRAIN.
 my_model = my_autoencoder()
 path = dataset_path + "weights_try.best.hdf5"
 checkpointer = ModelCheckpoint(filepath=dataset_path + "thebesttry1.hdf5", verbose=2, monitor="auc_roc",mode='max', save_best_only=True)
 #checkpointer with customize monitor
 history = my_model.fit(aug_images,aug_gt,validation_split=0.1, batch_size=8, verbose=2, shuffle=True, epochs=4,callbacks=[checkpointer])
 my_model.save_weights(path)
