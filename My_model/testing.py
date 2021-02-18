#test the image segmentation on an image and show it
from My_preprocessing import *
import numpy as np
from keras import models
channels = 3
height = 584
width = 565
img = np.empty((1,height,width,channels))
original_test_image = '/home/stud1/PycharmProjects/my_model/DRIVE/test/images/08_test.tif'


img1 = Image.open(original_test_image)
img[0] = np.asarray(img1)
l = rgb2gray(img)
imgs_normalized = np.empty(l.shape)
imgs_std = np.std(l)
imgs_mean = np.mean(l)
imgs_normalized = (l-imgs_mean)/imgs_std
imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized) - np.min(imgs_normalized))) * 255
morph_imgs = morph(imgs_normalized)
l = morph_imgs/255.
model = my_autoencoder()
dataset_path = "./Save_Drive_dataset/"
path = dataset_path + "thebesttry.hdf5"
model.load_weights(path, by_name=True)
pred = model.predict(l)
print(pred.shape)

#for upsos in range(pred.shape[1]):
#    for mhkos in range(pred.shape[2]):
#        if pred[0,upsos,mhkos,0] < 0.05:
#            pred[0, upsos, mhkos, 0] = 0
#        if pred[0,upsos,mhkos,0] > 0.75:
#            pred[0, upsos, mhkos, 0] = 1



pred = np.clip(pred, 0, 1)
example = np.reshape(pred, (pred.shape[1], pred.shape[2]))
im = Image.fromarray((example * 255).astype(np.uint8))
im.show()
#Visualize each layer
layer_outputs = [layer.output for layer in model.layers[0:]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#activations = activation_model.predict(l)

