''' Pre-Processing Datasets Util
'''
import pandas as pd 
import numpy as np
from PIL import ImageFilter, Image
from sklearn.preprocessing import LabelBinarizer

def preprocessing_image(img):
    image = img.reshape(28,28).astype(np.uint8)
    image = Image.fromarray(image)
    return image.filter(ImageFilter.FIND_EDGES)

def preprocessing_dataset(data):
    label_binrizer = LabelBinarizer()
    labels = data['label']
    data.drop('label', axis = 1, inplace = True)
    images = [np.array(preprocessing_image(img)).flatten() for img in data.values] 
    labels = label_binrizer.fit_transform(labels)
    len_temp = len(images)
    # Duplicate the images with flip ones
    for i in range(len_temp):
        img = images[i].reshape(28,28)
        img = np.flip(img,1).flatten()
        images.append(img)
    # Duplicate the labels for flip ones
    labels_new = labels.tolist()
    for i in range(len_temp):
        labels_new.append(labels[i])
        
    labels_new = np.asarray(labels_new, dtype=np.int64)
    images_new = np.asarray(images, dtype=np.int64)
    return images_new, labels_new
