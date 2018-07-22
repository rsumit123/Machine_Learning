import os

from PIL import Image
import os.path
import numpy as np


def create_database1(files_dir,label_value):
    images=[]
    labels=[]
    for files in os.listdir(files_dir):
        im=Image.open(files_dir+'/'+files).convert('L')
        im=np.array(im.resize((100,100)))
        images.append(im)
        
    images=np.array(images)
    labels=[label_value]*images.shape[0]
    labels=np.array(labels)
    labels=labels.reshape(labels.shape[0],1)
    return images,labels
