from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy as np
seed =7
np.random.seed(seed)
#testing
(X_train,y_train),(X_test,y_test)=mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
plt.show()
##################################
#pre_process data
num_pixels=X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test=X_test.reshape(X_test.shape[0],num_pixels).astype('float32')
#Normalizing data
X_train=X_train/255
X_test=X_test/255
#using a one hot encoding for multi class
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]
def create_model():
    model=Sequential()
    model.add(Dense(num_pixels,input_dim=num_pixels,activation='relu'))
    model.add(Dense(num_classes,kernel_initializer='normal',activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
print("Creating model...............")
model=create_model()
print("Model created..............")
print("Training model.............")
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=21,batch_size=300,verbose=2)
print("Testing.....")
scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

