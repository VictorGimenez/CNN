# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten


#Open dataset
datasetEEG_extended_topo = sio.loadmat("datasetEEG_extended_topo.mat")
dado_csv = pd.read_csv('datasetEEG_label_extended_topo.csv',header=None)

#prepare train and test sets
dado = datasetEEG_extended_topo["dataset2"]
datasetY = dado_csv.values


datasetX = np.stack(dado[:,0])
print(datasetX.shape)

imgplot = plt.imshow(datasetX[0])
plt.show() 

#normaliza pixel values
datasetY = datasetY.astype('float32')/ 255
datasetX = datasetX.astype('float32')/ 255

imgplot = plt.imshow(datasetX[0])
plt.show() 


X_train, X_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.33)


#CNN

# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(datasetX[0].shape)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# fit model
#model.fit(X_train, y_train, epochs=150, batch_size=128, verbose=0)
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)
