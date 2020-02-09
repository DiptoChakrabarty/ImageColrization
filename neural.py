import keras
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D



ip="./colored"
op="./black"

itl=[os.path.join(ip,f) for f in os.listdir(ip)]
otl=[os.path.join(op,f) for f in os.listdir(op)]

#create encoder
model=Sequential()
model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(400,350,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation="relu"))
model.compile(loss="mse",optimizer="adam")
model.summary()
