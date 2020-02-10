import keras
import cv2
import os
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Input



ip="./colored"
op="./black"

itl=[os.path.join(ip,f) for f in os.listdir(ip)]
otl=[os.path.join(op,f) for f in os.listdir(op)]
latent_dim=256
#create encoder
t=Input(shape=(400,350,1),name="encoder_input")
model=t
#Allows us to create model only knowing the inputs 
'''model=Conv2D(64,kernel_size=3,activation='relu',strides=2)(model)
model=Conv2D(128,kernel_size=3,activation='relu',strides=2)(model)
model=Conv2D(256,kernel_size=3,activation='relu',strides=2)(model)

#Generate Latent Layer
model=Flatten()(model)
latent=Dense(latent_dim,name="latent_vector")(model)

# make encoder model
encoder=Model(t,latent,name="encoder_layers")
encoder.summary()'''

#encoder model

encoder=Sequential()
encoder.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(400,350,1),strides=2))
encoder.add(Conv2D(128,kernel_size=3,activation='relu',strides=2))
encoder.add(Conv2D(256,kernel_size=3,activation='relu',strides=2))
encoder.add(Flatten())
encoder.add(Dense(latent_dim,name="latent_vector"))

# make encoder model
#encoder=Model(t,latent,name="encoder_layers")
encoder.summary()







