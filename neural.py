import keras
import cv2
import os
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Input,Reshape,Conv2DTranspose


ip="./colored"
op="./black"

color=[os.path.join(ip,f) for f in os.listdir(ip)]
black=[os.path.join(op,f) for f in os.listdir(op)]
latent_dim=256
#create encoder
t=Input(shape=(400,350,1),name="encoder_input")
model=t
#Allows us to create model only knowing the inputs 
model=Conv2D(64,kernel_size=3,activation='relu',strides=2)(model)
model=Conv2D(128,kernel_size=3,activation='relu',strides=2)(model)
model=Conv2D(256,kernel_size=3,activation='relu',strides=2)(model)

#Generate Latent Layer
model=Flatten()(model)
latent=Dense(latent_dim,name="latent_vector")(model)

# make encoder model
encoder=Model(t,latent,name="encoder_layers")
encoder.summary()

#encoder model

'''encoder=Sequential()
encoder.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(400,350,1),strides=2))
encoder.add(Conv2D(128,kernel_size=3,activation='relu',strides=2))
encoder.add(Conv2D(256,kernel_size=3,activation='relu',strides=2))
encoder.add(Flatten())
encoder.add(Dense(latent_dim,name="latent_vector"))

# make encoder model
#encoder=Model(t,latent,name="encoder_layers")
encoder.summary()


# Decoder Model
decoder=Sequential()
decoder.add(Dense(400*350*1))
decoder.add(Reshape((400,350,1)))
decoder.add(Conv2DTranspose(256,kernel_size=3,activation='relu',padding='same'))
decoder.add(Conv2DTranspose(128,kernel_size=3,activation='relu',padding='same'))
decoder.add(Conv2DTranspose(64,kernel_size=3,activation='relu',padding='same'))
decoder.add(Conv2DTranspose(itl[0].shape[-1],kernel_size=3,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output'))

decoder.summary()
pred=encoder.predict'''


# Decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
dec = Dense(400*350*1)(latent_inputs)
dec = Reshape((400,350,1))(dec)

dec = Conv2DTranspose(256,kernel_size=3,activation="relu",padding="same",strides=2)(dec)
dec = Conv2DTranspose(128,kernel_size=3,activation="relu",padding="same",strides=2)(dec)
dec = Conv2DTranspose(64,kernel_size=3,activation="relu",padding="same",strides=2)(dec)

output = Conv2DTranspose(filters=1,kernel_size=3,activation="sigmoid",padding="same",name="decoder_layers")(dec)

decoder = Model(latent_inputs,output,name="decoder_model")
decoder.summary()


# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(t, decoder(encoder(t)), name='autoencoder')
autoencoder.summary()

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(black,color)





