import os
print(os.getcwd())

import time
import math

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import LearningRateScheduler

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '../Data/dogscats/train'
validation_data_dir = '../Data/dogscats/validation'
nb_train_samples = 20000
nb_validation_samples = 5000
epochs = 150
batch_size = 50

# build the VGG16 network
base_model = applications.VGG16(weights=None, include_top=False,  input_shape = (img_width,img_height,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(base_model)
model.add(top_model)

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 0.1
   epochs_drop = 50.0
   lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
   return lrate

# Train the model
start_time = time.time()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0),
              metrics=['accuracy'])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
history = model.fit_generator(train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size,
            callbacks=callbacks_list,verbose=2)
print("--- %s seconds ---" % (time.time() - start_time))

full_model_weights_path = '../Data/keras/full_model.h5'
model.save_weights(full_model_weights_path)