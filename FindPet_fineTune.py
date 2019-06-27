import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import shutil
import math

import time
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils import to_categorical

#original_train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train/train/'
path = "../Data/"
original_train_dir = '../Data/train_images'
original_test_dir = '../Data/test_images'

train_images = [os.path.join(original_train_dir,'{}'.format(i)) for i in os.listdir(original_train_dir)] #training images
test_images = [os.path.join(original_test_dir,'{}'.format(i)) for i in os.listdir(original_test_dir)] #testing images

train = pd.read_csv(os.path.join(path,'train/train.csv'))

#Create new directory with classes->AdoptionSpeed
train_dir = '../Data/trainClasses'
# os.mkdir(train_dir)
# os.mkdir(os.path.join(train_dir, '0'))
# os.mkdir(os.path.join(train_dir, '1'))
# os.mkdir(os.path.join(train_dir, '2'))
# os.mkdir(os.path.join(train_dir, '3'))
# os.mkdir(os.path.join(train_dir, '4'))
#
# for im in train_images:
#     _, file = os.path.split(im)
#     pet_id = file.split('-')[0]
#
#     label = train.loc[np.where(train["PetID"] == pet_id)[0],'AdoptionSpeed'].item()
#     shutil.copyfile(im, os.path.join(train_dir, str(label), file))
#
test_dir = '../Data/testClasses'
# os.mkdir(test_dir)
# os.mkdir(os.path.join(test_dir,'test_images'))
#
# for im in test_images:
#     _, file = os.path.split(im)
#     pet_id = file.split('-')[0]
#     shutil.copyfile(im, os.path.join(test_dir, 'test_images', file))

# dimensions of our images.
img_width, img_height = 150, 150
epochs = 1
batch_size = 30
nb_train_samples = len(os.listdir(os.path.join(train_dir, '0')))\
                   +len(os.listdir(os.path.join(train_dir, '1')))\
                   +len(os.listdir(os.path.join(train_dir, '2')))\
                   +len(os.listdir(os.path.join(train_dir, '3')))\
                   +len(os.listdir(os.path.join(train_dir, '4')))

# Calculate bottleneck features
datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
model = applications.VGG16(weights='imagenet', include_top=False,  input_shape = (img_width,img_height,3))

generator = datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size,shuffle=False)

bottleneck_features_train = model.predict_generator(generator, math.ceil(nb_train_samples / batch_size))
train_data = np.array(bottleneck_features_train)
train_labels = [int(os.path.split(f)[0]) for f in generator.filenames]

########################################
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_squared_error'])

model.fit(train_data, to_categorical(train_labels),
          epochs=epochs,
          batch_size=batch_size, verbose=2)

top_model_weights_path = '../Data/keras/bottleneck_findPet_model.h5'
model.save_weights(top_model_weights_path)

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False,  input_shape = (img_width,img_height,3))

for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers:
    print(layer, layer.trainable)

# Create the model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

# # Combine base and top
model = Sequential()
model.add(base_model)
model.add(top_model)

model.compile(loss='mean_squared_error',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['mean_squared_error'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

# fine-tune the model
start_time = time.time()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(np.ceil(nb_train_samples / batch_size)),
    epochs=epochs, verbose=2)
print("--- %s seconds ---" % (time.time() - start_time))

#fineTune_model_weights_path = 'fineTune_model.h5'
#model.save_weights(fineTune_model_weights_path)




# Predict test
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

i = 0
ids = []
test_pred = []
gen = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=batch_size,
                                       shuffle=False, class_mode=None)
submission = pd.read_csv('../Data/test/test_probs.csv')

for batch in gen:
    pred = model.predict_classes(batch)
    test_pred.extend(pred)

    i += 1
    if i == math.ceil(len(submission['ImageID']) / batch_size):
        break

#test_preds = [item[0] for item in test_pred]

submission['PetID'] = ['{}'.format(s[s.find('/') + 1:s.rfind('.')]) for s in gen.filenames]
submission['AdoptionSpeed'] = test_pred
submission.to_csv("submission.csv", index=False)

if os.path.isdir(train_dir):
    shutil.rmtree(train_dir)
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)
