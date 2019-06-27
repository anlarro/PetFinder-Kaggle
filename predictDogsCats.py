import os
print(os.listdir("../Data"))

from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np
import pandas as pd
import math

# dimensions of our images
img_width, img_height = 150, 150
batch_size = 100

# build the VGG16 network
base_model = applications.VGG16(weights=None, include_top=False,  input_shape = (img_width,img_height,3))

#To load full_model weights (top layers different that the used for fine tuning)
# top_model = Sequential()
# top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(4096, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))
# model = Sequential()
# model.add(base_model)
# model.add(top_model)
# model.load_weights('../Data/keras/full_model.h5')

#To load fine_tune weights (top layers different than full_model, and fine tune has frozen layers
for layer in base_model.layers[:15]:
    layer.trainable = False
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
model = Sequential()
model.add(base_model)
model.add(top_model)
model.load_weights('../Data/keras/fineTune_model.h5')


#Predict training images as Dogs or Cats
train = pd.read_csv("../Data/train/train.csv")
train_pet_ids_labels = train[['PetID','Type']] #1=dog, 2=cat

test = pd.read_csv("../Data/test/test.csv")
test_pet_ids_labels = test[['PetID','Type']] #1=dog, 2=cat

#Imaging Data
train_dir = '../Data/train_images/train_images/'
test_dir = '../Data/test_images/test_images/'

##############
# predicting images
train_datagen = image.ImageDataGenerator(rescale=1. / 255)
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

i = 0
train_pred = []
train_prob = []
genTrain=train_datagen.flow_from_directory('../Data/train_images', target_size=(img_height, img_width), batch_size=batch_size, shuffle=False, class_mode=None)
train_images=genTrain.filenames
y_train = read_and_process_image(train_images, train_pet_ids_labels)
for batch in genTrain:
    pred = model.predict_classes(batch)
    prob = model.predict(batch)
    train_pred.extend(pred)
    train_prob.extend(prob)
    i+=1
    if i==math.ceil(len(train_images)/batch_size):
        break

i = 0
test_pred = []
test_prob = []
genTest=test_datagen.flow_from_directory('../Data/test_images', target_size=(img_height, img_width), batch_size=batch_size, shuffle=False, class_mode=None)
test_images=genTest.filenames
y_test = read_and_process_image(test_images, test_pet_ids_labels)
for batch in genTest:
    pred = model.predict_classes(batch)
    prob = model.predict(batch)
    test_pred.extend(pred)
    test_prob.extend(prob)
    i += 1
    if i == math.ceil(len(test_images) / batch_size):
        break

train_preds = [item[0] for item in train_pred]
test_preds = [item[0] for item in test_pred]
from sklearn.metrics import accuracy_score
print('Accuracy in train set =', accuracy_score(y_train, train_preds))
print('Accuracy in test set = ', accuracy_score(y_test, test_preds))

train_imageIDs = []
for im in train_images:
    path, file = os.path.split(im)
    train_imageIDs.append(file.split('.')[0])
train_probs = [item[0] for item in train_prob]
df_train = pd.DataFrame({'ImageID': train_imageIDs,
                         'Label': y_train,
                         'Pred': train_preds,
                         'Prob': train_probs})

test_imageIDs = []
for im in test_images:
    path, file = os.path.split(im)
    test_imageIDs.append(file.split('.')[0])
test_probs = [item[0] for item in test_prob]
df_test = pd.DataFrame({'ImageID': test_imageIDs,
                        'Label': y_test,
                        'Pred': test_preds,
                        'Prob': test_probs})


df_train.to_csv('../Data/train/train_probs.csv',index=False)
df_test.to_csv('../Data/test/test_probs.csv',index=False)



