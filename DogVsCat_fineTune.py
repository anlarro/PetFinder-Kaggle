import time

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

# path to the model weights files.
top_model_weights_path = '../Data/keras/bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '../Data/dogscats/train'
validation_data_dir = '../Data/dogscats/validation'
nb_train_samples = 20000
nb_validation_samples = 5000
epochs = 50
batch_size = 50

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False,  input_shape = (img_width,img_height,3))
print('Base Model loaded.')
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers:
    print(layer, layer.trainable)

# Create the model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights(top_model_weights_path)

# Combine base and top
model = Sequential()
model.add(base_model)
model.add(top_model)

#Check the trainable layers
for layer in model.layers:
    print(layer, layer.trainable)

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

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

# fine-tune the model
start_time = time.time()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)
print("--- %s seconds ---" % (time.time() - start_time))

fineTune_model_weights_path = '../Data/keras/fineTune_model.h5'
model.save_weights(fineTune_model_weights_path)