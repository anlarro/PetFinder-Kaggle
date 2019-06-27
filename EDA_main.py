import os
print(os.listdir("../Data"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read data
train = pd.read_csv("../Data/train/train.csv")
test = pd.read_csv("../Data/test/test.csv")
colors = pd.read_csv("../Data/color_labels.csv")
breeds = pd.read_csv("../Data/breed_labels.csv")
states = pd.read_csv("../Data/state_labels.csv")

#Let's check the most popular names
train['Name'].value_counts().head(50)
#Apparently those Names with puppy or kitty words are adopted faster. When people look for puppy with the search option of petfinder,
# these profiles appear first
train['AdoptionSpeed'].where(train['Name'].str.contains('puppy',case=False)).value_counts().sort_index().plot('barh', color='blue')
plt.title('AdoptionSpeed for names containing puppy/puppies')

train['AdoptionSpeed'].where(train['Name'].str.contains('kitty',case=False)).value_counts().sort_index().plot('barh', color='red')
plt.title('AdoptionSpeed for names containing kitty/kitties')

train['AdoptionSpeed'].where(~train['Name'].str.contains('dog|cat|puppy|kitty',case=False, na=False)).value_counts().sort_index().plot('barh', color='green')
plt.title('AdoptionSpeed for names not containing dog|cat|pup|kit')

#PhotoAmt
sns.boxplot(x='AdoptionSpeed', y='PhotoAmt', data=train)

#Imaging Data
train_dir = '../Data/train_images/'
test_dir = '../Data/test_images/'

train_images = [train_dir + '{}'.format(i) for i in os.listdir(train_dir)] #training images
test_images = [test_dir + '{}'.format(i) for i in os.listdir(test_dir)] #testing images

import matplotlib.image as mpimg
for i,ima in enumerate(train_images[4:9]): #check some train images
    img = mpimg.imread(ima)
    plt.figure(ima)
    plt.imshow(img)
    plt.show()

for i, ima in enumerate(test_images[0:5]):  # check some test images
    img = mpimg.imread(ima)
    plt.figure(ima)
    plt.imshow(img)
    plt.show()