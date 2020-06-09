import os

# len(os.listdir(r'../input/covid-chest-xray/images'))    #357 images


metadata = '../input/covid-chest-xray/metadata.csv'

import pandas as pd

data = pd.read_csv(metadata)

pd.set_option('display.max_columns', 20)

# data.head(10)

# data['finding'].unique()             #we have 11 categories in which covid 19 is one of that

if not os.path.exists('/kaggle/working/DATASET'):
    os.mkdir('/kaggle/working/DATASET')
    print('dataset directory created')

# print(data.columns)

covid_images = []

non_covid_images = []

for i in range(len(data['finding'])):

    if data['finding'].iloc[i] == 'COVID-19':
        covid_images.append(data['filename'].iloc[i])
    else:
        non_covid_images.append(data['filename'].iloc[i])

# print(len(covid_images))                           #296
# print(len(non_covid_images))                        #76
if not os.path.exists('/kaggle/working/DATASET/COVID'):
    os.mkdir('/kaggle/working/DATASET/COVID')
    print('path created')

TARGET_DIR = '/kaggle/working/DATASET/COVID/'

IMAGE_PATH = '../input/covid-chest-xray/images'

import shutil

for filename in covid_images:
    try:

        path = os.path.join(IMAGE_PATH, filename)
        shutil.copy2(path, TARGET_DIR)
        print(f'copying {filename}')
    except Exception as e:
        print(e)
        pass

print(len(os.listdir(TARGET_DIR)))  # 275 images of covid xrays after removing corrupt images

# from another data set of kaggle , i am loading another set of normal images and covid images

DIR = '../input/covid19-xray-dataset-trian-test-sets/train'

if not os.path.exists('/kaggle/working/DATASET/NORMAL'):
    os.mkdir('/kaggle/working/DATASET/NORMAL/')
    # print('normal directory created')
DIRECTORY = '/kaggle/working/DATASET/NORMAL'

for filename in os.listdir('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL'):
    try:
        path = os.path.join('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL', filename)
        shutil.copy2(path, DIRECTORY)
        print(f'copying {filename}')
    except Exception as e:
        print(e)
        pass
# only 74 so let us import from test set just to train
print(len(os.listdir(DIRECTORY)))
for filename in os.listdir('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/NORMAL'):
    try:
        path = os.path.join('../input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test/NORMAL', filename)
        shutil.copy2(path, DIRECTORY)
        print(f'copying {filename}')
    except Exception as e:
        print(e)
        pass

print(len(os.listdir(DIRECTORY)))  # 94 normal images
print(len(os.listdir('/kaggle/working/DATASET/COVID/')))  # 275 covid images

# keep in mind it is a imbalanced data set

# now that i have prepared immbalanced data set
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

print(model.summary())

# datagen = image.ImageDataGenerator(rescale=1/.255,
# zoom_range=0.2,
# horizontal_flip=True,
# height_shift_range = 0.2,

# fill_mode='nearest')


# train_generator = datagen.flow_from_directory('/kaggle/working/DATASET',target_size=(224, 224),batch_size=32,class_mode='binary')

# model.fit_generator(train_generator,
# epochs=12,
# steps_per_epoch = 8)


# training it on kaggle
# it has done pretty well on the train data  accuracy of around 80%(exact 81%)
# stop running gpu if not using
