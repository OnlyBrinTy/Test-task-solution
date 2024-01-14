# Данный пример взят с https://medium.com/@roshankg96/transfer-learning-and-fine-tuning-model-using-vgg-16-90b5401e1ebd

import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

tf.config.list_physical_devices('GPU')

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.1
                                   )

test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.05)

train_df = pd.read_csv('data/pro_train.csv')
test_df = pd.read_csv('data/pro_test.csv')
train_df["category_id"] = train_df["category_id"].values.astype(str)
test_df["category_id"] = test_df["category_id"].values.astype(str)
train_df['filename'] = train_df.product_id.apply(lambda x: f"..\images\\train/{x}.jpg")
test_df['filename'] = test_df.product_id.apply(lambda x: f"..\images\\train/{x}.jpg")

training_set = train_datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=32,
    x_col='filename',
    y_col='category_id',
    subset="training",
    class_mode='categorical')

validation_set = train_datagen.flow_from_dataframe(
    train_df,
    target_size=(224, 224),
    batch_size=16,
    x_col='filename',
    y_col='category_id',
    subset="validation",
    class_mode='categorical')


STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
STEP_SIZE_TEST = validation_set.n // validation_set.batch_size

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_SIZE = [224, 224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(1024, activation='relu')(x)
prediction = Dense(874, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

from keras import optimizers

learning_rate = 5e-5
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

from datetime import datetime
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


checkpoint = ModelCheckpoint(filepath='mymodel.h5', verbose=1, save_best_only=True)

start = datetime.now()

history = model.fit(training_set,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=6, verbose='auto',
                    validation_data=validation_set,
                    validation_steps=STEP_SIZE_TEST)

duration = datetime.now() - start
print("Training completed in time: ", duration)

score = model.evaluate(validation_set)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

model.save('models/vgg16_model2.h5')

import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()
