import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

os.chdir("/Users/jathiswarbhaskar/Desktop/chest_xray")

path="/Users/jathiswarbhaskar/Desktop/chest_xray/chest_xray"

dirs = os.listdir(path)

train = path + '/train/'
test = path + '/test/'
val = path + '/val/'

train_dir = os.listdir(train)

train_normal = train + 'NORMAL/'
train_pneumonia = train + 'PNEUMONIA/'

pneu_images   = glob(train_pneumonia + "*.jpeg")
normal_images = glob(train_normal + "*.jpeg")


training_generator=ImageDataGenerator(rescale=1/255)


train_generator=training_generator.flow_from_directory(train,target_size=(200,200),batch_size=4,class_mode='binary')


validation_dir= val
validation_generator=ImageDataGenerator(rescale=1/255)
val_generator=validation_generator.flow_from_directory(validation_dir,target_size=(200,200),batch_size=4,class_mode='binary')

test_dir= test
test_generator=ImageDataGenerator(rescale=1/255)
test_generator=test_generator.flow_from_directory(test_dir,target_size=(200,200),batch_size=32,class_mode='binary')





model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='binary_crossentropy',metrics=['acc'])

history = model.fit_generator(train_generator,validation_data=val_generator,epochs=30,verbose=1)


print("Accuracy : " , model.evaluate(test_generator)[1]*100 , "%")


