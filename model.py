import os
import math
import csv
import cv2
import numpy as np
import sklearn
import random
lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/TrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Define the generator
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.25)

def generator(samples, batch_size=45):
    correction_factor = 0.185
    
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
           batch_samples = samples[offset:offset+batch_size]
           images = []
           measurements = []
           for batch_sample in batch_samples:
              for i in range(3):   #Include the left and right image
                 source_path = batch_sample[i]
                 filename = source_path.split('\\')[-1]
                 current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/TrainingData/IMG/' + filename
                 image = cv2.imread(current_path)
                 if type(image).__module__=='numpy':
                  images.append(image)
                  measurement = float(line[3])+correction_factor*(i==1)-correction_factor*(i==2)  #Apply the corresponding correction factor to the left and right images
                  measurements.append(measurement)
           augmented_images, augmented_measurements = [], []
           for image, measurement in zip(images, measurements):  #Augmented the data by flipping the images
               augmented_images.append(image)
               augmented_measurements.append(measurement)
               augmented_images.append(cv2.flip(image,1))
               augmented_measurements.append(measurement*-1.0)
           X_train = np.array(images)
           y_train = np.array(measurements)
           yield sklearn.utils.shuffle(X_train, y_train)
batch_size=45  

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras import __version__ as keras_version
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Conv2D
from keras import regularizers

#NIVIDIA network
model = Sequential()
model.add(Lambda( lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(24, kernel_size = (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(36, kernel_size = (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(48, kernel_size = (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.005)))  #Applying the l2 regularizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=12, verbose=1)
model.save('model.h5')
    