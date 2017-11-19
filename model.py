import csv
import cv2
import numpy as np
import ntpath
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

lines = []
#with open('./my_training_data/driving_log.csv') as csvfile:
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = path_leaf(source_path)
    #current_path = './my_training_data/IMG/' + filename
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    #image = image[65:,:]
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print('The file name is: ', filename)
X_train = np.array(images)
y_train = np.array(measurements)

print('Total number of training images: ', X_train.shape)
print('The size of an image is: ', image.shape)

model = Sequential()
model.add(Cropping2D(cropping=((65,0),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 -0.5))
model.add(Flatten(input_shape=(95,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

model.save('model.h5')




