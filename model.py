import csv
import cv2
import numpy as np
import ntpath
import random
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Convolution2D, merge, Input, Dropout, MaxPooling2D
from keras.backend import tf as ktf


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def import_data(csv_file, image_folder, images, mesurements):
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        for line in reader:
            lines.append(line)
        
    for line in lines[1:]:
        source_path = line[0]
        filename = path_leaf(source_path)
        #current_path = './my_training_data/IMG/' + filename
        current_path = image_folder + filename
        image = cv2.imread(current_path)
        #image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


images = []
measurements = []
import_data('../data_simulator/driving_log_good_driving_1.csv',
            '../data_simulator/IMG_good_driving_1/',
            images, measurements)
# artificially double the data of driving in the middle of the road
#images.extend(images)
#measurements.extend(measurements)


import_data('../data_simulator/driving_log_trajectory_rectification_1.csv',
            '../data_simulator/IMG_trajectory_rectification_1/',
            images, measurements)
import_data('../data_simulator/driving_log_trajectory_rectification_2.csv',
            '../data_simulator/IMG_trajectory_rectification_2/',
            images, measurements)
import_data('../data_simulator/driving_log_bridge_shadow.csv',
            '../data_simulator/IMG_bridge_shadow/',
            images, measurements)
import_data('../data_simulator/driving_log_bridge_turn.csv',
            '../data_simulator/IMG_bridge_turn/',
            images, measurements)



print('The size of mesurements is: ', len(measurements))
#data augmentation - flip image and measurement
for i in range(0,len(images)):
    image_flipped = np.fliplr(images[i][:][:])
    images.append(image_flipped)
    measurement_flipped = -measurements[i]
    measurements.append(measurement_flipped)
    

#print('The file name is: ', current_path)
X_train = np.array(images)
y_train = np.array(measurements)

print('Total number of training images: ', X_train.shape)

def Normalisation(img):
    out = img / 255.0 -0.5
    return out

def My_module(input_layer):
    # 1x1 filter
    conv_1x1 = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(input_layer)
    # 3x3 filter
    conv_3x3 = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(input_layer)#,subsample=(2,2)
    # 5x5 filter after 1x1 filter
    conv1_5x5 = Convolution2D(1, 5, 5, border_mode='same', activation='relu')(conv_1x1) #, strides = (2,2)
    # 3x3 filter after 1x1 filter
    conv1_3x3 = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(conv_1x1)
    
    output_layer = merge([conv_1x1,conv_3x3,conv1_3x3,conv1_5x5], mode='concat', concat_axis=1)
    return output_layer

def My_net():
    input_img = Input(shape=(160,320,3))
    crop_img = Cropping2D(cropping=((65,0),(0,0)), input_shape=(160,320,3))(input_img)
    norm_img = Lambda(Normalisation, output_shape=(95,320,3))(crop_img)
    resised_img = Lambda(lambda image: ktf.image.resize_images(crop_img, (47, 160)))(norm_img)
    # 1x1 filter
    conv1 = Convolution2D(1, 1, 1, border_mode='same', activation='relu', input_shape=(47, 160,3))(resised_img)
    
    module1 = My_module(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(module1)
    module2 = My_module(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(module2)#, strides = (2,2)
    module3 = My_module(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4,4))(module3)#, strides = (2,2)
    module4 = My_module(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(module4)
    #module5 = My_module(maxpool4)
    #maxpool5 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module5)
    
    #module6 = My_module(maxpool5)
    #maxpool6 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module6)
    
   # module7 = My_module(maxpool5)
    #maxpool7 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module7)
        
    flatten =  Flatten()(maxpool4) #input_shape=(95,320,3)
    #tensor.shape.eval()


    drop = Dropout(0.5)(flatten)
    #connected1 = Dense(1000)(flatten)
    out_put = Dense(1)(drop)
    
    model=Model(input=input_img,output=out_put)
    model.compile(loss='mse', optimizer='adam')
    return model
    
                      

model = My_net()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.summary()

model.save('model.h5')


#model = Sequential()
#model.add(Cropping2D(cropping=((65,0),(0,0)), input_shape=(160,320,3)))
#model.add(Lambda(lambda x: x / 255.0 -0.5))
#model.add(My_module())
#model.add(Flatten(input_shape=(95,320,3)))
#model.add(Dense(1))
#model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)




