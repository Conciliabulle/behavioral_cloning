import csv
import cv2
import numpy as np
import ntpath
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Convolution2D, merge, Input, Dropout, MaxPooling2D


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
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


images = []
measurements = []
import_data('../data_simulator/driving_log_good_driving_1.csv',
            '../data_simulator/IMG_good_driving_1/',
            images, measurements)
import_data('../data_simulator/driving_log_trajectory_rectification_1.csv',
            '../data_simulator/IMG_trajectory_rectification_1/',
            images, measurements)
import_data('../data_simulator/driving_log_trajectory_rectification_2.csv',
            '../data_simulator/IMG_trajectory_rectification_2/',
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
    conv_3x3 = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(input_layer)
    # 5x5 filter after 1x1 filter
    conv1_5x5 = Convolution2D(1, 5, 5, border_mode='same', activation='relu')(conv_1x1)
    # 3x3 filter after 1x1 filter
    conv1_3x3 = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(conv_1x1)
    
    output_layer = merge([conv_1x1,conv_3x3,conv1_3x3,conv1_5x5], mode='concat', concat_axis=1)
    return output_layer

def My_net():
    input_img = Input(shape=(160,320,3))
    crop_img = Cropping2D(cropping=((65,0),(0,0)), input_shape=(160,320,3))(input_img)
    norm_img = Lambda(Normalisation, output_shape=(95,320,3))(crop_img)
    module1 = My_module(norm_img)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(module1)
    module2 = My_module(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module2)
    module3 = My_module(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module3)
    module4 = My_module(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module4)
    module5 = My_module(maxpool4)
    maxpool5 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(module5)
        
    flatten =  Flatten()(maxpool3) #input_shape=(95,320,3)
    #tensor.shape.eval()


    drop = Dropout(0.5)(flatten)  
    out_put = Dense(1)(flatten)
    
    model=Model(input=input_img,output=out_put)
    model.compile(loss='mse', optimizer='adam')
    return model
    
                      

model = My_net()
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

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




